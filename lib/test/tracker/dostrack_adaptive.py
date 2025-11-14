"""
OSTrack with Adaptive Sizing
Enhanced version of OSTrack that dynamically adjusts template and search region sizes.
"""

import math
import time
from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.utils.adaptive_sizing import AdaptiveSizer, get_preset_sizer

# for debug
import cv2
import os


class OSTrackAdaptive(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrackAdaptive, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        # Initialize adaptive sizer
        self.adaptive_sizer = self._initialize_adaptive_sizer(params)

        # Dynamic feature size calculation
        self.current_search_size = getattr(params, 'search_size', 256)
        self.feat_sz = self.current_search_size // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # Performance tracking
        self.frame_times = []
        self.target_fps = getattr(params, 'target_fps', 30)

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug_adaptive"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)

        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

        # Size adaptation tracking
        self.size_history = []
        self.performance_scores = []

    def _initialize_adaptive_sizer(self, params):
        """Initialize the adaptive sizing system."""
        # Get sizing strategy from params
        sizing_strategy = getattr(params, 'sizing_strategy', 'balanced')

        if sizing_strategy in ['speed_optimized', 'quality_optimized', 'balanced', 'small_targets']:
            return get_preset_sizer(sizing_strategy)
        else:
            # Custom configuration
            template_factor = getattr(params, 'template_factor', 2.0)
            search_factor = getattr(params, 'search_factor', 4.0)
            strategy = getattr(params, 'adaptive_strategy', 'target_adaptive')

            from lib.utils.adaptive_sizing import create_adaptive_sizer
            return create_adaptive_sizer(
                strategy=strategy,
                template_factor=template_factor,
                search_factor=search_factor
            )

    def _update_output_window(self, search_size):
        """Update the output window based on current search size."""
        feat_sz = search_size // self.cfg.MODEL.BACKBONE.STRIDE
        if feat_sz != self.feat_sz:
            self.feat_sz = feat_sz
            self.output_window = hann2d(torch.tensor([feat_sz, feat_sz]).long(), centered=True).cuda()

    def initialize(self, image, info: dict):
        """Initialize tracker with adaptive template sizing."""
        start_time = time.time()

        # Compute adaptive sizes based on initial target
        template_size, search_size = self.adaptive_sizer.compute_adaptive_sizes(
            target_bbox=info['init_bbox'],
            image_shape=image.shape[:2]
        )

        # Log size decision
        if self.debug:
            target_scale = self.adaptive_sizer.compute_target_scale(info['init_bbox'])
            patch_info = self.adaptive_sizer.get_patch_counts(template_size, search_size)
            cost = self.adaptive_sizer.estimate_computational_cost(template_size, search_size)

            print(f"Adaptive Initialization:")
            print(f"  Target scale: {target_scale:.1f}")
            print(f"  Template size: {template_size} ({patch_info['template_patches']} patches)")
            print(f"  Search size: {search_size} ({patch_info['search_patches']} patches)")
            print(f"  Computational cost: {cost:.2f}x baseline")

        # Extract template with adaptive size
        template_factor = template_size / self.adaptive_sizer.compute_target_scale(info['init_bbox'])
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image, info['init_bbox'], template_factor, output_sz=template_size
        )

        self.z_patch_arr = z_patch_arr
        self.current_template_size = template_size
        self.current_search_size = search_size

        # Update output window for new search size
        self._update_output_window(search_size)

        # Process template
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        # Handle candidate elimination
        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # Save states
        self.state = info['init_bbox']
        self.frame_id = 0

        # Track initialization time
        init_time = time.time() - start_time
        self.frame_times.append(1.0 / init_time if init_time > 0 else float('inf'))

        if self.save_all_boxes:
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        """Track with adaptive search region sizing."""
        start_time = time.time()

        H, W, _ = image.shape
        self.frame_id += 1

        # Compute adaptive search size based on current state
        _, search_size = self.adaptive_sizer.compute_adaptive_sizes(
            target_bbox=self.state,
            image_shape=(H, W),
            frame_info={'frame_id': self.frame_id}
        )

        # Adapt search size based on performance if needed
        if hasattr(self, 'adaptive_performance') and self.adaptive_performance:
            if self.adaptive_sizer.should_adapt_for_speed(self.target_fps):
                search_size = int(search_size * 0.8)  # Reduce size for speed
            elif len(self.performance_scores) > 0 and np.mean(self.performance_scores[-5:]) < 0.7:
                search_size = int(search_size * 1.2)  # Increase size for quality

        # Ensure size is within bounds and properly aligned
        search_size = max(self.adaptive_sizer.config.min_search_size,
                         min(self.adaptive_sizer.config.max_search_size, search_size))
        search_size = self.adaptive_sizer.ensure_patch_alignment(search_size)

        # Update output window if search size changed
        if search_size != self.current_search_size:
            self.current_search_size = search_size
            self._update_output_window(search_size)

        # Extract search region with adaptive size
        search_factor = search_size / self.adaptive_sizer.compute_target_scale(self.state)
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, self.state, search_factor, output_sz=search_size
        )

        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        # Forward pass
        with torch.no_grad():
            x_dict = search
            out_dict = self.network.forward(
                template=self.z_dict1.tensors,
                search=x_dict.tensors,
                ce_template_mask=self.box_mask_z
            )

        # Process predictions
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)

        # Get final prediction
        pred_box = (pred_boxes.mean(dim=0) * search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # Track performance
        frame_time = time.time() - start_time
        fps = 1.0 / frame_time if frame_time > 0 else float('inf')
        self.frame_times.append(fps)

        # Update adaptive sizer with performance info
        self.adaptive_sizer.update_performance_history(fps)

        # Log performance periodically
        if self.debug and self.frame_id % 30 == 0:  # Every 30 frames
            recent_fps = sum(self.frame_times[-30:]) / min(30, len(self.frame_times))
            target_scale = self.adaptive_sizer.compute_target_scale(self.state)
            patch_info = self.adaptive_sizer.get_patch_counts(self.current_template_size, search_size)

            print(f"Frame {self.frame_id}: FPS={recent_fps:.1f}, "
                  f"Target scale={target_scale:.1f}, "
                  f"Search size={search_size}, "
                  f"Patches={patch_info['total_patches']}")

        # Debug visualization
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1+w), int(y1+h)), color=(0,0,255), thickness=2)

                # Add size info to image
                text = f"Search: {search_size}x{search_size}"
                cv2.putText(image_BGR, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')
                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')

        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(pred_boxes * search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()
            return {"target_bbox": self.state, "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        """Map box coordinates back to original image space."""
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.current_search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        """Map batch of box coordinates back to original image space."""
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.current_search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def get_adaptive_stats(self):
        """Get statistics about adaptive sizing performance."""
        if len(self.frame_times) < 2:
            return {}

        recent_fps = sum(self.frame_times[-30:]) / min(30, len(self.frame_times))
        avg_fps = sum(self.frame_times) / len(self.frame_times)

        return {
            "recent_fps": recent_fps,
            "average_fps": avg_fps,
            "current_template_size": self.current_template_size,
            "current_search_size": self.current_search_size,
            "total_frames": self.frame_id,
            "adaptive_config": str(self.adaptive_sizer.config)
        }