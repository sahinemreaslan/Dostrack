"""
OSTrack with Dynamic Sizing
Simplified approach: constant factors (2x, 4x) but dynamic output dimensions.
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
from lib.utils.dynamic_sizing import DynamicSizer, create_dynamic_sizer

# for debug
import cv2
import os
import numpy as np


class OSTrackDynamic(BaseTracker):
    """OSTrack with dynamic output sizing while keeping factors constant."""

    def __init__(self, params, dataset_name):
        super(OSTrackDynamic, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        # Initialize dynamic sizer
        sizing_preset = getattr(params, 'sizing_preset', 'balanced')
        self.dynamic_sizer = create_dynamic_sizer(sizing_preset)

        # Get constant factors
        self.template_factor, self.search_factor = self.dynamic_sizer.get_factors()

        # Current dynamic sizes (will be updated per frame)
        self.current_template_size = 128
        self.current_search_size = 256

        # Update output window for initial size
        self._update_output_window(self.current_search_size)

        # Performance tracking
        self.frame_times = []
        self.size_history = []

        # Debug settings
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug_dynamic"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)

        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

        if self.debug:
            print(f"Dynamic OSTrack initialized:")
            print(f"  Template factor: {self.template_factor}")
            print(f"  Search factor: {self.search_factor}")
            print(f"  Sizing preset: {sizing_preset}")

    def _update_output_window(self, search_size):
        """Update the output window based on current search size."""
        feat_sz = search_size // self.cfg.MODEL.BACKBONE.STRIDE
        self.feat_sz = feat_sz
        self.output_window = hann2d(torch.tensor([feat_sz, feat_sz]).long(), centered=True).cuda()

    def initialize(self, image, info: dict):
        """Initialize tracker with dynamic template sizing."""
        start_time = time.time()

        # Compute dynamic sizes for initialization
        template_size, search_size = self.dynamic_sizer.compute_dynamic_sizes(info['init_bbox'])

        # Store current sizes
        self.current_template_size = template_size
        self.current_search_size = search_size

        # Update output window for search size
        self._update_output_window(search_size)

        # Log initialization info
        if self.debug:
            target_scale = self.dynamic_sizer.compute_target_scale(info['init_bbox'])
            patch_info = self.dynamic_sizer.get_patch_counts(template_size, search_size)
            speed_gain = self.dynamic_sizer.estimate_speed_gain(template_size, search_size)

            print(f"\nInitialization:")
            print(f"  Target: {info['init_bbox']} (scale: {target_scale:.1f})")
            print(f"  Template: {template_size}x{template_size} ({patch_info['template_patches']} patches)")
            print(f"  Search: {search_size}x{search_size} ({patch_info['search_patches']} patches)")
            print(f"  Total patches: {patch_info['total_patches']}")
            print(f"  Speed gain: {speed_gain:.2f}x")

        # Extract template with dynamic size but constant factor
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image,
            info['init_bbox'],
            self.template_factor,  # Constant factor
            output_sz=template_size  # Dynamic size
        )

        self.z_patch_arr = z_patch_arr
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

        # Track timing
        init_time = time.time() - start_time
        self.frame_times.append(1.0 / init_time if init_time > 0 else float('inf'))

        # Track size decision
        self.size_history.append({
            'frame': 0,
            'target_scale': target_scale,
            'template_size': template_size,
            'search_size': search_size,
            'patches': patch_info['total_patches']
        })

        if self.save_all_boxes:
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        """Track with dynamic search region sizing."""
        start_time = time.time()

        H, W, _ = image.shape
        self.frame_id += 1

        # Compute dynamic search size based on current state
        _, search_size = self.dynamic_sizer.compute_dynamic_sizes(self.state)

        # Update output window if search size changed
        if search_size != self.current_search_size:
            self.current_search_size = search_size
            self._update_output_window(search_size)

        # Extract search region with constant factor but dynamic size
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image,
            self.state,
            self.search_factor,  # Constant factor
            output_sz=search_size  # Dynamic size
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

        # Track size decisions
        target_scale = self.dynamic_sizer.compute_target_scale(self.state)
        patch_info = self.dynamic_sizer.get_patch_counts(self.current_template_size, search_size)
        self.size_history.append({
            'frame': self.frame_id,
            'target_scale': target_scale,
            'template_size': self.current_template_size,
            'search_size': search_size,
            'patches': patch_info['total_patches']
        })

        # Debug output
        if self.debug and self.frame_id % 30 == 0:  # Every 30 frames
            recent_fps = np.mean(self.frame_times[-30:]) if len(self.frame_times) >= 30 else np.mean(self.frame_times)
            speed_gain = self.dynamic_sizer.estimate_speed_gain(self.current_template_size, search_size)

            print(f"Frame {self.frame_id}: FPS={recent_fps:.1f}, "
                  f"Target scale={target_scale:.1f}, "
                  f"Search={search_size}, "
                  f"Patches={patch_info['total_patches']}, "
                  f"Speed gain={speed_gain:.2f}x")

        # Debug visualization
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1+w), int(y1+h)), color=(0,0,255), thickness=2)

                # Add dynamic size info
                text1 = f"Template: {self.current_template_size}x{self.current_template_size}"
                text2 = f"Search: {search_size}x{search_size}"
                text3 = f"Target scale: {target_scale:.1f}"

                cv2.putText(image_BGR, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(image_BGR, text2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(image_BGR, text3, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

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

    def get_dynamic_stats(self):
        """Get statistics about dynamic sizing performance."""
        if len(self.frame_times) < 2:
            return {}

        recent_fps = np.mean(self.frame_times[-30:]) if len(self.frame_times) >= 30 else np.mean(self.frame_times)
        avg_fps = np.mean(self.frame_times)

        # Size statistics
        if len(self.size_history) > 0:
            recent_sizes = self.size_history[-30:] if len(self.size_history) >= 30 else self.size_history
            avg_template_size = np.mean([s['template_size'] for s in recent_sizes])
            avg_search_size = np.mean([s['search_size'] for s in recent_sizes])
            avg_patches = np.mean([s['patches'] for s in recent_sizes])
        else:
            avg_template_size = avg_search_size = avg_patches = 0

        return {
            "recent_fps": recent_fps,
            "average_fps": avg_fps,
            "current_template_size": self.current_template_size,
            "current_search_size": self.current_search_size,
            "avg_template_size": avg_template_size,
            "avg_search_size": avg_search_size,
            "avg_patches": avg_patches,
            "total_frames": self.frame_id,
            "template_factor": self.template_factor,
            "search_factor": self.search_factor
        }

    def print_final_stats(self):
        """Print final performance statistics."""
        stats = self.get_dynamic_stats()

        print(f"\n=== Dynamic OSTrack Performance ===")
        print(f"Total frames: {stats['total_frames']}")
        print(f"Average FPS: {stats['average_fps']:.2f}")
        print(f"Recent FPS: {stats['recent_fps']:.2f}")
        print(f"Template factor: {stats['template_factor']} (constant)")
        print(f"Search factor: {stats['search_factor']} (constant)")
        print(f"Average template size: {stats['avg_template_size']:.1f}")
        print(f"Average search size: {stats['avg_search_size']:.1f}")
        print(f"Average patches: {stats['avg_patches']:.1f}")

        if len(self.size_history) > 1:
            baseline_speed = self.dynamic_sizer.estimate_speed_gain(128, 256)
            avg_speed_gain = np.mean([
                self.dynamic_sizer.estimate_speed_gain(s['template_size'], s['search_size'])
                for s in self.size_history
            ])
            print(f"Average speed gain: {avg_speed_gain:.2f}x over baseline")
        print("=" * 35)