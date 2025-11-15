"""
DOSTrack - Dynamic Object Tracking with Adaptive Sizing
Addresses fundamental limitations of fixed template/search sizes:
- Small objects (30x30) get poor representation in large templates (128x128)
- Large objects may exceed fixed search regions
- Out-of-view scenarios require dynamic search expansion
"""
import math
import numpy as np

from lib.models.dostrack import build_dostrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond

# Adaptive utilities
from lib.utils.adaptive_utils import (
    AdaptiveSizeSelector,
    ConfidenceBasedExpansion,
    TemplateQualityAssessor,
    TemplateBank,
    MultiScaleSearch,
    compute_bbox_size
)


class DOSTrack(BaseTracker):
    """
    DOSTrack: Dynamic adaptive object tracker
    """
    def __init__(self, params, dataset_name):
        super(DOSTrack, self).__init__(params)
        network = build_dostrack(params.cfg, training=False)
        network.load_state_dict(
            torch.load(self.params.checkpoint, map_location='cpu', weights_only=False)['net'],
            strict=True
        )
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        # Adaptive sizing enabled?
        self.adaptive_enabled = getattr(self.cfg.TEST.ADAPTIVE, 'ENABLED', False)

        if self.adaptive_enabled:
            # Initialize adaptive modules
            self.size_selector = AdaptiveSizeSelector(self.cfg)
            self.conf_expansion = ConfidenceBasedExpansion(self.cfg)
            self.quality_assessor = TemplateQualityAssessor(self.cfg)
            self.template_bank = TemplateBank(self.cfg, device='cuda')
            self.multi_scale_search = MultiScaleSearch(self.cfg)

            # Tracking state
            self.current_template_size = self.params.template_size
            self.current_search_size = self.params.search_size
            self.current_search_factor = self.params.search_factor
            self.last_confidence = 1.0
            self.lost_count = 0
            self.max_lost_frames = 10
        else:
            # Use fixed sizes
            self.current_template_size = self.params.template_size
            self.current_search_size = self.params.search_size
            self.current_search_factor = self.params.search_factor

        # Feature size for output window (will be updated dynamically)
        self.feat_sz = self.current_search_size // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(
            torch.tensor([self.feat_sz, self.feat_sz]).long(),
            centered=True
        ).cuda()

        # Debug settings
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)

        # Save boxes
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        """Initialize tracker with first frame"""
        # Get initial bbox
        init_bbox = info['init_bbox']  # [x, y, w, h]

        if self.adaptive_enabled:
            # Select initial template size based on object size
            bbox_wh = (init_bbox[2], init_bbox[3])
            template_size, search_size, search_factor = self.size_selector.select_sizes(bbox_wh)

            self.current_template_size = template_size
            self.current_search_size = search_size
            self.current_search_factor = search_factor

            print(f"[DOSTrack] Adaptive sizing initialized:")
            print(f"  Object size: {bbox_wh[0]:.1f}x{bbox_wh[1]:.1f}")
            print(f"  Template size: {template_size}x{template_size}")
            print(f"  Search size: {search_size}x{search_size}")
            print(f"  Search factor: {search_factor:.2f}")

        # Extract template
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image,
            init_bbox,
            self.current_search_factor / 2.0,  # Template uses half of search factor
            output_sz=self.current_template_size
        )
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)

        with torch.no_grad():
            self.z_dict1 = template

        # Initialize template bank
        if self.adaptive_enabled:
            self.template_bank.initialize(template.tensors, frame_id=0)

        # Candidate elimination (if enabled)
        self.box_mask_z = None
        if hasattr(self.cfg.MODEL.BACKBONE, 'CE_LOC') and self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(
                init_bbox,
                resize_factor,
                template.tensors.device
            ).squeeze(1)
            self.box_mask_z = generate_mask_cond(
                self.cfg,
                1,
                template.tensors.device,
                template_bbox
            )

        # Update output window for current search size
        self._update_output_window(self.current_search_size)

        # Save initial state
        self.state = init_bbox
        self.frame_id = 0

        if self.save_all_boxes:
            all_boxes_save = init_bbox * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        """Track object in current frame"""
        H, W, _ = image.shape
        self.frame_id += 1

        # Adaptive search region sizing based on confidence
        if self.adaptive_enabled and self.last_confidence is not None:
            # Adjust search factor based on confidence
            adjusted_factor = self.conf_expansion.get_search_factor(
                self.last_confidence,
                self.current_search_factor
            )
        else:
            adjusted_factor = self.current_search_factor

        # Extract search region
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image,
            self.state,
            adjusted_factor,
            output_sz=self.current_search_size
        )
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        # Get best template from bank
        if self.adaptive_enabled:
            template_tensor = self.template_bank.get_best_template()
        else:
            template_tensor = self.z_dict1.tensors

        # Forward pass
        with torch.no_grad():
            out_dict = self.network.forward(
                template=template_tensor,
                search=search.tensors,
                ce_template_mask=self.box_mask_z
            )

        # Process predictions
        pred_score_map = out_dict['score_map']

        # Update output window if search size changed
        if pred_score_map.shape[-1] != self.feat_sz:
            self._update_output_window(self.current_search_size)

        # Apply hann window
        response = self.output_window * pred_score_map

        # Get bounding box prediction
        pred_boxes = self.network.box_head.cal_bbox(
            response,
            out_dict['size_map'],
            out_dict['offset_map']
        )
        pred_boxes = pred_boxes.view(-1, 4)

        # Compute confidence from score map
        max_score = response.max().item()
        mean_score = response.mean().item()
        confidence = max_score  # Use max score as confidence

        self.last_confidence = confidence

        # Check if object is lost
        if self.adaptive_enabled and self.multi_scale_search.is_lost(confidence):
            self.lost_count += 1
            print(f"[DOSTrack] Low confidence ({confidence:.3f}), lost count: {self.lost_count}")

            # Attempt multi-scale search
            if self.lost_count <= self.multi_scale_search.max_attempts:
                pred_box = self._multi_scale_redetection(image)
                if pred_box is not None:
                    print(f"[DOSTrack] Re-detected at scale {self.lost_count}")
                    self.lost_count = 0
                    confidence = 0.7  # Assume reasonable confidence after re-detection
                else:
                    # Use current prediction even with low confidence
                    pred_box = (pred_boxes.mean(dim=0) * self.current_search_size / resize_factor).tolist()
            else:
                # Too many lost frames, use current prediction
                pred_box = (pred_boxes.mean(dim=0) * self.current_search_size / resize_factor).tolist()
        else:
            # Normal tracking
            self.lost_count = 0
            pred_box = (pred_boxes.mean(dim=0) * self.current_search_size / resize_factor).tolist()

        # Map box back to original image coordinates
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # Adaptive template update
        if self.adaptive_enabled:
            self._update_template(image, confidence, pred_score_map)

        # Adaptive size adjustment based on predicted object size
        if self.adaptive_enabled and self.frame_id % 5 == 0:  # Check every 5 frames
            self._adjust_sizes()

        # Debug visualization
        if self.debug:
            self._visualize(image, x_patch_arr, pred_score_map, info, confidence)

        # Prepare output
        score_map_vis = pred_score_map.view(pred_score_map.shape[-2], pred_score_map.shape[-1]).cpu().numpy()

        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(
                pred_boxes * self.current_search_size / resize_factor,
                resize_factor
            )
            all_boxes_save = all_boxes.view(-1).tolist()
            return {
                "target_bbox": self.state,
                "all_boxes": all_boxes_save,
                "score_map": score_map_vis,
                "confidence": confidence
            }
        else:
            return {
                "target_bbox": self.state,
                "score_map": score_map_vis,
                "confidence": confidence
            }

    def _update_template(self, image, confidence, score_map):
        """Update template bank if quality is sufficient"""
        # Extract current template
        bbox_wh = (self.state[2], self.state[3])
        initial_wh = (
            self.template_bank.initial_template.shape[-1],
            self.template_bank.initial_template.shape[-2]
        )

        # Assess quality
        is_good, metrics = self.quality_assessor.assess_quality(
            confidence,
            bbox_wh,
            bbox_wh,  # Use current size as reference
            score_map
        )

        if is_good:
            # Extract new template
            z_patch_arr, _, z_amask_arr = sample_target(
                image,
                self.state,
                self.current_search_factor / 2.0,
                output_sz=self.current_template_size
            )
            template = self.preprocessor.process(z_patch_arr, z_amask_arr)

            # Update bank
            updated = self.template_bank.update(
                template.tensors,
                confidence,
                self.frame_id
            )

            if updated and self.debug:
                print(f"[DOSTrack] Template updated at frame {self.frame_id}, conf={confidence:.3f}")

    def _adjust_sizes(self):
        """Adjust template and search sizes based on current object size"""
        bbox_wh = (self.state[2], self.state[3])

        # Get new sizes
        new_template_size, new_search_size, new_search_factor = \
            self.size_selector.select_sizes(bbox_wh)

        # Update if sizes changed significantly
        if (abs(new_template_size - self.current_template_size) > 32 or
            abs(new_search_size - self.current_search_size) > 64):

            print(f"[DOSTrack] Adjusting sizes:")
            print(f"  Template: {self.current_template_size} -> {new_template_size}")
            print(f"  Search: {self.current_search_size} -> {new_search_size}")

            self.current_template_size = new_template_size
            self.current_search_size = new_search_size
            self.current_search_factor = new_search_factor

            # Update output window
            self._update_output_window(new_search_size)

    def _update_output_window(self, search_size):
        """Update Hann window for new search size"""
        self.feat_sz = search_size // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(
            torch.tensor([self.feat_sz, self.feat_sz]).long(),
            centered=True
        ).cuda()

    def _multi_scale_redetection(self, image):
        """Attempt to re-detect object at multiple scales"""
        scales = self.multi_scale_search.get_search_scales()

        best_box = None
        best_score = 0.0

        initial_template = self.template_bank.get_initial_template()

        for scale in scales:
            # Try searching at this scale
            x_patch_arr, resize_factor, x_amask_arr = sample_target(
                image,
                self.state,
                scale / self.current_template_size,
                output_sz=scale
            )
            search = self.preprocessor.process(x_patch_arr, x_amask_arr)

            with torch.no_grad():
                out_dict = self.network.forward(
                    template=initial_template,
                    search=search.tensors,
                    ce_template_mask=None
                )

            # Get score
            score_map = out_dict['score_map']
            max_score = score_map.max().item()

            if max_score > best_score:
                best_score = max_score
                # Get box prediction
                pred_boxes = self.network.box_head.cal_bbox(
                    score_map,
                    out_dict['size_map'],
                    out_dict['offset_map']
                )
                best_box = (pred_boxes.mean(dim=0) * scale / resize_factor).tolist()

        # Return best box if score is sufficient
        if best_score > 0.5:
            return best_box
        return None

    def _visualize(self, image, x_patch_arr, pred_score_map, info, confidence):
        """Debug visualization"""
        if not self.use_visdom:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw bbox
            cv2.rectangle(
                image_BGR,
                (int(x1), int(y1)),
                (int(x1+w), int(y1+h)),
                color=(0, 0, 255),
                thickness=2
            )

            # Add confidence text
            cv2.putText(
                image_BGR,
                f"Conf: {confidence:.3f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # Add size info
            cv2.putText(
                image_BGR,
                f"T:{self.current_template_size} S:{self.current_search_size}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        else:
            if info is not None:
                self.visdom.register(
                    (image, info['gt_bbox'].tolist(), self.state),
                    'Tracking',
                    1,
                    'Tracking'
                )

            self.visdom.register(
                torch.from_numpy(x_patch_arr).permute(2, 0, 1),
                'image',
                1,
                'search_region'
            )
            self.visdom.register(
                torch.from_numpy(self.z_patch_arr).permute(2, 0, 1),
                'image',
                1,
                'template'
            )
            self.visdom.register(
                pred_score_map.view(pred_score_map.shape[-2], pred_score_map.shape[-1]),
                'heatmap',
                1,
                'score_map'
            )

    def map_box_back(self, pred_box: list, resize_factor: float):
        """Map predicted box back to original image coordinates"""
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.current_search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        """Map batch of predicted boxes back to original image coordinates"""
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.current_search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return DOSTrack
