"""
Adaptive utilities for DOSTrack
Handles dynamic template and search region sizing based on object characteristics
"""
import torch
import numpy as np
from typing import Tuple, Optional, List


class AdaptiveSizeSelector:
    """
    Selects appropriate template and search sizes based on object dimensions.
    Addresses the fundamental problem where small objects (30x30) get poor representation
    in large templates (128x128), and large objects may exceed search regions.
    """

    def __init__(self, cfg):
        """
        Initialize adaptive size selector with configuration.

        Args:
            cfg: Configuration object containing ADAPTIVE settings
        """
        self.template_sizes = cfg.DATA.ADAPTIVE.TEMPLATE_SIZES  # [64, 128, 224, 384]
        self.search_factors = cfg.DATA.ADAPTIVE.SEARCH_FACTORS  # [3.0, 2.0, 2.0, 2.0]
        self.size_thresholds = cfg.DATA.ADAPTIVE.SIZE_THRESHOLDS  # [50, 150, 300]

    def select_sizes(self, bbox_wh: Tuple[float, float]) -> Tuple[int, int, float]:
        """
        Select template size, search size, and search factor based on object size.

        Args:
            bbox_wh: Tuple of (width, height) of the bounding box in pixels

        Returns:
            template_size: Selected template size
            search_size: Selected search region size
            search_factor: Search factor relative to template
        """
        w, h = bbox_wh
        obj_size = max(w, h)  # Use maximum dimension

        # Determine size category
        if obj_size < self.size_thresholds[0]:  # Small (< 50px)
            idx = 0
        elif obj_size < self.size_thresholds[1]:  # Medium (50-150px)
            idx = 1
        elif obj_size < self.size_thresholds[2]:  # Large (150-300px)
            idx = 2
        else:  # Very large (> 300px)
            idx = 3

        template_size = self.template_sizes[idx]
        search_factor = self.search_factors[idx]
        search_size = int(template_size * search_factor)

        return template_size, search_size, search_factor


class ConfidenceBasedExpansion:
    """
    Expands search region based on tracking confidence.
    Low confidence indicates potential object loss, requiring wider search.
    """

    def __init__(self, cfg):
        """
        Initialize confidence-based expansion.

        Args:
            cfg: Configuration object containing SEARCH_EXPANSION settings
        """
        self.high_conf_factor = cfg.DATA.ADAPTIVE.SEARCH_EXPANSION.HIGH_CONF_FACTOR  # 2.0
        self.med_conf_factor = cfg.DATA.ADAPTIVE.SEARCH_EXPANSION.MED_CONF_FACTOR   # 3.0
        self.low_conf_factor = cfg.DATA.ADAPTIVE.SEARCH_EXPANSION.LOW_CONF_FACTOR   # 5.0
        self.thresholds = cfg.DATA.ADAPTIVE.SEARCH_EXPANSION.CONFIDENCE_THRESHOLDS  # [0.5, 0.8]

    def get_search_factor(self, confidence: float, base_factor: float = 2.0) -> float:
        """
        Adjust search factor based on tracking confidence.

        Args:
            confidence: Tracking confidence score [0, 1]
            base_factor: Base search factor from size selector

        Returns:
            Adjusted search factor
        """
        if confidence > self.thresholds[1]:  # High confidence (> 0.8)
            return base_factor * self.high_conf_factor / 2.0
        elif confidence > self.thresholds[0]:  # Medium confidence (0.5 - 0.8)
            return base_factor * self.med_conf_factor / 2.0
        else:  # Low confidence (< 0.5)
            return base_factor * self.low_conf_factor / 2.0


class TemplateQualityAssessor:
    """
    Assesses template quality to decide whether to update.
    Prevents updating with low-quality frames (occluded, blurred, etc.)
    """

    def __init__(self, cfg):
        """
        Initialize quality assessor.

        Args:
            cfg: Configuration object
        """
        self.conf_threshold = cfg.DATA.ADAPTIVE.TEMPLATE_UPDATE.CONFIDENCE_THRESHOLD
        self.occlusion_threshold = cfg.MODEL.HEAD.QUALITY_ASSESSMENT.OCCLUSION_THRESHOLD
        self.size_var_threshold = cfg.MODEL.HEAD.QUALITY_ASSESSMENT.SIZE_VARIANCE_THRESHOLD

    def assess_quality(
        self,
        confidence: float,
        current_size: Tuple[float, float],
        reference_size: Tuple[float, float],
        score_map: Optional[torch.Tensor] = None
    ) -> Tuple[bool, dict]:
        """
        Assess if current frame is suitable for template update.

        Args:
            confidence: Prediction confidence
            current_size: Current bbox size (w, h)
            reference_size: Reference bbox size (w, h)
            score_map: Optional score map for occlusion detection

        Returns:
            is_good_quality: Boolean indicating if quality is sufficient
            metrics: Dictionary of quality metrics
        """
        metrics = {}

        # Check confidence
        conf_pass = confidence > self.conf_threshold
        metrics['confidence'] = confidence
        metrics['conf_pass'] = conf_pass

        # Check size variance (avoid updating during rapid size changes)
        w_ratio = current_size[0] / (reference_size[0] + 1e-6)
        h_ratio = current_size[1] / (reference_size[1] + 1e-6)
        size_variance = max(abs(1 - w_ratio), abs(1 - h_ratio))
        size_pass = size_variance < self.size_var_threshold
        metrics['size_variance'] = size_variance
        metrics['size_pass'] = size_pass

        # Estimate occlusion from score map (if available)
        occlusion_pass = True
        if score_map is not None:
            # Use score map entropy as occlusion indicator
            # High entropy suggests uncertain/occluded object
            score_map_flat = score_map.flatten()
            probs = torch.softmax(score_map_flat, dim=0)
            entropy = -(probs * torch.log(probs + 1e-6)).sum()

            # Normalize entropy (log of number of elements)
            max_entropy = np.log(score_map_flat.numel())
            normalized_entropy = entropy / max_entropy

            occlusion_score = normalized_entropy.item()
            occlusion_pass = occlusion_score < self.occlusion_threshold
            metrics['occlusion_score'] = occlusion_score
            metrics['occlusion_pass'] = occlusion_pass

        # Overall quality decision
        is_good_quality = conf_pass and size_pass and occlusion_pass
        metrics['overall_quality'] = is_good_quality

        return is_good_quality, metrics


class TemplateBank:
    """
    Maintains a bank of high-quality templates with exponential moving average.
    Provides robustness against temporary occlusions and quality degradation.
    """

    def __init__(self, cfg, device='cuda'):
        """
        Initialize template bank.

        Args:
            cfg: Configuration object
            device: Device to store templates
        """
        self.bank_size = cfg.DATA.ADAPTIVE.TEMPLATE_UPDATE.BANK_SIZE
        self.ema_alpha = cfg.DATA.ADAPTIVE.TEMPLATE_UPDATE.EMA_ALPHA
        self.update_interval = cfg.DATA.ADAPTIVE.TEMPLATE_UPDATE.UPDATE_INTERVAL
        self.device = device

        # Storage
        self.templates = []  # List of template tensors
        self.qualities = []  # List of quality scores
        self.frame_ids = []  # List of frame IDs

        # Tracking state
        self.primary_template = None
        self.initial_template = None
        self.frames_since_update = 0

    def initialize(self, template: torch.Tensor, frame_id: int = 0):
        """
        Initialize bank with first template.

        Args:
            template: Initial template tensor
            frame_id: Frame ID
        """
        self.initial_template = template.clone()
        self.primary_template = template.clone()
        self.templates = [template.clone()]
        self.qualities = [1.0]
        self.frame_ids = [frame_id]

    def update(self, template: torch.Tensor, quality: float, frame_id: int) -> bool:
        """
        Update template bank if conditions are met.

        Args:
            template: New template tensor
            quality: Quality score [0, 1]
            frame_id: Current frame ID

        Returns:
            updated: Whether update was performed
        """
        self.frames_since_update += 1

        # Check if update interval reached
        if self.frames_since_update < self.update_interval:
            return False

        # Update primary template with EMA
        self.primary_template = (
            self.ema_alpha * template +
            (1 - self.ema_alpha) * self.primary_template
        )

        # Add to bank
        self.templates.append(template.clone())
        self.qualities.append(quality)
        self.frame_ids.append(frame_id)

        # Maintain bank size (keep most recent and highest quality)
        if len(self.templates) > self.bank_size:
            # Remove oldest (except keep initial template)
            self.templates.pop(1)  # Keep index 0 as initial
            self.qualities.pop(1)
            self.frame_ids.pop(1)

        self.frames_since_update = 0
        return True

    def get_best_template(self) -> torch.Tensor:
        """
        Get the best template from bank.
        Uses primary template (EMA) for stability.

        Returns:
            Best template tensor
        """
        return self.primary_template

    def get_initial_template(self) -> torch.Tensor:
        """
        Get initial template for reference.

        Returns:
            Initial template tensor
        """
        return self.initial_template


class MultiScaleSearch:
    """
    Performs multi-scale pyramid search when object is lost.
    Implements re-detection mechanism for recovery.
    """

    def __init__(self, cfg):
        """
        Initialize multi-scale search.

        Args:
            cfg: Configuration object
        """
        self.enabled = cfg.TEST.ADAPTIVE.MULTI_SCALE_SEARCH.ENABLED
        self.lost_threshold = cfg.TEST.ADAPTIVE.MULTI_SCALE_SEARCH.LOST_THRESHOLD
        self.scales = cfg.TEST.ADAPTIVE.MULTI_SCALE_SEARCH.SCALES  # [384, 512, 768]
        self.max_attempts = cfg.TEST.ADAPTIVE.MULTI_SCALE_SEARCH.MAX_ATTEMPTS

    def is_lost(self, confidence: float) -> bool:
        """
        Determine if object is lost based on confidence.

        Args:
            confidence: Current tracking confidence

        Returns:
            True if object is considered lost
        """
        return confidence < self.lost_threshold

    def get_search_scales(self) -> List[int]:
        """
        Get pyramid search scales for re-detection.

        Returns:
            List of search sizes
        """
        return self.scales


def compute_bbox_size(bbox: np.ndarray) -> Tuple[float, float]:
    """
    Compute bounding box width and height.

    Args:
        bbox: Bounding box in format [x, y, w, h] or [cx, cy, w, h]

    Returns:
        Tuple of (width, height)
    """
    if len(bbox) == 4:
        return bbox[2], bbox[3]  # w, h
    else:
        raise ValueError(f"Invalid bbox format: {bbox}")


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute IoU between two bounding boxes.

    Args:
        bbox1: First bbox [x, y, w, h]
        bbox2: Second bbox [x, y, w, h]

    Returns:
        IoU score [0, 1]
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Convert to xyxy
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    # Intersection
    x_inter1 = max(x1, x2)
    y_inter1 = max(y1, y2)
    x_inter2 = min(x1_max, x2_max)
    y_inter2 = min(y1_max, y2_max)

    inter_area = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)

    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-6)
