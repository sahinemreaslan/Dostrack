"""
Adaptive Template and Search Region Sizing for OSTrack
Dynamically adjusts template and search region sizes based on target characteristics.
"""

import math
import torch
import numpy as np


class AdaptiveSizingConfig:
    """Configuration for adaptive sizing strategy."""

    def __init__(
        self,
        # Size ranges (pixels)
        min_template_size: int = 64,
        max_template_size: int = 320,
        min_search_size: int = 128,
        max_search_size: int = 640,

        # Scaling factors
        template_factor: float = 2.0,  # Template = target_size * template_factor
        search_factor: float = 4.0,    # Search = target_size * search_factor

        # Size quantization (for efficiency)
        size_step: int = 16,  # Quantize sizes to multiples of 16 (for better memory alignment)

        # Patch size (from backbone)
        patch_size: int = 14,  # DINOv3 uses 14x14 patches

        # Adaptive strategies
        strategy: str = "target_adaptive",  # "target_adaptive", "speed_adaptive", "quality_adaptive"

        # Speed/Quality trade-offs
        speed_priority: float = 0.5,  # 0.0 = quality first, 1.0 = speed first
    ):
        self.min_template_size = min_template_size
        self.max_template_size = max_template_size
        self.min_search_size = min_search_size
        self.max_search_size = max_search_size

        self.template_factor = template_factor
        self.search_factor = search_factor

        self.size_step = size_step
        self.patch_size = patch_size
        self.strategy = strategy
        self.speed_priority = speed_priority

    def __repr__(self):
        return f"AdaptiveSizingConfig(strategy={self.strategy}, template_factor={self.template_factor}, search_factor={self.search_factor})"


class AdaptiveSizer:
    """
    Adaptive sizing system that adjusts template and search region sizes
    based on target characteristics and performance requirements.
    """

    def __init__(self, config: AdaptiveSizingConfig = None):
        self.config = config or AdaptiveSizingConfig()

        # Performance tracking for adaptive decisions
        self.performance_history = []
        self.timing_history = []

    def compute_target_scale(self, target_bbox):
        """
        Compute target scale metric from bounding box.

        Args:
            target_bbox: [x, y, w, h] bounding box

        Returns:
            float: Target scale (geometric mean of width and height)
        """
        if isinstance(target_bbox, torch.Tensor):
            w, h = target_bbox[2], target_bbox[3]
        elif isinstance(target_bbox, (list, tuple)):
            w, h = target_bbox[2], target_bbox[3]
        else:
            w, h = target_bbox[2], target_bbox[3]

        return math.sqrt(w * h)

    def quantize_size(self, size):
        """Quantize size to multiples of size_step for efficiency."""
        return int(math.ceil(size / self.config.size_step) * self.config.size_step)

    def ensure_patch_alignment(self, size):
        """Ensure size is compatible with patch size for optimal performance."""
        # Make sure size is divisible by patch_size for clean tokenization
        remainder = size % self.config.patch_size
        if remainder != 0:
            size = size + (self.config.patch_size - remainder)
        return size

    def compute_adaptive_sizes(self, target_bbox, image_shape=None, frame_info=None):
        """
        Compute adaptive template and search sizes based on target characteristics.

        Args:
            target_bbox: [x, y, w, h] target bounding box
            image_shape: (H, W) of the image (optional)
            frame_info: dict with additional frame information (optional)

        Returns:
            tuple: (template_size, search_size) both as integers
        """
        target_scale = self.compute_target_scale(target_bbox)

        if self.config.strategy == "target_adaptive":
            return self._target_adaptive_sizing(target_scale, target_bbox, image_shape)
        elif self.config.strategy == "speed_adaptive":
            return self._speed_adaptive_sizing(target_scale, frame_info)
        elif self.config.strategy == "quality_adaptive":
            return self._quality_adaptive_sizing(target_scale, target_bbox)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

    def _target_adaptive_sizing(self, target_scale, target_bbox, image_shape):
        """Size regions based on target characteristics."""
        # Base sizes from target scale
        base_template_size = target_scale * self.config.template_factor
        base_search_size = target_scale * self.config.search_factor

        # Apply constraints
        template_size = max(self.config.min_template_size,
                          min(self.config.max_template_size, base_template_size))
        search_size = max(self.config.min_search_size,
                        min(self.config.max_search_size, base_search_size))

        # Special handling for very small or very large targets
        if target_scale < 32:  # Very small target
            template_size = max(template_size, 96)  # Ensure minimum context
            search_size = max(search_size, 192)
        elif target_scale > 200:  # Very large target
            # For large targets, we might want to reduce relative context
            template_size = min(template_size, target_scale * 1.5)
            search_size = min(search_size, target_scale * 2.5)

        # Apply quantization and alignment
        template_size = self.ensure_patch_alignment(self.quantize_size(template_size))
        search_size = self.ensure_patch_alignment(self.quantize_size(search_size))

        return int(template_size), int(search_size)

    def _speed_adaptive_sizing(self, target_scale, frame_info):
        """Size regions with speed priority."""
        # Compute base sizes
        base_template, base_search = self._target_adaptive_sizing(target_scale, None, None)

        # Apply speed reduction factor
        speed_factor = 1.0 - self.config.speed_priority * 0.5  # Reduce by up to 50%

        template_size = max(self.config.min_template_size, int(base_template * speed_factor))
        search_size = max(self.config.min_search_size, int(base_search * speed_factor))

        # Ensure alignment
        template_size = self.ensure_patch_alignment(template_size)
        search_size = self.ensure_patch_alignment(search_size)

        return template_size, search_size

    def _quality_adaptive_sizing(self, target_scale, target_bbox):
        """Size regions with quality priority."""
        # Compute base sizes
        base_template, base_search = self._target_adaptive_sizing(target_scale, target_bbox, None)

        # Apply quality enhancement factor
        quality_factor = 1.0 + (1.0 - self.config.speed_priority) * 0.3  # Increase by up to 30%

        template_size = min(self.config.max_template_size, int(base_template * quality_factor))
        search_size = min(self.config.max_search_size, int(base_search * quality_factor))

        # Ensure alignment
        template_size = self.ensure_patch_alignment(template_size)
        search_size = self.ensure_patch_alignment(search_size)

        return template_size, search_size

    def get_patch_counts(self, template_size, search_size):
        """
        Compute number of patches for given sizes.

        Returns:
            dict: Information about patch counts and total tokens
        """
        template_patches = (template_size // self.config.patch_size) ** 2
        search_patches = (search_size // self.config.patch_size) ** 2
        total_patches = template_patches + search_patches + 1  # +1 for CLS token

        return {
            "template_patches": template_patches,
            "search_patches": search_patches,
            "total_patches": total_patches,
            "template_size": template_size,
            "search_size": search_size
        }

    def estimate_computational_cost(self, template_size, search_size):
        """
        Estimate relative computational cost for given sizes.

        Returns:
            float: Relative cost compared to baseline (128, 256)
        """
        patch_info = self.get_patch_counts(template_size, search_size)
        current_patches = patch_info["total_patches"]

        # Baseline: 128x128 template + 256x256 search with 14x14 patches
        baseline_template_patches = (128 // 14) ** 2  # 81
        baseline_search_patches = (256 // 14) ** 2    # 324
        baseline_patches = baseline_template_patches + baseline_search_patches + 1  # 406

        # Transformer cost scales quadratically with sequence length
        relative_cost = (current_patches / baseline_patches) ** 2

        return relative_cost

    def update_performance_history(self, fps, accuracy_score=None):
        """Update performance history for adaptive decisions."""
        self.timing_history.append(fps)
        if accuracy_score is not None:
            self.performance_history.append(accuracy_score)

        # Keep only recent history (last 100 frames)
        if len(self.timing_history) > 100:
            self.timing_history = self.timing_history[-100:]
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def should_adapt_for_speed(self, target_fps=30):
        """Check if we should reduce sizes for speed."""
        if len(self.timing_history) < 10:
            return False

        recent_fps = np.mean(self.timing_history[-10:])
        return recent_fps < target_fps * 0.8  # If 20% below target

    def should_adapt_for_quality(self, min_accuracy=0.8):
        """Check if we should increase sizes for quality."""
        if len(self.performance_history) < 10:
            return False

        recent_accuracy = np.mean(self.performance_history[-10:])
        return recent_accuracy < min_accuracy


def create_adaptive_sizer(strategy="target_adaptive", template_factor=2.0, search_factor=4.0, **kwargs):
    """
    Factory function to create adaptive sizer with common configurations.

    Args:
        strategy: Sizing strategy ("target_adaptive", "speed_adaptive", "quality_adaptive")
        template_factor: Template size factor relative to target
        search_factor: Search size factor relative to target
        **kwargs: Additional configuration parameters

    Returns:
        AdaptiveSizer: Configured adaptive sizer
    """
    config = AdaptiveSizingConfig(
        strategy=strategy,
        template_factor=template_factor,
        search_factor=search_factor,
        **kwargs
    )
    return AdaptiveSizer(config)


# Predefined configurations for common use cases
CONFIGS = {
    "speed_optimized": AdaptiveSizingConfig(
        strategy="speed_adaptive",
        template_factor=1.5,
        search_factor=3.0,
        max_template_size=224,
        max_search_size=448,
        speed_priority=0.8
    ),

    "quality_optimized": AdaptiveSizingConfig(
        strategy="quality_adaptive",
        template_factor=2.5,
        search_factor=5.0,
        max_template_size=384,
        max_search_size=768,
        speed_priority=0.2
    ),

    "balanced": AdaptiveSizingConfig(
        strategy="target_adaptive",
        template_factor=2.0,
        search_factor=4.0,
        max_template_size=320,
        max_search_size=640,
        speed_priority=0.5
    ),

    "small_targets": AdaptiveSizingConfig(
        strategy="target_adaptive",
        template_factor=3.0,  # More context for small targets
        search_factor=6.0,
        min_template_size=96,
        min_search_size=192,
        speed_priority=0.3
    )
}


def get_preset_sizer(preset_name):
    """Get a preconfigured adaptive sizer."""
    if preset_name not in CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(CONFIGS.keys())}")

    return AdaptiveSizer(CONFIGS[preset_name])