"""
Dynamic Sizing for OSTrack
Keeps search_factor and template_factor constant but dynamically adjusts resize dimensions.
"""

import math
import torch
import numpy as np


class DynamicSizer:
    """
    Dynamic sizing that maintains constant area factors but adjusts output dimensions.

    Principle:
    - template_factor = 2.0 (template area = 2x target area) - CONSTANT
    - search_factor = 4.0 (search area = 4x target area) - CONSTANT
    - output_sz = DYNAMIC based on target size
    """

    def __init__(
        self,
        template_factor: float = 2.0,
        search_factor: float = 4.0,
        min_output_size: int = 64,
        max_output_size: int = 512,
        patch_size: int = 14,
        size_step: int = 16,  # Quantization step
        base_target_size: int = 64,  # Reference target size for baseline 128/256
    ):
        self.template_factor = template_factor
        self.search_factor = search_factor
        self.min_output_size = min_output_size
        self.max_output_size = max_output_size
        self.patch_size = patch_size
        self.size_step = size_step
        self.base_target_size = base_target_size

    def compute_target_scale(self, target_bbox):
        """Compute target scale (geometric mean of width and height)."""
        if isinstance(target_bbox, torch.Tensor):
            w, h = target_bbox[2], target_bbox[3]
        else:
            w, h = target_bbox[2], target_bbox[3]
        return math.sqrt(w * h)

    def quantize_size(self, size):
        """Quantize size to multiples of size_step."""
        return int(math.ceil(size / self.size_step) * self.size_step)

    def ensure_patch_alignment(self, size):
        """Ensure size is compatible with patch size."""
        remainder = size % self.patch_size
        if remainder != 0:
            size = size + (self.patch_size - remainder)
        return size

    def compute_dynamic_sizes(self, target_bbox):
        """
        Compute dynamic output sizes while keeping factors constant.

        Args:
            target_bbox: [x, y, w, h] target bounding box

        Returns:
            tuple: (template_output_size, search_output_size)
        """
        target_scale = self.compute_target_scale(target_bbox)

        # Calculate base output sizes proportional to target scale
        # Baseline: 64px target → 128px template, 256px search
        scale_ratio = target_scale / self.base_target_size

        base_template_output = 128 * scale_ratio
        base_search_output = 256 * scale_ratio

        # Apply constraints
        template_output = max(self.min_output_size,
                            min(self.max_output_size, base_template_output))
        search_output = max(self.min_output_size,
                          min(self.max_output_size, base_search_output))

        # Ensure proper alignment and quantization
        template_output = self.ensure_patch_alignment(self.quantize_size(template_output))
        search_output = self.ensure_patch_alignment(self.quantize_size(search_output))

        return int(template_output), int(search_output)

    def get_factors(self):
        """Get the constant factors."""
        return self.template_factor, self.search_factor

    def get_patch_counts(self, template_size, search_size):
        """Get patch count information."""
        template_patches = (template_size // self.patch_size) ** 2
        search_patches = (search_size // self.patch_size) ** 2
        total_patches = template_patches + search_patches + 1  # +1 for CLS token

        return {
            "template_patches": template_patches,
            "search_patches": search_patches,
            "total_patches": total_patches,
            "template_size": template_size,
            "search_size": search_size
        }

    def estimate_speed_gain(self, template_size, search_size):
        """
        Estimate speed gain/loss compared to baseline (128, 256).

        Returns:
            float: Speed multiplier (>1 means faster, <1 means slower)
        """
        current_patches = self.get_patch_counts(template_size, search_size)["total_patches"]
        baseline_patches = self.get_patch_counts(128, 256)["total_patches"]

        # Transformer cost scales quadratically with sequence length
        cost_ratio = (current_patches / baseline_patches) ** 2
        speed_gain = 1.0 / cost_ratio

        return speed_gain


def create_dynamic_sizer(preset="balanced"):
    """Create dynamic sizer with preset configurations."""

    presets = {
        "speed": DynamicSizer(
            min_output_size=64,
            max_output_size=320,
            base_target_size=64
        ),
        "balanced": DynamicSizer(
            min_output_size=64,
            max_output_size=448,
            base_target_size=64
        ),
        "quality": DynamicSizer(
            min_output_size=96,
            max_output_size=576,
            base_target_size=64
        )
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

    return presets[preset]


# Example usage function
def demo_dynamic_sizing():
    """Demonstrate dynamic sizing with different target sizes."""
    sizer = create_dynamic_sizer("balanced")

    # Test with different target sizes
    test_targets = [
        [100, 100, 20, 20],    # Small target (20x20)
        [100, 100, 40, 60],    # Medium target (40x60)
        [100, 100, 80, 80],    # Large target (80x80)
        [100, 100, 150, 100],  # Very large target (150x100)
    ]

    print("Dynamic Sizing Demo:")
    print("Target Size | Template Out | Search Out | Patches | Speed Gain")
    print("-" * 65)

    for target_bbox in test_targets:
        target_scale = sizer.compute_target_scale(target_bbox)
        template_size, search_size = sizer.compute_dynamic_sizes(target_bbox)
        patch_info = sizer.get_patch_counts(template_size, search_size)
        speed_gain = sizer.estimate_speed_gain(template_size, search_size)

        print(f"{target_scale:10.1f} | {template_size:11} | {search_size:9} | "
              f"{patch_info['total_patches']:6} | {speed_gain:9.2f}x")


if __name__ == "__main__":
    demo_dynamic_sizing()