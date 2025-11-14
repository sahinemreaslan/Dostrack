"""
DINOv3 model builders for OSTrack integration
"""
import os
from .dinov3_backbone import DINOv3Backbone
from lib.models.layers.lora import LoRAConfig


def _create_dinov3_backbone(variant, pretrained=False, frozen=True, use_lora=False,
                           lora_rank=4, lora_alpha=1.0, **kwargs):
    """Create DINOv3 backbone with specified configuration."""

    # Model configurations for different DINOv3 variants
    variant_configs = {
        'dinov3_vits16': {
            'img_size': 224,
            'patch_size': 16,  # DINOv3 uses 16x16 patches
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6,
        },
        'dinov3_vitb16': {
            'img_size': 224,
            'patch_size': 16,  # DINOv3 uses 16x16 patches
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
        },
        'dinov3_vitl16': {
            'img_size': 224,
            'patch_size': 16,  # DINOv3 uses 16x16 patches
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
        },
        'dinov3_vitg16': {
            'img_size': 224,
            'patch_size': 16,  # DINOv3 uses 16x16 patches
            'embed_dim': 1536,
            'depth': 40,
            'num_heads': 24,
        }
    }

    if variant not in variant_configs:
        raise ValueError(f"Unknown DINOv3 variant: {variant}")

    config = variant_configs[variant]
    config.update(kwargs)

    # Create LoRA config if needed
    lora_config = None
    if use_lora:
        lora_config = LoRAConfig(
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=["qkv", "proj"]  # Apply LoRA to attention layers
        )

    # Determine frozen stages
    frozen_stages = -1 if frozen else 0

    model = DINOv3Backbone(
        pretrained=pretrained if pretrained else None,
        frozen_stages=frozen_stages,
        use_lora=use_lora,
        lora_config=lora_config,
        **config
    )

    return model


def dinov3_vits16(pretrained=False, **kwargs):
    """DINOv3 Small model with 16x16 patches."""
    return _create_dinov3_backbone('dinov3_vits16', pretrained=pretrained, **kwargs)


def dinov3_vitb16(pretrained=False, **kwargs):
    """DINOv3 Base model with 16x16 patches."""
    return _create_dinov3_backbone('dinov3_vitb16', pretrained=pretrained, **kwargs)


def dinov3_vitl16(pretrained=False, **kwargs):
    """DINOv3 Large model with 16x16 patches."""
    return _create_dinov3_backbone('dinov3_vitl16', pretrained=pretrained, **kwargs)


def dinov3_vitg16(pretrained=False, **kwargs):
    """DINOv3 Giant model with 16x16 patches."""
    return _create_dinov3_backbone('dinov3_vitg16', pretrained=pretrained, **kwargs)