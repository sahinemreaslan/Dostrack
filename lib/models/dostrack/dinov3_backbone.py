"""
DINOv3 backbone adapter for DOSTrack
Integrates DINOv3 Vision Transformer with DOSTrack framework
"""
import math
import sys
import os
from typing import Tuple, Optional
import torch
import torch.nn as nn
from functools import partial

# Add DINOv3 to path
dinov3_path = os.path.join(os.path.dirname(__file__), '../../../../../dinov3')
if dinov3_path not in sys.path:
    sys.path.insert(0, dinov3_path)

try:
    from dinov3.models.vision_transformer import DinoVisionTransformer
    from dinov3.layers import RopePositionEmbedding
except ImportError as e:
    print(f"Warning: Could not import DINOv3 components: {e}")
    DinoVisionTransformer = None
    RopePositionEmbedding = None

from lib.models.dostrack.base_backbone import BaseBackbone
from lib.models.layers.lora import apply_lora_to_model, LoRAConfig, print_lora_info


class DINOv3Backbone(BaseBackbone):
    """
    DINOv3 Vision Transformer backbone adapted for DOSTrack.

    This class wraps the DINOv3 model to provide an interface compatible
    with DOSTrack's tracking framework while maintaining the powerful
    features of DINOv3 including RoPE positional embeddings and
    self-supervised learning capabilities.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        pretrained: Optional[str] = None,
        frozen_stages: int = -1,  # -1 means freeze all stages except head
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        use_rope: bool = True,
        rope_base: float = 100.0,
        rope_dtype: str = "bf16",
        use_lora: bool = False,
        lora_config: Optional[LoRAConfig] = None,
        **kwargs
    ):
        """
        Initialize DINOv3 backbone.

        Args:
            img_size: Input image size
            patch_size: Patch size for tokenization
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            ffn_ratio: MLP expansion ratio
            qkv_bias: Whether to use bias in QKV projection
            drop_path_rate: Stochastic depth rate
            pretrained: Path to pretrained model weights
            frozen_stages: Number of stages to freeze (-1 for all)
            norm_layer: Normalization layer type
            ffn_layer: FFN layer type
            use_rope: Whether to use RoPE positional embeddings
            rope_base: Base frequency for RoPE
            rope_dtype: Data type for RoPE computations
            use_lora: Whether to apply LoRA for parameter efficient fine-tuning
            lora_config: LoRA configuration object
        """
        super().__init__()

        if DinoVisionTransformer is None:
            raise ImportError("DINOv3 not found. Please install DINOv3 properly.")

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.frozen_stages = frozen_stages
        self.use_lora = use_lora

        # Initialize DINOv3 model
        self.dinov3 = DinoVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ffn_layer=ffn_layer,
            pos_embed_rope_base=rope_base,
            pos_embed_rope_dtype=rope_dtype,
            **kwargs
        )

        # Initialize weights
        self.dinov3.init_weights()

        # Load pretrained weights if provided
        if pretrained:
            self.load_pretrained(pretrained)

        # Apply LoRA if requested
        if use_lora:
            if lora_config is None:
                lora_config = LoRAConfig()
            self._apply_lora(lora_config)
            print_lora_info(self.dinov3)

        # Freeze specified stages
        self._freeze_stages()

    def load_pretrained(self, pretrained_path: str):
        """Load pretrained DINOv3 weights."""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')

            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Load with strict=False to handle potential mismatches
            missing_keys, unexpected_keys = self.dinov3.load_state_dict(
                state_dict, strict=False
            )

            if missing_keys:
                print(f"Missing keys in pretrained model: {missing_keys}")
            if unexpected_keys:
                # Filter out known safe unexpected keys
                safe_keys = ['storage_tokens']
                safe_patterns = ['ls1.gamma', 'ls2.gamma', 'bias_mask']

                dangerous_keys = []
                for key in unexpected_keys:
                    if key not in safe_keys and not any(pattern in key for pattern in safe_patterns):
                        dangerous_keys.append(key)

                if dangerous_keys:
                    print(f"Unexpected keys in pretrained model: {dangerous_keys}")
                # Ignore safe unexpected keys silently

            print(f"Successfully loaded pretrained DINOv3 from: {pretrained_path}")

        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")

    def _apply_lora(self, lora_config: LoRAConfig):
        """Apply LoRA to the DINOv3 model."""
        print(f"Applying LoRA with config: {lora_config}")

        # Apply LoRA to the model
        self.dinov3 = apply_lora_to_model(
            self.dinov3,
            rank=lora_config.rank,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
            exclude_modules=lora_config.exclude_modules
        )

        # Only LoRA parameters should be trainable
        for name, param in self.dinov3.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _freeze_stages(self):
        """Freeze specified stages of the backbone."""
        if self.frozen_stages >= 0:
            # Freeze patch embedding
            self.dinov3.patch_embed.eval()
            for param in self.dinov3.patch_embed.parameters():
                param.requires_grad = False

            # Freeze positional embeddings
            if hasattr(self.dinov3, 'rope_embed'):
                for param in self.dinov3.rope_embed.parameters():
                    param.requires_grad = False

            # Freeze cls token
            self.dinov3.cls_token.requires_grad = False

            # Freeze transformer blocks
            if self.frozen_stages == -1:
                # Freeze all blocks
                for block in self.dinov3.blocks:
                    block.eval()
                    for param in block.parameters():
                        param.requires_grad = False
            else:
                # Freeze specified number of blocks
                for i in range(min(self.frozen_stages, len(self.dinov3.blocks))):
                    self.dinov3.blocks[i].eval()
                    for param in self.dinov3.blocks[i].parameters():
                        param.requires_grad = False

    def train(self, mode: bool = True):
        """Override train mode to respect frozen stages."""
        super().train(mode)

        if self.frozen_stages >= 0:
            # Keep patch embedding in eval mode
            self.dinov3.patch_embed.eval()

            # Keep frozen blocks in eval mode
            if self.frozen_stages == -1:
                for block in self.dinov3.blocks:
                    block.eval()
            else:
                for i in range(min(self.frozen_stages, len(self.dinov3.blocks))):
                    self.dinov3.blocks[i].eval()

        return self

    def finetune_track(self, cfg, patch_start_index=1):
        """
        Override finetune_track for DINOv3 which uses RoPE instead of learned positional embeddings.
        Most of the positional embedding logic is not needed for DINOv3.
        """
        # Set basic tracking parameters
        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.return_stage = cfg.MODEL.RETURN_STAGES
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG

        # Calculate feature map sizes for tracking
        search_size = cfg.DATA.SEARCH.SIZE
        template_size = cfg.DATA.TEMPLATE.SIZE
        stride = cfg.MODEL.BACKBONE.STRIDE

        self.search_feat_size = search_size // stride
        self.template_feat_size = template_size // stride

        # DINOv3 uses RoPE, so no need for learned positional embeddings
        # The model automatically handles different input sizes through RoPE

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using DINOv3."""
        # Use DINOv3's forward_features method
        output = self.dinov3.forward_features(x, masks=None)

        # Extract normalized tokens
        cls_token = output["x_norm_clstoken"]  # (B, C)
        patch_tokens = output["x_norm_patchtokens"]  # (B, H*W, C)

        # Combine cls token and patch tokens for OSTrack compatibility
        cls_token = cls_token.unsqueeze(1)  # (B, 1, C)
        x = torch.cat([cls_token, patch_tokens], dim=1)  # (B, 1+H*W, C)

        return x

    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        ce_template_mask: Optional[torch.Tensor] = None,
        ce_keep_rate: Optional[float] = None,
        return_last_attn: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass compatible with DOSTrack interface.

        Args:
            z: Template image tensor (B, C, H, W)
            x: Search image tensor (B, C, H, W)
            ce_template_mask: Candidate elimination mask for template
            ce_keep_rate: Keep rate for candidate elimination
            return_last_attn: Whether to return attention weights

        Returns:
            Combined features and auxiliary dict
        """
        # Process template and search images
        z_feat = self.forward_features(z)  # (B, 1+Hz*Wz, C)
        x_feat = self.forward_features(x)  # (B, 1+Hx*Wx, C)

        # Remove cls tokens for concatenation
        z_patches = z_feat[:, 1:]  # (B, Hz*Wz, C)
        x_patches = x_feat[:, 1:]  # (B, Hx*Wx, C)

        # Concatenate template and search features
        combined_feat = torch.cat([z_patches, x_patches], dim=1)

        # Auxiliary dictionary for compatibility
        aux_dict = {}
        if return_last_attn:
            # TODO: Implement attention extraction if needed
            aux_dict['attn'] = None

        return combined_feat, aux_dict


def build_dinov3_backbone(
    model_name: str = "dinov3_vits14",
    pretrained: Optional[str] = None,
    frozen: bool = True,
    use_lora: bool = False,
    lora_rank: int = 4,
    lora_alpha: float = 1.0,
    **kwargs
) -> DINOv3Backbone:
    """
    Build DINOv3 backbone with predefined configurations.

    Args:
        model_name: Name of the DINOv3 model variant
        pretrained: Path to pretrained weights
        frozen: Whether to freeze the backbone
        use_lora: Whether to apply LoRA for parameter efficient fine-tuning
        lora_rank: LoRA rank parameter
        lora_alpha: LoRA alpha scaling parameter
        **kwargs: Additional arguments for the backbone

    Returns:
        DINOv3Backbone instance
    """

    # Predefined model configurations
    model_configs = {
        "dinov3_vits14": {
            "patch_size": 16,
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
        },
        "dinov3_vitb14": {
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
        },
        "dinov3_vitl14": {
            "patch_size": 16,
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
        },
        "dinov3_vitg14": {
            "patch_size": 16,
            "embed_dim": 1536,
            "depth": 40,
            "num_heads": 24,
        },
    }

    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_configs.keys())}")

    config = model_configs[model_name]
    config.update(kwargs)

    # Set frozen stages
    if frozen and not use_lora:
        config['frozen_stages'] = -1  # Freeze all stages
    else:
        config['frozen_stages'] = 0   # No freezing

    # Configure LoRA
    if use_lora:
        config['use_lora'] = True
        config['lora_config'] = LoRAConfig(
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=['qkv', 'proj'],  # Target attention layers
            dropout=0.1
        )

    return DINOv3Backbone(pretrained=pretrained, **config)


# Convenience functions for different model variants
def dinov3_vits14(pretrained=None, **kwargs):
    """DINOv3 ViT-Small with 14x14 patches"""
    return build_dinov3_backbone("dinov3_vits14", pretrained, **kwargs)

def dinov3_vitb14(pretrained=None, **kwargs):
    """DINOv3 ViT-Base with 14x14 patches"""
    return build_dinov3_backbone("dinov3_vitb14", pretrained, **kwargs)

def dinov3_vitl14(pretrained=None, **kwargs):
    """DINOv3 ViT-Large with 14x14 patches"""
    return build_dinov3_backbone("dinov3_vitl14", pretrained, **kwargs)

def dinov3_vitg14(pretrained=None, **kwargs):
    """DINOv3 ViT-Giant with 14x14 patches"""
    return build_dinov3_backbone("dinov3_vitg14", pretrained, **kwargs)