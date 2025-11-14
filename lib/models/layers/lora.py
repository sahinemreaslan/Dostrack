"""
LoRA (Low-Rank Adaptation) implementation for Parameter Efficient Fine-tuning
Based on "LoRA: Low-Rank Adaptation of Large Language Models" (https://arxiv.org/abs/2106.09685)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class LoRALinear(nn.Module):
    """
    LoRA adaptation for Linear layers.

    Replaces W with W + (B * A) where A and B are low-rank matrices.
    Only A and B are trainable, while W is frozen.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
        merge_weights: bool = True
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.merge_weights = merge_weights

        # Frozen linear layer (original weights)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False

        # LoRA parameters
        if rank > 0:
            self.lora_A = nn.Parameter(torch.randn(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self.scaling = alpha / rank

            # Initialize A with random values, B with zeros
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def merge(self):
        """Merge LoRA weights into the original linear layer."""
        if self.rank > 0 and not self.merged:
            delta_w = (self.lora_B @ self.lora_A).T * self.scaling
            self.linear.weight.data += delta_w
            self.merged = True

    def unmerge(self):
        """Unmerge LoRA weights from the original linear layer."""
        if self.rank > 0 and self.merged:
            delta_w = (self.lora_B @ self.lora_A).T * self.scaling
            self.linear.weight.data -= delta_w
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank == 0 or self.merged:
            return self.linear(x)

        # Standard linear transformation
        output = self.linear(x)

        # Add LoRA adaptation
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        output += lora_output * self.scaling

        return output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'rank={self.rank}, alpha={self.alpha}'


class LoRAAttention(nn.Module):
    """
    LoRA adaptation for multi-head attention layers.
    Applies LoRA to Q, K, V, and output projections.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None
    ):
        super().__init__()

        self.original_attn = original_attn
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

        # Default target modules for ViT attention
        if target_modules is None:
            target_modules = ['qkv', 'proj']

        self.target_modules = target_modules

        # Replace target linear layers with LoRA versions
        self._replace_linear_layers()

    def _replace_linear_layers(self):
        """Replace specified linear layers with LoRA versions."""
        for name in self.target_modules:
            if hasattr(self.original_attn, name):
                original_layer = getattr(self.original_attn, name)
                if isinstance(original_layer, nn.Linear):
                    lora_layer = LoRALinear(
                        in_features=original_layer.in_features,
                        out_features=original_layer.out_features,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=self.dropout,
                        bias=original_layer.bias is not None
                    )

                    # Copy original weights
                    lora_layer.linear.weight.data.copy_(original_layer.weight.data)
                    if original_layer.bias is not None:
                        lora_layer.linear.bias.data.copy_(original_layer.bias.data)

                    setattr(self.original_attn, name, lora_layer)

    def forward(self, *args, **kwargs):
        """Forward pass through the modified attention layer."""
        return self.original_attn(*args, **kwargs)


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None
) -> nn.Module:
    """
    Apply LoRA to a model's linear layers.

    Args:
        model: The model to apply LoRA to
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout rate
        target_modules: List of module names to target (e.g., ['qkv', 'proj'])
        exclude_modules: List of module names to exclude

    Returns:
        Model with LoRA applied
    """
    if target_modules is None:
        target_modules = ['qkv', 'proj']
    if exclude_modules is None:
        exclude_modules = []

    # Find and replace linear layers
    replaced_modules = []
    for name, module in model.named_modules():
        # Skip excluded modules
        if any(exclude in name for exclude in exclude_modules):
            continue

        # Check if this is a target linear layer
        if isinstance(module, nn.Linear):
            # Check if the module name contains any target pattern
            module_basename = name.split('.')[-1]
            if any(target in module_basename for target in target_modules):
                # Get parent module and attribute name
                *parent_names, attr_name = name.split('.')
                parent = model
                for parent_name in parent_names:
                    parent = getattr(parent, parent_name)

                # Create LoRA version
                lora_layer = LoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    bias=module.bias is not None
                )

                # Copy original weights
                lora_layer.linear.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    lora_layer.linear.bias.data.copy_(module.bias.data)

                # Replace the module
                setattr(parent, attr_name, lora_layer)
                replaced_modules.append(name)

    print(f"Applied LoRA to {len(replaced_modules)} modules: {replaced_modules}")
    return model


def get_lora_parameters(model: nn.Module) -> List[torch.Tensor]:
    """
    Get all LoRA parameters from a model.

    Args:
        model: Model with LoRA layers

    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALinear) and module.rank > 0:
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def count_lora_parameters(model: nn.Module) -> int:
    """
    Count the number of LoRA parameters in a model.

    Args:
        model: Model with LoRA layers

    Returns:
        Number of LoRA parameters
    """
    return sum(p.numel() for p in get_lora_parameters(model))


def count_total_parameters(model: nn.Module) -> int:
    """
    Count total parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def print_lora_info(model: nn.Module):
    """
    Print information about LoRA parameters in a model.

    Args:
        model: Model with LoRA layers
    """
    total_params = count_total_parameters(model)
    lora_params = count_lora_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"LoRA ratio: {lora_params / total_params:.4f} ({lora_params / total_params * 100:.2f}%)")
    print(f"Trainable ratio: {trainable_params / total_params:.4f} ({trainable_params / total_params * 100:.2f}%)")


class LoRAConfig:
    """Configuration class for LoRA parameters."""

    def __init__(
        self,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        exclude_modules: Optional[List[str]] = None,
        merge_weights: bool = True
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ['qkv', 'proj']
        self.exclude_modules = exclude_modules or []
        self.merge_weights = merge_weights

    def __repr__(self):
        return f"LoRAConfig(rank={self.rank}, alpha={self.alpha}, dropout={self.dropout}, " \
               f"target_modules={self.target_modules}, exclude_modules={self.exclude_modules})"