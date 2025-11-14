"""
Adaptive Data Processing for OSTrack Training
Enhanced processing that supports dynamic template and search sizes during training.
"""

import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F
from lib.utils.adaptive_sizing import AdaptiveSizer, AdaptiveSizingConfig
import random
import math


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class AdaptiveProcessing:
    """
    Adaptive processing that dynamically adjusts template and search sizes
    based on target characteristics during training.
    """

    def __init__(
        self,
        search_area_factor_range=(3.0, 5.0),  # Range for search area factors
        template_area_factor_range=(1.5, 2.5),  # Range for template area factors
        output_sz_range=(128, 384),  # Range for output sizes
        center_jitter_factor={'template': 0, 'search': 3},
        scale_jitter_factor={'template': 0, 'search': 0.25},
        mode='pair',
        settings=None,
        adaptive_config=None,
        use_fixed_sizes=False,  # For compatibility with original training
        transform=transforms.ToTensor(),
        template_transform=None,
        search_transform=None,
        joint_transform=None
    ):
        """
        Args:
            search_area_factor_range: (min, max) range for search area factors
            template_area_factor_range: (min, max) range for template area factors
            output_sz_range: (min, max) range for output sizes
            center_jitter_factor: Dict containing jittering amounts
            scale_jitter_factor: Dict containing scale jittering amounts
            mode: Either 'pair' or 'sequence'
            settings: Additional settings
            adaptive_config: AdaptiveSizingConfig for adaptive sizing
            use_fixed_sizes: If True, uses original fixed-size processing
        """
        self.search_area_factor_range = search_area_factor_range
        self.template_area_factor_range = template_area_factor_range
        self.output_sz_range = output_sz_range
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings
        self.use_fixed_sizes = use_fixed_sizes

        # Transform setup
        self.transform = {
            'template': transform if template_transform is None else template_transform,
            'search': transform if search_transform is None else search_transform,
            'joint': joint_transform
        }

        # Initialize adaptive sizer
        if adaptive_config is None:
            adaptive_config = AdaptiveSizingConfig(
                strategy="target_adaptive",
                template_factor=2.0,
                search_factor=4.0,
                min_template_size=96,
                max_template_size=320,
                min_search_size=192,
                max_search_size=640
            )
        self.adaptive_sizer = AdaptiveSizer(adaptive_config)

    def _sample_adaptive_parameters(self, target_bbox):
        """
        Sample adaptive parameters based on target characteristics.

        Args:
            target_bbox: Target bounding box [x, y, w, h]

        Returns:
            dict: Sampled parameters
        """
        if self.use_fixed_sizes:
            # Original fixed-size behavior
            return {
                'template_factor': 2.0,
                'search_factor': 4.0,
                'template_size': 128,
                'search_size': 256
            }

        # Compute target characteristics
        target_scale = self.adaptive_sizer.compute_target_scale(target_bbox)

        # Sample adaptive sizes
        template_size, search_size = self.adaptive_sizer.compute_adaptive_sizes(target_bbox)

        # Add some randomization during training for robustness
        size_noise = random.uniform(0.9, 1.1)
        template_size = int(template_size * size_noise)
        search_size = int(search_size * size_noise)

        # Ensure sizes are within valid ranges
        template_size = max(self.output_sz_range[0], min(self.output_sz_range[1], template_size))
        search_size = max(self.output_sz_range[0], min(self.output_sz_range[1], search_size))

        # Ensure patch alignment
        template_size = self.adaptive_sizer.ensure_patch_alignment(template_size)
        search_size = self.adaptive_sizer.ensure_patch_alignment(search_size)

        # Compute corresponding factors
        template_factor = template_size / target_scale
        search_factor = search_size / target_scale

        # Clamp factors to reasonable ranges
        template_factor = max(self.template_area_factor_range[0],
                            min(self.template_area_factor_range[1], template_factor))
        search_factor = max(self.search_area_factor_range[0],
                          min(self.search_area_factor_range[1], search_factor))

        return {
            'template_factor': template_factor,
            'search_factor': search_factor,
            'template_size': template_size,
            'search_size': search_size,
            'target_scale': target_scale
        }

    def _get_jittered_box(self, box, mode):
        """
        Jitter the input box.

        Args:
            box: input bounding box
            mode: string 'template' or 'search' indicating template or search data

        Returns:
            torch.Tensor: jittered box
        """
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        Process data with adaptive sizing.

        Args:
            data: The input data, should contain:
                'template_images', 'search_images', 'template_anno', 'search_anno'

        Returns:
            TensorDict: output data block with processed images and annotations
        """
        # Apply joint transforms first
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

        # Sample adaptive parameters based on template annotation
        template_bbox = data['template_anno'][0]  # First template
        adaptive_params = self._sample_adaptive_parameters(template_bbox)

        # Process template image
        template_bbox_jittered = self._get_jittered_box(template_bbox, 'template')

        # Extract template with adaptive size
        template_img, template_resize_factor, template_att_mask = prutils.sample_target(
            data['template_images'][0],
            template_bbox_jittered,
            adaptive_params['template_factor'],
            output_sz=adaptive_params['template_size']
        )

        # Process search image
        search_bbox = data['search_anno'][0]  # First search
        search_bbox_jittered = self._get_jittered_box(search_bbox, 'search')

        # Extract search with adaptive size
        search_img, search_resize_factor, search_att_mask = prutils.sample_target(
            data['search_images'][0],
            search_bbox_jittered,
            adaptive_params['search_factor'],
            output_sz=adaptive_params['search_size']
        )

        # Transform images
        template_img = self.transform['template'](template_img)
        search_img = self.transform['search'](search_img)

        # Transform bounding boxes to crop coordinates
        template_bbox_crop = prutils.transform_image_to_crop(
            template_bbox, template_bbox_jittered, template_resize_factor,
            torch.tensor([adaptive_params['template_size'], adaptive_params['template_size']]).float()
        )

        search_bbox_crop = prutils.transform_image_to_crop(
            search_bbox, search_bbox_jittered, search_resize_factor,
            torch.tensor([adaptive_params['search_size'], adaptive_params['search_size']]).float()
        )

        # Prepare output data
        output_data = TensorDict({
            'template_images': template_img.unsqueeze(0),
            'search_images': search_img.unsqueeze(0),
            'template_anno': template_bbox_crop.unsqueeze(0),
            'search_anno': search_bbox_crop.unsqueeze(0),
            'template_att_masks': torch.from_numpy(template_att_mask).unsqueeze(0),
            'search_att_masks': torch.from_numpy(search_att_mask).unsqueeze(0),
        })

        # Add adaptive metadata
        output_data['adaptive_info'] = {
            'template_size': adaptive_params['template_size'],
            'search_size': adaptive_params['search_size'],
            'template_factor': adaptive_params['template_factor'],
            'search_factor': adaptive_params['search_factor'],
            'target_scale': adaptive_params.get('target_scale', 0),
            'template_resize_factor': template_resize_factor,
            'search_resize_factor': search_resize_factor
        }

        return output_data


class MultiScaleProcessing(AdaptiveProcessing):
    """
    Multi-scale processing that trains with multiple scales simultaneously.
    This helps the model generalize better across different target sizes.
    """

    def __init__(self, scale_variants=3, **kwargs):
        """
        Args:
            scale_variants: Number of different scales to generate per sample
        """
        super().__init__(**kwargs)
        self.scale_variants = scale_variants

    def __call__(self, data: TensorDict):
        """Generate multiple scale variants for training."""
        variants = []

        for i in range(self.scale_variants):
            # Temporarily modify the sizing parameters for each variant
            original_ranges = (self.search_area_factor_range, self.template_area_factor_range)

            # Create scale variations
            scale_multiplier = random.uniform(0.7, 1.4)  # 0.7x to 1.4x scale variation
            self.search_area_factor_range = (
                original_ranges[0][0] * scale_multiplier,
                original_ranges[0][1] * scale_multiplier
            )
            self.template_area_factor_range = (
                original_ranges[1][0] * scale_multiplier,
                original_ranges[1][1] * scale_multiplier
            )

            # Process with this scale
            variant_data = super().__call__(data.copy())
            variants.append(variant_data)

            # Restore original ranges
            self.search_area_factor_range, self.template_area_factor_range = original_ranges

        # Stack variants or return one randomly
        if len(variants) == 1:
            return variants[0]
        else:
            # For now, return a random variant
            # In future, could stack them for multi-scale training
            return random.choice(variants)


def create_adaptive_processing(
    mode='adaptive',
    search_factor_range=(3.0, 5.0),
    template_factor_range=(1.5, 2.5),
    size_range=(128, 384),
    **kwargs
):
    """
    Factory function to create adaptive processing with different modes.

    Args:
        mode: 'adaptive', 'multiscale', or 'fixed'
        search_factor_range: Range for search area factors
        template_factor_range: Range for template area factors
        size_range: Range for output sizes
        **kwargs: Additional arguments

    Returns:
        Processing object
    """
    common_args = {
        'search_area_factor_range': search_factor_range,
        'template_area_factor_range': template_factor_range,
        'output_sz_range': size_range,
        **kwargs
    }

    if mode == 'fixed':
        # Original fixed-size processing for compatibility
        common_args['use_fixed_sizes'] = True
        return AdaptiveProcessing(**common_args)
    elif mode == 'multiscale':
        return MultiScaleProcessing(**common_args)
    elif mode == 'adaptive':
        return AdaptiveProcessing(**common_args)
    else:
        raise ValueError(f"Unknown processing mode: {mode}")