"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.models.ostrack.dinov3_models import dinov3_vits16, dinov3_vitb16, dinov3_vitl16, dinov3_vitg16
from lib.utils.box_ops import box_xyxy_to_cxcywh


class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
        elif head_type == "TrulyDynamicCenterPredictor":
            # TrulyDynamicCenterPredictor doesn't have fixed feat_sz
            # Use default values, will be calculated dynamically
            self.feat_sz_s = 16  # Default fallback
            self.feat_len_s = 256  # Default fallback

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # For DINOv3: dynamically calculate search region length
        # cat_feature format: [B, template_patches + search_patches, C]
        B, total_len, C = cat_feature.shape

        # Calculate template patches from known template size
        if hasattr(self.backbone, 'template_feat_size'):
            template_patches = self.backbone.template_feat_size ** 2
        else:
            # Fallback: assume template is 1/4 the area of search (2:4 ratio squared = 1:4)
            template_patches = self.feat_len_s // 4

        search_patches = total_len - template_patches

        # Extract search region features
        enc_opt = cat_feature[:, -search_patches:]  # actual search patches, not fixed feat_len_s
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()

        # Check if we're using TrulyDynamicCenterPredictor
        if hasattr(self.box_head, '__class__') and 'TrulyDynamic' in self.box_head.__class__.__name__:
            # TrulyDynamicCenterPredictor can handle any rectangular feature maps
            # Calculate the best rectangular dimensions
            factors = []
            for i in range(1, int(HW**0.5) + 1):
                if HW % i == 0:
                    factors.append((i, HW // i))

            # Choose the most square-like factor pair
            best_h, best_w = min(factors, key=lambda x: abs(x[0] - x[1]))
            opt_feat = opt.view(-1, C, best_h, best_w)

        else:
            # Original logic for static heads - need square features
            actual_feat_sz = int(HW ** 0.5)

            # Handle non-square feature maps by truncation
            if actual_feat_sz * actual_feat_sz != HW:
                # For non-square, truncate to largest possible square that fits
                while actual_feat_sz * actual_feat_sz > HW:
                    actual_feat_sz -= 1

                # Truncate the features to make them square
                square_patches = actual_feat_sz * actual_feat_sz
                enc_opt = enc_opt[:, :square_patches]  # Truncate to square
                opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
                bs, Nq, C, HW = opt.size()

            opt_feat = opt.view(-1, C, actual_feat_sz, actual_feat_sz)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out

        elif self.head_type == "TrulyDynamicCenterPredictor":
            # run the truly dynamic center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out

        else:
            raise NotImplementedError


def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'dinov3_vits16':
        backbone = dinov3_vits16(pretrained,
                                frozen=cfg.MODEL.BACKBONE.get('FROZEN', True),
                                use_lora=cfg.MODEL.BACKBONE.get('USE_LORA', False),
                                lora_rank=cfg.MODEL.BACKBONE.get('LORA_RANK', 4),
                                lora_alpha=cfg.MODEL.BACKBONE.get('LORA_ALPHA', 1.0),
                                drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'dinov3_vitb16':
        backbone = dinov3_vitb16(pretrained,
                                frozen=cfg.MODEL.BACKBONE.get('FROZEN', True),
                                use_lora=cfg.MODEL.BACKBONE.get('USE_LORA', False),
                                lora_rank=cfg.MODEL.BACKBONE.get('LORA_RANK', 4),
                                lora_alpha=cfg.MODEL.BACKBONE.get('LORA_ALPHA', 1.0),
                                drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'dinov3_vitl16':
        backbone = dinov3_vitl16(pretrained,
                                frozen=cfg.MODEL.BACKBONE.get('FROZEN', True),
                                use_lora=cfg.MODEL.BACKBONE.get('USE_LORA', False),
                                lora_rank=cfg.MODEL.BACKBONE.get('LORA_RANK', 4),
                                lora_alpha=cfg.MODEL.BACKBONE.get('LORA_ALPHA', 1.0),
                                drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'dinov3_vitg16':
        backbone = dinov3_vitg16(pretrained,
                                frozen=cfg.MODEL.BACKBONE.get('FROZEN', True),
                                use_lora=cfg.MODEL.BACKBONE.get('USE_LORA', False),
                                lora_rank=cfg.MODEL.BACKBONE.get('LORA_RANK', 4),
                                lora_alpha=cfg.MODEL.BACKBONE.get('LORA_ALPHA', 1.0),
                                drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = OSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
