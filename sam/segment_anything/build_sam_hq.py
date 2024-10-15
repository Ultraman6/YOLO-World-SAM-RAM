# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import re

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from . import SamPredictor, SamAutomaticMaskGenerator
from .automatic_mask_generator_hq import SamAutomaticMaskGeneratorHQ
from .modeling import ImageEncoderViT, MaskDecoderHQ, PromptEncoder, Sam, TwoWayTransformer
from .modeling.image_encoder_hq import ImageEncoderViTHQ
from .predictor_hq import SamPredictorHQ

def _load_sam_checkpoint(sam: Sam, checkpoint=None):
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        info = sam.load_state_dict(state_dict, strict=False)
        print(info)
    for _, p in sam.named_parameters():
        p.requires_grad = False
    return sam




def build_sam_hq_vit_h(checkpoint=None):
    return _build_sam_hq(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam_hq = build_sam_hq_vit_h


def build_sam_hq_vit_l(checkpoint=None):
    return _build_sam_hq(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_hq_vit_b(checkpoint=None):
    return _build_sam_hq(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_hq_model_registry = {
    "default": build_sam_hq_vit_h,
    "vit_h": build_sam_hq_vit_h,
    "vit_l": build_sam_hq_vit_l,
    "vit_b": build_sam_hq_vit_b,
}


def _build_sam_hq(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViTHQ(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoderHQ(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=encoder_embed_dim,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    return _load_sam_checkpoint(sam, checkpoint)


class SAM_hq:
    def  __init__(self, model_id):
        use_sam2 = False
        if not use_sam2:
            match = re.search(r'vit_[lh]', model_id)
            print(match)
            if match:
                model_type = match.group(0)
            else:
                raise ValueError("Model type not found in the URL")
            print("Loading model")
            self.sam = sam_hq_model_registry[model_type](checkpoint=model_id).to('cuda')
            print("Finishing loading")
            self.predictor = SamPredictorHQ(self.sam, True)
            # self.mask_generator = SamAutomaticMaskGeneratorHQ(self.sam)  # 全自动sam

    def set_image(self, img):
        self.predictor.set_image(img)

    def infer(self, box):
        masks_, scores_, logits_ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False
        )
        idx, max_score = 0, 0
        for i, score in enumerate(scores_):
            if score > max_score:
                max_score = score
                idx = i
        return masks_[idx], scores_[idx], logits_[idx]