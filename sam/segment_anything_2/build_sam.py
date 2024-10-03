# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sam.segment_anything_2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam.segment_anything_2.sam2_image_predictor import SAM2ImagePredictor


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    print(config_file)
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):
    hydra_overrides = [
        "++model._target_=segment_anything_2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=0",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")


SAM2_PRETRAIN_PATH = "F:/Github/YOLO-World-SAM-RAM/sam/segment_anything_2/configs/"
SAM2_WEIGHTS_PATH = "F:/Github/YOLO-World-SAM-RAM/weights/sam/segment_anything_2/"

class SAM2:
    def __init__(self, model_id):
        print(model_id)
        model_cfg = os.path.join(SAM2_PRETRAIN_PATH, f'{model_id}.yaml')
        sam2_checkpoint = os.path.join(SAM2_WEIGHTS_PATH, f'{model_id}.pt')

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
        self.predictor = SAM2ImagePredictor(sam2_model)  # for single image
        # self.vid_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)  # for video
        # self.mask_generator = SAM2AutomaticMaskGenerator(sam2_model)  # seg_everything

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
