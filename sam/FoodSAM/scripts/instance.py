import cv2
import argparse
import os
import numpy as np
from typing import Any, Dict, List
import shutil, logging

from sam.FoodSAM.scripts.options import get_amg_kwargs, get_args
from sam.FoodSAM.tools.panoramic_segment import panoramic_segment, instance_segment
from sam.FoodSAM.tools.predict_semantic_mask import semantic_predict, sam_predict
from sam.segment_anything import sam_model_registry, SamAutomaticMaskGenerator



def main(args: argparse.Namespace, res_dict: Dict, logger, generator):
    os.makedirs(args.output, exist_ok=True)

    t = args.img_path
    img_folder = os.path.join(args.output, os.path.basename(t)).split('.')[0]
    logger.info("running sam!")
    logger.info(f"Processing '{t}'...")
    sam_predict(t, args.output, generator, logger)
    logger.info("sam done!\n")

    logger.info("running semantic seg model!")  # 返回语义分割结果
    res_dict['semantic'] = semantic_predict(args.semantic_config, args.options,
                             args.aug_test, args.semantic_checkpoint,
                             args.color_list_path, args.img_path, img_folder)
    logger.info("semantic predict done!\n")

    logger.info("instance segmentation!")  # 返回实例分割结果
    res_dict['instance'] = instance_segment(img_folder, args.category_txt, args.color_list_path,
                     num_class=args.num_class, area_thr=args.area_thr, ratio_thr=args.ratio_thr, top_k=args.top_k)
    logger.info("instance segmentation done!\n")

    logger.info("The results saved in {}!\n".format(img_folder))


if __name__ == "__main__":
    args = get_args()
    sam = sam_model_registry[args.model_type](checkpoint=args.SAM_checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
    path_dict = {}
    main(args, path_dict, generator)