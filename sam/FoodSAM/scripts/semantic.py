import cv2
import argparse

import os
import numpy as np
from typing import Any, Dict, List
import shutil, logging
from sam.FoodSAM.tools.predict_semantic_mask import semantic_predict, sam_predict
from sam.FoodSAM.tools.enhance_semantic_masks import enhance_masks
from sam.FoodSAM.tools.evaluate_foodseg103 import evaluate

from sam.FoodSAM.scripts.options import get_amg_kwargs, get_args


def main(args: argparse.Namespace, res_dict: Dict, logger, generator=None) -> None:
    os.makedirs(args.output, exist_ok=True)
    t = args.img_path
    img_folder = os.path.join(args.output, os.path.basename(t)).split('.')[0]

    # 语义分割与sam无关
    logger.info("running semantic seg model!")  # 返回语义分割结果
    res_dict['semantic'] = semantic_predict(args.semantic_config, args.options, args.aug_test,
                     args.semantic_checkpoint, args.color_list_path, args.img_path, img_folder)
    logger.info("semantic predict done!\n")

    if args.mask_enhance and generator is not None:
        logger.info("running sam!")
        logger.info(f"Processing '{t}'...")
        sam_predict(t, args.output, generator, logger)
        logger.info("sam done!\n")

        logger.info("enhance semantic masks") # 返回增强语义结果
        res_dict['enhance'] = enhance_masks(img_folder, args.category_txt, args.color_list_path, num_class=args.num_class,
                      area_thr=args.area_thr, ratio_thr=args.ratio_thr, top_k=args.top_k)
        logger.info("enhance semantic masks done!\n")

    logger.info("The results saved in {}!\n".format(img_folder))


if __name__ == "__main__":
    args = get_args()
    path_dict = {}
    main(args, path_dict)

