import argparse
import os
from typing import Any, Dict, List
import shutil, logging
from .options import get_args
from sam.FoodSAM.tools.object_detection import object_detect


# 单纯物体检测
def main(args: argparse.Namespace, res_dict: Dict, logger, *kwargs) -> None:
    t = args.img_path
    img_folder = os.path.join(args.output, os.path.basename(t)).split('.')[0]
    os.makedirs(args.output, exist_ok=True)

    logger.info("running object detection model")
    res_dict['detection'] = object_detect(img_folder, args)  # 返回目标检测
    logger.info("object detection done!\n")

if __name__ == "__main__":
    args = get_args()
    path_dict = {}
    main(args, path_dict)
