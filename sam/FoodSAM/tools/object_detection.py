# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import logging
import multiprocessing as mp
import os
import time
import cv2
import sys
import tqdm
import json
import copy
import sys

sys.path.append('.')
sys.path.append('./UNIDET/detectron2')
# from detectron2.detectron2.utils.logger import setup_logger
from UNIDET.detectron2.detectron2.config import get_cfg
from UNIDET.detectron2.detectron2.data.detection_utils import read_image
from UNIDET.unidet.predictor import UnifiedVisualizationDemo
from UNIDET.unidet.config import add_unidet_config

# constants
WINDOW_NAME = "Unified detections"
import json
import os


def visualize_to_json(predictions, new_metadata, output_dir):
    instances = predictions["instances"]
    boxes = instances.pred_boxes.tensor.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    class_names = new_metadata
    names ={}
    data = []
    for box, score, cls_id in zip(boxes, scores, classes):
        cls_name = class_names[cls_id]
        names[cls_id] = cls_name
        box = [round(x, 2) for x in box]
        score = round(score, 2)

        item = {
            "bounding_box": box,
            "type": "UniDet",
            "category_id": cls_id,
            "category_name": cls_name,
            "confidence": score
        }

        data.append(item)

    json_name = "od_UniDet.json"
    output_path = os.path.join(output_dir, json_name)
    basedir = os.path.dirname(output_path)
    os.makedirs(basedir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f)

    return boxes, scores, classes, names


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_unidet_config(cfg)
    cfg.merge_from_file(args.detection_config)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def object_detect(img_folder, args):
    mp.set_start_method("spawn", force=True)
    logger = logging.getLogger()

    cfg = setup_cfg(args)

    demo = UnifiedVisualizationDemo(cfg)

    path = args.img_path
    img = read_image(path, format="BGR")
    start_time = time.time()
    predictions, visualized_output = demo.run_on_image(img)

    output_dir = os.path.join(img_folder, 'object_detection')
    results = visualize_to_json(predictions, demo.metadata.thing_classes, output_dir)

    logger.info(
        "{}: {} in {:.2f}s".format(
            path,
            "detected {} instances".format(len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            time.time() - start_time,
        )
    )

    if os.path.isdir(img_folder):
        out_filename = os.path.join(img_folder, 'detection_vis.png')
        visualized_output.save(out_filename)
        print(out_filename)
        return (out_filename,), results
