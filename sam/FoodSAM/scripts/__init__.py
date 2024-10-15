import logging
import os

import cv2

from .instance import main as instance_infer
from .panoptic import main as panoptic_infer
from sam.segment_anything import SamAutomaticMaskGenerator
from sam.segment_anything.automatic_mask_generator_hq import SamAutomaticMaskGeneratorHQ
from sam.segment_anything_2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from .semantic import main as semantic_infer
from .object import main as object_infer
from .options import get_args, get_amg_kwargs

infer_mapping = {
    'semantic': semantic_infer,
    'instance': instance_infer,
    'panoptic': panoptic_infer,
    'object': object_infer
}

mask_mapping = {
    'segment_anything': SamAutomaticMaskGenerator,
    'segment_anything_hq': SamAutomaticMaskGeneratorHQ,
    'segment_anything_2': SAM2AutomaticMaskGenerator
}

other_mapping = [
    'num_class','area_thr','ratio_thr','top_k','confidence_threshold',
    'points_per_side','points_per_batch','pred_iou_thresh','stability_score_thresh'
]


def create_logger(save_folder):
    log_file = f"sam_process.log"
    final_log_file = os.path.join(save_folder, log_file)

    logging.basicConfig(
        format=
        '[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(final_log_file, mode='w'),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()
    print(f"Create Logger success in {final_log_file}")
    return logger

def han_res(results):
    path_dict, data_dict = [], {}
    for key, value in results.items():
        paths, data = value
        path_dict.extend(paths)
        data_dict[key] = {}
        if key == 'detection':
            boxes, scores, classes, classnames = data
            data_dict[key]['results'] = (boxes, classes, scores)
            data_dict[key]['names'] = classnames
        elif key in ['semantic', 'enhance']:
            # 使用 cv2.boundingRect 获取最小裁剪矩形框
            masks, classes = data
            boxes, names = [], {}
            for mask, cls in zip(masks, classes):
                x, y, w, h = cv2.boundingRect(mask)
                boxes.append([x, y, x + w, y + h])
                names[cls] = f"Class_{cls}"
            data_dict[key]['results'] = (boxes, classes, masks)
            data_dict[key]['names'] = names
        elif key in ['instance', 'panoptic']:
            boxes, masks, info = data
            classes, names = [], {}
            for id, (label, name) in enumerate(info):
                names[label] = name
                classes.append(label)
                x0, y0, w, h, area = boxes[id]
                boxes[id] = [x0, y0, x0 + w, y0 + h]
            data_dict[key]['results'] = (boxes, classes, masks)
            data_dict[key]['names'] = names
        else:
            raise ValueError(f"key {key} not found in han_res")

    return path_dict, data_dict

# 可配置参数: SAM_checkpoint, mode_type, mask_enhance
def _infer(mode, version, sam=None, keys=None, *args, **kwargs):
    options = get_args()
    res_dict = {}
    generator = None
    if keys is not None:
        for i, k in enumerate(keys):
            if hasattr(options, k):
                setattr(options, k, args[i])

    for k, v in kwargs.items():
        if hasattr(options, k):
            setattr(options, k, v)
    if mode not in infer_mapping:
        raise ValueError(f"mode {mode} not found in infer_mapping")
    if sam is not None:
        generator = mask_mapping[version](sam, output_mode="binary_mask",
                                          **get_amg_kwargs(options))
    options.output = os.path.join(options.output, mode)
    logger = create_logger(options.output)
    infer_mapping[mode](options, res_dict, logger, generator)
    path_dict, data_dict = han_res(res_dict)
    return path_dict, data_dict
