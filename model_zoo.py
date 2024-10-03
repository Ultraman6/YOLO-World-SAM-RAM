import json
import os
from typing import Union

from PIL import Image
import cv2
import numpy as np
import torch
from torch import nn

from ram import inference_ram_openset as inference
from ram import get_transform
from ram.utils import build_openset_llm_label_embedding
from sam.segment_anything_2 import SAM2
from world.groundingdino.model import Ground_Dino
from world.groundingdino.util.inference import annotate
from world.mm.model import WORLD_MM
from ram.models import ram_plus, ram
from sam.efficientvit.models.efficientvit import (EfficientViTSam,
                                                  efficientvit_sam_l0,
                                                  efficientvit_sam_l1,
                                                  efficientvit_sam_l2,
                                                  efficientvit_sam_xl0,
                                                  efficientvit_sam_xl1, EfficientViTSamPredictor)
from sam.efficientvit.models.nn.norm import set_norm_eps
from sam.efficientvit.models.utils import load_state_dict_from_file
from sam.segment_anything import SAM
from inference.models import YOLOWorld
from ultralytics import YOLO
import supervision as sv

from ultralytics.models.fastsam import FastSAMPredictor

__all__ = ["create_sam_model"]


REGISTERED_RAM = {
    'ram_plus': ram_plus,
    'ram': ram
}


REGISTERED_RAM_MODEL = {
    'ram_plus': {
        'swin_l': 'weights/ram/ram_plus_swin_large_14m.pth'
    },
    'ram': {
        'swin_l': 'weights/ram/ram_swin_large_14m.pth'
    },
}

REGISTERED_WORLD_MODEL = {
    'ultralytics': {
        "yolov8s-world": "weights/world/ultralytics/yolov8s-world.pt",
        "yolov8m-world": "weights/world/ultralytics/yolov8m-world.pt",
        "yolov8l-world": "weights/world/ultralytics/yolov8l-world.pt",
        "yolov8x-world": "weights/world/ultralytics/yolov8x-world.pt",
        "yolov8s-worldv2": "weights/world/ultralytics/yolov8s-worldv2.pt",
        "yolov8m-worldv2": "weights/world/ultralytics/yolov8m-worldv2.pt",
        "yolov8l-worldv2": "weights/world/ultralytics/yolov8l-worldv2.pt",
        "yolov8x-worldv2": "weights/world/ultralytics/yolov8x-worldv2.pt",
    },
    'roboflow': [
        'yolo_world/l',
        'yolo_world/m',
        'yolo_world/s',
        'yolo_world/x',
    ],
    'mm':  [
        "yolo_world_v2_l_640",
        "yolo_world_v2_l_1280",
        "yolo_world_v2_m_640",
        "yolo_world_v2_m_1280",
        "yolo_world_v2_s_640",
        "yolo_world_v2_s_1280",
        "yolo_world_v2_x_640",
        "yolo_world_v2_x_1280",
        "yolo_world_v2_xl_640"
    ],
    'dino': [
        'GroundingDINO-B',
        'GroundingDINO-T',
    ]
}

REGISTERED_SAM_MODEL = {
    'efficient': {
        "l0": "weights/sam/efficient/l0.pt",
        "l1": "weights/sam/efficient/l1.pt",
        "l2": "weights/sam/efficient/l2.pt",
        "xl0": "weights/sam/efficient/xl0.pt",
        "xl1": "weights/sam/efficient/xl1.pt",
    },
    'ultralytics': {
        'sam_b': 'weights/sam/ultralytics/sam_b.pt',
        'sam_l': 'weights/sam/ultralytics/sam_l.pt',
        'sam2_b': 'weights/sam/ultralytics/sam2_b.pt',
        'sam2_l': 'weights/sam/ultralytics/sam2_l.pt',
        'sam2_s': 'weights/sam/ultralytics/sam2_s.pt',
        'sam2_t': 'weights/sam/ultralytics/sam2_t.pt',
    },
    'segment_anything': {
        'sam_h': 'weights/sam/segment_anything/sam_vit_h_4b8939.pth' ,
        'sam_l': 'weights/sam/segment_anything/sam_vit_l_0b3195.pth',
        'sam_hq_tiny': 'weights/sam/segment_anything_hq/sam_hq_vit_t.pth',
        'sam_hq_b': 'weights/sam/segment_anything_hq/sam_hq_vit_b.pth',
        'sam_hq_h': 'weights/sam/segment_anything_hq/sam_hq_vit_h.pth',
        'sam_hq_l': 'weights/sam/segment_anything_hq/sam_hq_vit_l.pth',
    },
    'segment_anything_hq': {
        'sam_hq_tiny': 'weights/sam/segment_anything/sam_vit_l_0b3195.pth',
        'sam_hq_b': 'weights/sam/segment_anything/sam_vit_h_4b8939.pth',
        'sam_hq_h': 'weights/sam/segment_anything/sam_vit_l_0b3195.pth',
        'sam_hq_l': 'weights/sam/segment_anything/sam_vit_l_0b3195.pth',
    },
    'segment_anything_2': [
        'sam2_hiera_b+',
        'sam2_hiera_l',
        'sam2_hiera_s',
        'sam2_hiera_t'
    ],
    'fast': {
        'FastSAM-s': 'weights/sam/fast/FastSAM-s.pt',
        'FastSAM-x': 'weights/sam/fast/FastSAM-x.pt',
    },
    'mobile': {
        'MobileSAM': 'weights/sam/mobile/mobile_sam.pt',
    }
}

REGISTERED_MODEL={
    "ram": REGISTERED_RAM_MODEL,
    "world": REGISTERED_WORLD_MODEL,
    "sam": REGISTERED_SAM_MODEL
}

REGISTERED_NAME={
    "ram": "Recongnze-Anything",
    "world": 'YOLO-World',
    "sam": "Segment-Anything"
}


def create_sam_model(
    name: str, pretrained=True, weight_url: str or None = None, **kwargs
) -> EfficientViTSam:
    model_dict = {
        "l0": efficientvit_sam_l0,
        "l1": efficientvit_sam_l1,
        "l2": efficientvit_sam_l2,
        "xl0": efficientvit_sam_xl0,
        "xl1": efficientvit_sam_xl1,
    }

    model_id = name.split("-")[0]
    if model_id not in model_dict:
        raise ValueError(
            f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}"
        )
    else:
        model = model_dict[model_id](**kwargs)
    set_norm_eps(model, 1e-6)

    if pretrained:
        weight_url = weight_url or REGISTERED_SAM_MODEL.get(name, None)
        if weight_url is None:
            raise ValueError(f"Do not find the pretrained weight of {name}.")
        else:
            weight = load_state_dict_from_file(weight_url)
            model.load_state_dict(weight)

    return model

class _RAM:

    llm_tag_des_url = "datasets/openimages_rare_200/openimages_rare_200_llm_tag_descriptions.json"
    def __init__(self, version, type, url, img_size, llm_tag_des):
        if version == 'ram':
            self.model = ram(pretrained=url, vit=type, image_size=img_size)
        elif version == 'ram_plus':
            self.model = ram_plus(pretrained=url, vit=type, image_size=img_size)
        else:
            raise ValueError(f"Version {version} is not supported.")
        self._llm_tag_des(llm_tag_des)

    def _llm_tag_des(self, llm_tag_des, cls_thr=0.5):
        if llm_tag_des:
            print('Building tag embedding:')
            with open(self.llm_tag_des_url, 'rb') as fo:
                llm_tag_des = json.load(fo)
            openset_label_embedding, openset_categories = build_openset_llm_label_embedding(llm_tag_des)
            self.model.tag_list = np.array(openset_categories)
            self.model.label_embed = nn.Parameter(openset_label_embedding.float())
            self.model.num_class = len(openset_categories)
            # the threshold for unseen categories is often lower
            self.model.class_threshold = torch.ones(self.model.num_class) * cls_thr
        else:
            self.model.tag_list = None
            self.model.label_embed = None
            self.model.num_class = None
            self.model.class_threshold = None

    def infer(self, img, img_size):
        transform = get_transform(image_size=img_size)
        img = Image.fromarray(img)
        img = transform(img)
        img = img.unsqueeze(0).to('cuda:0')
        print(type(img))
        self.model.eval()
        self.model = self.model.to('cuda:0')
        return inference(img, self.model)


class _SAM:
    def __init__(self, version, url):
        overrides = dict(task="segment", mode="predict", model=url, retina_masks=True)
        if version in ['v1', 'v2', 'mobile']:
            self.model = SAMPredictor(overrides=overrides)
        elif version in ['fast']:
            # model = FastSAM("FastSAM-s.pt")
            self.model = FastSAMPredictor(overrides=overrides)
        elif version in ['efficient']:
            name = url.split("/")[-1].split(".")[0]
            self.model = EfficientViTSamPredictor(create_sam_model(name=name, weight_url=url).to('cuda:0').eval())
        elif version == 'segment_anything':
            self.model = SAM(model_id=url)
        elif version == 'segment_anything_2':
            self.model = SAM2(model_id=url)
        else:
            raise ValueError(f"Version {version} is not supported.")
        self.version = version

    def set_classes(self, classes):
        self.names = classes

    def check_mask(self, mask):
        return

    def infer(self, img, detections, masks=False):
        boxes, classes, confs = detections
        mask_list = []
        for box, cls, conf in zip(boxes, classes, confs):
            if self.version in ['efficient']:
                assert isinstance(self.model, EfficientViTSamPredictor)
                self.model.set_image(img)
                mask, _, _ = self.model.predict(box=box, multimask_output=masks)
                mask_list.append(mask.squeeze())
            elif self.version in ['fast']:
                assert isinstance(self.model, FastSAMPredictor)
                er = self.model(img)
                results = self.model.prompt(er, bboxes=box)
                mask = results[0].masks.data.cpu().numpy()
                print(mask, mask.shape)
                mask = mask.astype(bool)
                print(mask, mask.shape)
                mask_list.append(mask.squeeze())
            elif self.version in ['v1', 'v2', 'mobile']:
                assert isinstance(self.model, SAMPredictor)
                results = self.model(img, bboxes=box, retina_masks=masks)
                mask = results[0].masks.data.cpu().numpy()
                mask_list.append(mask.squeeze())
            elif self.version in ['segment_anything', 'segment_anything_2']:
                assert (isinstance(self.model, SAM)
                        or isinstance(self.model, SAM2))
                self.model.set_image(img)
                mask, _, _ = self.model.infer(box=box)
                mask_list.append(mask.squeeze())

        mask_list = np.array(mask_list)
        return self.annotate_image(img, (boxes, classes, confs, mask_list))

    def _results(self, results):
        boxes, classes, confs, masks = results
        return boxes, classes, confs, masks

    def annotate_image(self, img, results):
        """
        Annotates the original image with bounding boxes and class names.
        Args:
            im0 (np.ndarray): Original image to be annotated.
            results (ultralytics YOLO results): YOLO model's detection results.

        Returns:
            np.ndarray: Annotated image.
        """
        boxes, classes, confs, masks = self._results(results)
        detections = sv.Detections(xyxy=boxes, class_id=classes,
                                   mask=masks, confidence=confs)
        labels = [
            f"{self.names[class_id]}: {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        img = sv.MaskAnnotator().annotate(img, detections)
        img = sv.LabelAnnotator().annotate(img, detections, labels=labels)

        cls_names = []
        # 遍历每个分割的实例
        for box, cls, mask in zip(boxes, classes, masks):
            name = self.names[cls]
            if name not in cls_names:
                cls_names.append(name)

        return img, results, cls_names


class _WORLD:

    def __init__(self, version, url):
        self.version = version
        if version == 'ultralytics':
            self.model = YOLO(url)
        elif version == 'mm':
            self.model = WORLD_MM(url)
        elif version == 'roboflow':
            self.model = YOLOWorld(model_id=url)
        elif version == 'dino':
            self.model = Ground_Dino(model_id=url)
        else:
            raise ValueError(f"Version {version} is not supported.")

    def set_classes(self, classes):
        self.names = classes
        self.model.set_classes(classes)

    def infer(self, image, score_thr=0.3, max_det=100, nms_thr=0.5, amp=False, agnostic=False):
        if self.version == 'ultralytics':
            results = self.model.predict(image, amp=amp, conf=score_thr,
                                         iou=nms_thr, max_det=max_det, agnostic_nms=agnostic)
        elif self.version == 'mm':
            results = image, self.model.infer(image, score_thr=score_thr,
                                                  max_det=max_det, use_amp=amp, agn_cls=agnostic)

        elif self.version == 'roboflow':  # 暂未部署类别无关nms
            pred = self.model.infer(image, confidence=score_thr, amp=amp)
            if len(pred.predictions) > max_det:
                indices = pred.predictions.confidence.float().topk(max_det)[1]
                pred = pred[indices]

            results = image, sv.Detections.from_inference(pred).with_nms(
                class_agnostic=agnostic, threshold=nms_thr
            )
        elif self.version == 'dino':
            results = self.model.infer(image, box_threshold=nms_thr,
                                       text_threshold=score_thr)
        else:
            raise ValueError(f"Version {self.version} is not supported.")

        return self.annotate_image(results)

    def _results(self, results):
        if self.version == 'ultralytics':
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.float64)  # Bounding boxes
            classes = results[0].boxes.cls.cpu().numpy().astype(np.int8)  # Class IDs
            confs = results[0].boxes.conf.cpu().numpy().astype(np.float16)
        elif self.version in ['mm', 'roboflow']:
            result = results[1]
            boxes = result.xyxy
            classes = result.class_id
            confs = result.confidence
        elif self.version == 'dino':
            result = results[1]
            boxes, confs, labels = result
            boxes = boxes.cpu().numpy().astype(np.float64)
            confs = confs.cpu().numpy().astype(np.float16)
            classes = [self.names.index(label) for label in labels]
            classes = np.array(classes, dtype=np.int8)
        else:
            raise ValueError(f"Version {self.version} is not supported.")

        return boxes, classes, confs

    def annotate_image(self, results):
        """
        Annotates the original image with bounding boxes and class names.
        Args:
            im0 (np.ndarray): Original image to be annotated.
            results (ultralytics YOLO results): YOLO model's detection results.

        Returns:
            np.ndarray: Annotated image.
        """
        if self.version == 'ultralytics':
            ann_img = results[0].plot()
        elif self.version in ['mm', 'roboflow']:
            image, detections = results
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            labels = [
                f"{self.names[class_id]}: {confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
            ann_img = sv.BoxAnnotator().annotate(image, detections)
            ann_img = sv.LabelAnnotator().annotate(ann_img, detections, labels=labels)
        elif self.version == 'dino':
            image_pil, detections = results
            boxes, logits, phrases = detections
            annotated_frame = annotate(image_source=np.asarray(image_pil), boxes=boxes, logits=logits, phrases=phrases)
            ann_img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError(f"Version {self.version} is not supported.")

        boxes, classes, confs = self._results(results)
        # Create a mapping from original ids to new sequential ids, keeping the original relative order
        id_map = {id: i for i, id in enumerate(id for id in range(len(self.names)) if id in classes)}
        filter_names = [self.names[id] for id in id_map]
        filter_classes = np.array([id_map[cis] for cis in classes], dtype=np.int8)
        print(boxes, filter_classes, confs)
        return ann_img, filter_names, (boxes, filter_classes, confs)


if __name__ == "__main__":
    os.chdir("F:/Github/YOLO-World-SAM-RAM")  # 设置工作路径
    # yolo_world = YOLO_WORLD(version="ultralytics",
    #                         url="F:/Github/YOLO-World-SAM-RAM/weights/world/ultralytics/yolov8s-worldv2.pt")
    # yolo_world = YOLO_WORLD(version="roboflow",
    #                         url="yolo_world/l")
    # yolo_world = _YOLO_WORLD(version="mm", url='yolo_world_v2_l_640')
    # sam = _SAM(version='fast', url='weights/sam/fast/FastSAM-s.pt')
    # categories = ['person', 'bus']
    # yolo_world.set_classes(categories)
    image = "world/mm/demo/sample_images/bus.jpg"
    image = cv2.imread(image)
    # ann_img, labels, results = yolo_world.infer(image, score_thr=0.1, max_det=100, nms_thr=0.5, amp=False)
    # cv2.imshow("result", ann_img)
    # cv2.waitKey(0)
    # sam.set_classes(labels)
    # ann_img, results, _ = sam.infer(image, results)
    # cv2.imshow("result", ann_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    from ultralytics.models.sam import Predictor as SAMPredictor

    # Create SAMPredictor
    overrides = dict(conf=0.25, task="segment", mode="predict", model="weights/sam/fast/FastSAM-s.pt")
    # overrides = dict(conf=0.25, task="segment", mode="predict", model="weights/sam/v1/sam_b.pt")
    predictor = FastSAMPredictor(overrides=overrides)
    # predictor = SAMPredictor(overrides=overrides)
    # predictor = SAM("weights/sam/v1/sam_b.pt")
    # Set image
    # predictor.set_image(image)  # set with np.ndarray
    r = predictor(image)  # set with np.ndarray
    results = predictor.prompt(r, bboxes=[439, 437, 524, 709])  # predict with bboxes
    results[0].names = ['person']
    results[0].show()
    # r = predictor(image)
    b = results[0].masks.data.cpu().numpy()
    print(b, b.shape, b.dtype)
    # results = predictor.prompt(r, bboxes=[0, 0, 100, 100])
    # results[0].names = ['bus']
    # results[0].show(boxes=False)
    # Reset image
    # predictor.reset_image()