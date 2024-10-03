# Copyright (c) Tencent Inc. All rights reserved.
import os.path as osp
import sys
import cv2
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from mmengine.runner.amp import autocast
import supervision as sv

class WORLD_MM:
    def __init__(self, config_file, checkpoint, device='cuda:0', reparameterize=True):
        cfg = Config.fromfile(config_file)
        cfg.work_dir = osp.join('./work_dirs')
        cfg.load_from = checkpoint
        self.model = init_detector(cfg, checkpoint=checkpoint, device=device)
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
        test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(test_pipeline_cfg)
        self.reparameterize = reparameterize

    def set_classes(self, classes):
        texts = [[clss] for clss in classes]
        self.texts = texts
        if self.reparameterize:
            self.model.reparameterize(texts)


    def inference(self, image, score_thr=0.3, max_dets=100, use_amp=False):
        if not hasattr(self, 'texts'):
            raise ValueError("Please set texts first.")
        data_info = dict(img=image, img_id=0, texts=texts)
        data_info = self.test_pipeline(data_info)
        data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                          data_samples=[data_info['data_samples']])
        with autocast(enabled=use_amp), torch.no_grad():
            output = self.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        # score thresholding
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
        # max detections
        if len(pred_instances.scores) > max_dets:
            indices = pred_instances.scores.float().topk(max_dets)[1]
            pred_instances = pred_instances[indices]

        pred_instances = pred_instances.cpu().numpy()
        boxes = pred_instances['bboxes']
        labels = pred_instances['labels']
        scores = pred_instances['scores']
        label_texts = [texts[x][0] for x in labels]

        return boxes, labels, label_texts, scores

# 简单推理器
def inference(model, image, texts, test_pipeline, max_dets=100, score_thr=0.3, nms_threshold=0.7, use_amp=False):
    image = cv2.imread(image)
    image = image[:, :, [2, 1, 0]]
    data_info = dict(img=image, img_id=0, texts=texts)
    data_info = test_pipeline(data_info) # 标签提取器
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])
    with autocast(enabled=use_amp), torch.no_grad():
        output = model.test_step(data_batch)[0]
    pred_instances = output.pred_instances
    # score thresholding
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    # max detections
    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    # pred_instances = pred_instances.cpu().numpy()
    # boxes = pred_instances['bboxes']
    # labels = pred_instances['labels']
    # scores = pred_instances['scores']
    # label_texts = [texts[x][0] for x in labels]
    detections = sv.Detections.from_inference(pred_instances).with_nms(
        class_agnostic=True, threshold=nms_threshold
    )
    return detections


class YOLO_WORLD:

    def __init__(self, version, url, **kwargs):
        self.version = version
        if version == 'ultralytics':
            self.model = YOLO(url)
        elif version == 'mm':
            cfg_path = url.copy()
            cfg_path = cfg_path.replace("weights", "configs/pretrain")
            cfg_path = cfg_path.replace(".pt", ".py")
            self.model = WORLD_MM(cfg_path, url)
        elif version == 'roboflow':
            self.model = YOLOWorld(model_id=url)
        else:
            raise ValueError(f"Version {version} is not supported.")

    def set_classes(self, classes):
        self.categories = classes
        self.model.set_classes(classes)

    def inference(self, image, score_thr=0.3, max_dets=100):
        if self.version == 'ultralytics':
            return self.model.predict(image)
        elif self.version == 'mm':
            return image, self.model.infer(image)
        elif self.version == 'roboflow':
            return image, self.model.infer(image)
        else:
            raise ValueError(f"Version {self.version} is not supported.")

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
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
            classes = results[0].boxes.cls.cpu().numpy()  # Class IDs
            confs = results[0].boxes.conf.cpu().numpy()
            ann_img = results[0].plot()
        elif self.version in ['mm', 'roboflow']:
            image, detections = results
            boxes = detections.boxes
            classes = detections.class_id
            confs = detections.confidence
            labels = [
                f"{self.categories[class_id]}: {confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
            ann_image = sv.BoundingBoxAnnotator().annotate(image, detections)
            ann_image = sv.LabelAnnotator().annotate(image, detections, labels=labels)
        else:
            raise ValueError(f"Version {self.version} is not supported.")

        cls_ids = []
        for box, cls_id, conf in zip(boxes, classes, confs):
            if cls_id not in cls_ids:
                cls_ids.append(cls_id)

        return results[0].plot(), [self.categories[id] for id in cls_ids]





if __name__ == "__main__":
    sys.path.append('F:/Github/YOLO-World')
    config_file = "F:/Github/YOLO-World/configs/pretrain/yolo_world_v2_x_1280.py"
    checkpoint = "F:/Github/YOLO-World/weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth"

    model = WORLD_MM(config_file, checkpoint)
    texts = [['person'], ['bus'], [' ']]
    model.set_classes(texts)
    image = "F:/Github/YOLO-World/demo/sample_images/bus.jpg"
    print(f"starting to detect: {image}")
    results = model.inference(image, score_thr=0.3, max_dets=100, use_amp=False)
    format_str = [
        f"obj-{idx}: {box}, label-{lbl}, class-{lbl_text}, score-{score}"
        for idx, (box, lbl, lbl_text, score) in enumerate(zip(*results))
    ]
    print("detecting results:")
    for q in format_str:
        print(q)
