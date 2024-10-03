# Copyright (c) Tencent Inc. All rights reserved.
import inspect
import os
import os.path as osp
import sys
import cv2
import torch
from torchvision.ops import nms
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from mmengine.runner.amp import autocast
import supervision as sv
from inference.models import YOLOWorld
from mmcv.ops import batched_nms

MM_PRETRAIN_PATH = "world/mm/configs/pretrain/"
MM_WEIGHTS_PATH = "weights/world/mm/"

class WORLD_MM:
    def __init__(self, url, device='cuda:0'):
        cfg_path = os.path.join(MM_PRETRAIN_PATH, f'{url}.py')
        checkpoint = os.path.join(MM_WEIGHTS_PATH, f'{url}.pth')
        print(cfg_path, checkpoint)
        cfg = Config.fromfile(cfg_path)
        cfg.work_dir = osp.join('./work_dirs')
        cfg.load_from = checkpoint
        self.model = init_detector(cfg, checkpoint=checkpoint, device=device)
        print(cfg_path, checkpoint)
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
        test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(test_pipeline_cfg)

    def set_classes(self, classes):
        texts = [[clss] for clss in classes]
        self.texts = texts
        self.model.reparameterize(texts)


    def infer(self, image, score_thr=0.3, max_det=100, nms_thr=0.7, use_amp=False, agn_cls=False):
        if not hasattr(self, 'texts'):
            raise ValueError("Please set texts first.")
        image = image[:, :, [2, 1, 0]]
        data_info = dict(img=image, img_id=0, texts=self.texts)
        data_info = self.test_pipeline(data_info)
        data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                          data_samples=[data_info['data_samples']])
        with autocast(enabled=use_amp), torch.no_grad():
            output = self.model.test_step(data_batch)[0]
            pred_instances = output.pred_instances
        # 提取预测结果
        boxes = pred_instances.bboxes  # (N, 4)
        scores = pred_instances.scores  # (N,)
        labels = pred_instances.labels  # (N,)
        # 定义 NMS 配置
        nms_cfg = {'iou_threshold': nms_thr}
        # 执行 batched_nms
        dets, keep = batched_nms(
            boxes,
            scores,
            labels,
            nms_cfg,
            class_agnostic=agn_cls  # 设置为 True 表示类别无关 NMS
        )
        # dets 是形状为 (M, 5) 的张量，包含经过 NMS 后的边界框和分数
        # keep 是保留下来的框在原始输入中的索引
        # 过滤得分低于阈值的检测框
        score_mask = dets[:, 4] > score_thr
        dets = dets[score_mask]
        keep = keep[score_mask]
        # 限制最大检测数
        if dets.size(0) > max_det:
            topk_indices = dets[:, 4].topk(max_det)[1]
            dets = dets[topk_indices]
            keep = keep[topk_indices]
        # 更新 pred_instances
        pred_instances = pred_instances[keep]
        # 将 pred_instances 转移到 CPU（如果需要）
        pred_instances = pred_instances.cpu().numpy()
        if 'masks' in pred_instances:
            masks = pred_instances['masks']
        else:
            masks = None

        return sv.Detections(xyxy=pred_instances['bboxes'],
                                   class_id=pred_instances['labels'],
                                   confidence=pred_instances['scores'],
                                   mask=masks)


if __name__ == "__main__":
    sys.path.append('F:/Github/YOLO-World')
    config_file = "configs/pretrain/yolo_world_v2_x_1280.py"
    checkpoint = "./weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth"

    # model = WORLD_MM(config_file, checkpoint)
    #
    # model.set_classes(['person', 'bus'])
    image = "./demo/sample_images/bus.jpg"
    # print(f"starting to detect: {image}")
    image = cv2.imread(image)  # 通道转换
    # image = image[:, :, [2, 1, 0]]
    # _, results = model.inference(image, score_thr=0.3, max_dets=100, use_amp=False)
    # print(results)
    yolo_world = YOLOWorld(model_id="yolo_world/l")
    categories = ['person', 'bus']
    yolo_world.set_classes(categories)
    results = yolo_world.infer(image, confidence=0.1)
    if len(results.predictions) > 100:
        indices = results.predictions.confidence.float().topk(100)[1]
        results = results[indices]
    detections = sv.Detections.from_inference(results).with_nms(
        class_agnostic=True, threshold=0.5
    )
    labels = [
        f"{categories[class_id]}: {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ann_img = sv.BoundingBoxAnnotator().annotate(scene=image, detections=detections)
    ann_img = sv.LabelAnnotator().annotate(ann_img, detections, labels=labels)
    cv2.imshow("result", ann_img)
    cv2.waitKey(0)
    # 获取函数的签名
    signature = inspect.signature(yolo_world.infer)
    print(signature)  # 输出: (a, b, c=10, *args, **kwargs)

    # 获取参数的详细信息
    for name, param in signature.parameters.items():
        print(f"参数名: {name}, 类型: {param.kind}, 默认值: {param.default}")
    # format_str = [
    #     f"obj-{idx}: {box}, label-{lbl}, class-{lbl_text}, score-{score}"
    #     for idx, (box, lbl, lbl_text, score) in enumerate(zip(*results))
    # ]
    # print("detecting results:")
    # for q in format_str:
    #     print(q)
