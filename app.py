"""Fast text to segmentation with yolo-world and efficient-vit sam."""
import copy
import json
import os
from typing import Tuple

import cv2
import gradio as gr
import numpy as np
import supervision as sv
import torch
import yaml
from PIL import Image
from inference.models import YOLOWorld
from torch import nn
from ram import inference_ram_openset as inference
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_sam_model
from ram import get_transform
from ram.models import ram_plus
from ram.utils import build_openset_llm_label_embedding
# import freezegun
# freezegun.configure(extend_ignore_list=["pydantic"])

# Download model weights.
os.system("make model")

# Load configuration from YAML file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def load_models(m_cfg):

    yolo_world = YOLOWorld(model_id=m_cfg['yolo_world']['model_id'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = EfficientViTSamPredictor(
        create_sam_model(name=m_cfg['sam']['name'], weight_url=m_cfg['sam']['weight_url']).to(device).eval()
    )
    #######load model
    ramp = ram_plus(pretrained=m_cfg['ram']['pretrained'],
                     image_size=m_cfg['ram']['image_size'],
                     vit=m_cfg['ram']['vit'])

    #######set openset interference
    print('Building tag embedding:')
    with open(m_cfg['ram']['llm_tag_des'], 'rb') as fo:
        llm_tag_des = json.load(fo)
    openset_label_embedding, openset_categories = build_openset_llm_label_embedding(llm_tag_des)

    ramp.tag_list = np.array(openset_categories)

    ramp.label_embed = nn.Parameter(openset_label_embedding.float())

    ramp.num_class = len(openset_categories)
    # the threshold for unseen categories is often lower
    ramp.class_threshold = torch.ones(ramp.num_class) * 0.5
    #######
    return yolo_world, sam, ramp

# Load annotators.
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_world, sam, ramp = load_models(config['model'])
ramp.eval()
ramp.to(device)
transform = get_transform(image_size=config['transform']['image_size'])


def recognize(
        image: np.ndarray,
) -> Tuple:
    image = transform(Image.fromarray(image)).unsqueeze(0).to(device)
    labels = inference(image, ramp)
    return labels, copy.deepcopy(labels)

# 检测函数
def detect(
    image: np.ndarray,
    query: str,
    confidence_threshold: float,
    nms_threshold: float,
) -> Tuple:
    # Preparation.
    categories = [category.strip() for category in query.split(",")]
    yolo_world.set_classes(categories)
    print("categories:", categories)
    # Object detection.
    results = yolo_world.infer(image, confidence=confidence_threshold)
    detections = sv.Detections.from_inference(results).with_nms(
        class_agnostic=True, threshold=nms_threshold
    )
    print("detected:", detections)
    # Annotation.
    output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    labels = [
        f"{categories[class_id]}: {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    # 返回检测后的图像
    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), detections

# 分割函数
def segment(
    image: np.ndarray,
    detections: sv.Detections
) -> np.ndarray:
    # Segmentation.
    sam.set_image(image, image_format="RGB")
    masks = []
    for xyxy in detections.xyxy:
        mask, _, _ = sam.predict(box=xyxy, multimask_output=False)
        masks.append(mask.squeeze())
    detections.mask = np.array(masks)
    print("masks shaped as", detections.mask.shape)

    # 将分割掩码应用到图像上
    output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)

    # 返回带分割掩码的图像
    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)


# Gradio界面
def interface():
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                # 输入图像组件
                input_image = gr.Image(type="numpy", label="Input Image")
                # 识别插槽
                recognize_label = gr.Textbox(label="Recognized Objects (as query for detection)")
                recognize_button = gr.Button("Recognize")

            with gr.Column():
                # 检测插槽
                detected_image = gr.Image(type="numpy", label="Detected Image")
                detect_label = gr.Textbox(label="Detected Objects")
                detect_ct = gr.Slider(minimum=0, maximum=1, value=0.3, step=0.01, label="Confidence Threshold")
                detect_nms = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="NMS Threshold")
                detect_button = gr.Button("Detect")
                detections = gr.State()  # 状态保存，存储Python普通对象(检测结果)

            with gr.Column():
                # 分割结果展示
                segmented_image = gr.Image(type="numpy", label="Segmented Image")
                segment_button = gr.Button("Segment")


        # 识别功能绑定
        recognize_button.click(
            fn=recognize,
            inputs=input_image,
            outputs=[recognize_label, detect_label]
        )
        # 检测功能绑定
        detect_button.click(
            fn=detect,
            inputs=[input_image, detect_label, detect_ct, detect_nms],
            outputs=[detected_image, detections]
        )

        # 分割功能绑定
        segment_button.click(
            fn=segment,
            inputs=[input_image, detections],
            outputs=segmented_image
        )

    return app


# 启动应用
interface().launch(server_name=config['gradio']['server_name'],
                   share=config['gradio']['share'])