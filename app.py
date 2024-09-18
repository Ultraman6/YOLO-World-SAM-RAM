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
from efficientvit.sam_model_zoo import create_sam_model, REGISTERED_SAM_MODEL
from ram import get_transform
from ram.mapping import REGISTERED_RAM, llm_tag_des_url, REGISTERED_RAM_MODEL, REGISTERED_VIT_MODEL
from ram.models import ram_plus
from ram.utils import build_openset_llm_label_embedding

# Load configuration from YAML file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load annotators.
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()

# Placeholder for models, initially set to None.
yolo_world, sam, ramp = None, None, None
transform = None


# Function to load models based on user-provided configuration
def load_models(m_cfg, log_output):
    global yolo_world, sam, ramp
    log_output.append("Starting model loading...")

    try:
        yolo_world = YOLOWorld(model_id=m_cfg['yolo_world']['model_id'])
        log_output.append(f"YOLOWorld model {m_cfg['yolo_world']['model_id']} loaded.")

        sam = EfficientViTSamPredictor(
            create_sam_model(name=m_cfg['sam']['name'], weight_url=m_cfg['sam']['weight_url']).to(device).eval()
        )
        log_output.append(f"SAM model {m_cfg['sam']['name']} loaded.")

        # Load RAM++ model
        ramp = REGISTERED_RAM[m_cfg['ram']['name']](pretrained=m_cfg['ram']['pretrained'],
                                                    image_size=m_cfg['ram']['img_size'], vit=m_cfg['ram']['vit'])

        log_output.append("Models loaded successfully.")
    except Exception as e:
        log_output.append(f"Error loading models: {str(e)}")

    return "\n".join(log_output)


# Function to handle the model loading with updated parameters.
def load_all_models(model_id, sam_name, sam_weight_url, img_size,
                    ram_type, ram_pretrained, llm_tag_des, vit_type):
    global yolo_world, sam, ramp, transform
    log_output = []

    # Checking if files are selected
    if not sam_weight_url or not ram_pretrained:
        return "Please select all required files!"

    # Build model configuration from UI inputs
    model_cfg = {
        'yolo_world': {'model_id': model_id},
        'sam': {'name': sam_name, 'weight_url': sam_weight_url.name},  # Use file name from gr.File()
        'ram': {
            'name': ram_type,
            'pretrained': ram_pretrained.name,  # Use file name from gr.File()
            'llm_tag_des': llm_tag_des,  # Use file name from gr.File()
            'img_size': img_size,
            'vit': vit_type
        }
    }

    # Load models and capture logs
    log_output = load_models(model_cfg, log_output)

    # Set up transform
    try:
        ramp.eval()
        ramp.to(device)
        log_output += "\nTransform set up successfully."
    except Exception as e:
        log_output += f"\nError setting up transform: {str(e)}"

    return log_output
# Recognize function
def recognize(image: np.ndarray, class_threshold: float, llm_tag_des: bool, img_size: int) -> Tuple:
    if llm_tag_des:
        with open(llm_tag_des_url, 'rb') as fo:
            llm_tag = json.load(fo)
        openset_label_embedding, openset_categories = build_openset_llm_label_embedding(llm_tag)
        ramp.tag_list = np.array(openset_categories)
        ramp.label_embed = nn.Parameter(openset_label_embedding.float())
        ramp.num_class = len(openset_categories)
        ramp.class_threshold = torch.ones(ramp.num_class) * class_threshold

    transform = get_transform(image_size=img_size)
    image = transform(Image.fromarray(image)).unsqueeze(0).to(device)
    labels = inference(image, ramp)
    return labels, copy.deepcopy(labels)

# Detect function
def detect(image: np.ndarray, query: str, confidence_threshold: float, nms_threshold: float) -> Tuple:
    categories = [category.strip() for category in query.split(",")]
    yolo_world.set_classes(categories)
    results = yolo_world.infer(image, confidence=confidence_threshold)
    detections = sv.Detections.from_inference(results).with_nms(
        class_agnostic=True, threshold=nms_threshold
    )
    output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    labels = [f"{categories[class_id]}: {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), detections

# Segment function
def segment(image: np.ndarray, detections: sv.Detections) -> np.ndarray:
    sam.set_image(image, image_format="RGB")
    masks = []
    for xyxy in detections.xyxy:
        mask, _, _ = sam.predict(box=xyxy, multimask_output=False)
        masks.append(mask.squeeze())
    detections.mask = np.array(masks)
    output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

def update_sam(sel):
    # 根据选中的模型名称，返回相应的路径
    file_path = REGISTERED_SAM_MODEL.get(sel, "No path available")
    return file_path

def update_ram(sel):
    # 根据选中的模型名称，返回相应的路径
    file_path = REGISTERED_RAM_MODEL.get(sel, "No path available")
    return file_path


# Gradio interface
def interface():
    with gr.Blocks() as app:
        with gr.Row():
            # Configuration UI section for the models and inference parameters
            with gr.Column():
                gr.Markdown("### YOLOWorld Configuration")
                world_type = gr.Dropdown(value=config['model']['yolo_world']['model_id'],
                                         choices=["yolo_world/l", "yolo_world/x"], label="YOLO World Type")
            with gr.Column():
                gr.Markdown("### SAM Configuration")
                sam_type = gr.Dropdown(value=config['model']['sam']['name'],
                                       choices=list(REGISTERED_SAM_MODEL.keys()), label="SAM Model Type")
                sam_weight_url = gr.File(label="SAM Weight File", value=REGISTERED_SAM_MODEL[sam_type.value])
                sam_type.change(
                    fn=update_sam,
                    inputs=sam_type,
                    outputs=sam_weight_url
                )
            with gr.Column():
                gr.Markdown("### RAM++ Configuration")
                ram_type = gr.Dropdown(choices=list(REGISTERED_RAM_MODEL.keys()),
                                       label="RAM Type", value=config['model']['ram']['name'])
                ram_pretrained = gr.File(label="RAM Pretrained Path", value=REGISTERED_RAM_MODEL[ram_type.value])
                ram_type.change(
                    fn=update_ram,
                    inputs=ram_type,
                    outputs=ram_pretrained
                )
                vit_type = gr.Dropdown(choices=REGISTERED_VIT_MODEL,
                                       label="VIT Type", value=config['model']['ram']['vit'])
                img_size = gr.Dropdown(choices=[224, 384], value=config['model']['ram']['image_size'],
                                       label="RAM Input Size for model")
                llm_tag_des = gr.Checkbox(label="OpenSet LLM Tag", value=config['model']['ram']['llm_tag_des'])

        load_button = gr.Button("Load Models")
        load_output = gr.Textbox(label="Model Loading Status")

        # Model loading button click event
        load_button.click(
            fn=load_all_models,
            inputs=[world_type,
                    sam_type, sam_weight_url, img_size,
                    ram_type, ram_pretrained, llm_tag_des,
                    vit_type
                    ],
            outputs=load_output
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", label="Input Image")
                recognize_label = gr.Textbox(label="Recognized Objects (as query for detection)")
                class_threshold = gr.Slider(minimum=0, maximum=1, value=config['inference']['class_threshold'],
                                            step=0.01, label="Class Threshold")
                recognize_button = gr.Button("Recognize")

            with gr.Column():
                detected_image = gr.Image(type="numpy", label="Detected Image")
                detect_label = gr.Textbox(label="Detected Objects")
                detect_ct = gr.Slider(minimum=0, maximum=1, value=config['inference']['confidence_threshold'],
                                      step=0.01, label="Confidence Threshold")
                detect_nms = gr.Slider(minimum=0, maximum=1, value=config['inference']['nms_threshold'], step=0.01,
                                       label="NMS Threshold")
                detect_button = gr.Button("Detect")
                detections = gr.State()  # Store detection results

            with gr.Column():
                segmented_image = gr.Image(type="numpy", label="Segmented Image")
                segment_button = gr.Button("Segment")

        # Recognize function binding
        recognize_button.click(
            fn=recognize,
            inputs=[input_image, class_threshold],
            outputs=[recognize_label, detect_label]
        )

        # Detect function binding
        detect_button.click(
            fn=detect,
            inputs=[input_image, detect_label, detect_ct, detect_nms],
            outputs=[detected_image, detections]
        )

        # Segment function binding
        segment_button.click(
            fn=segment,
            inputs=[input_image, detections],
            outputs=segmented_image
        )

    return app


# Launch application
interface().launch(server_name='127.0.0.1', share=True)
