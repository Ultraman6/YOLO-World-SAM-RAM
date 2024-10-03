"""Fast text to segmentation with yolo-world and efficient-vit sam."""
import copy
import os
from typing import Tuple
import gradio as gr
import numpy as np
import supervision as sv
import torch
import uvicorn
import yaml
from PIL import Image
from fastapi import FastAPI

from ram import get_transform
from sam.segment_anything_2.build_sam import SAM2_WEIGHTS_PATH
from world.groundingdino.model import DINO_WEIGHTS_PATH
from world.mm.model import MM_WEIGHTS_PATH
from world.ultralytics.solutions.isolate_segment import IsolateSegment
from model_zoo import (REGISTERED_SAM_MODEL, REGISTERED_RAM_MODEL,
                       REGISTERED_WORLD_MODEL, REGISTERED_MODEL, REGISTERED_NAME, _WORLD, _RAM, _SAM, )
from world.ultralytics.solutions.object_crop import ObjectCropper

# Load configuration from YAML file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Placeholder for models, initially set to None.
yolo_world, sam, ramp = None, None, None
OBJ_CROP = ObjectCropper()
ISO_SEG = IsolateSegment(
     # save_isolated=True, save_cropped=True,
     isolate_output_dir='isolated_results',
     crop_output_dir='cropped_results',
     isolate_background='green',
     crop_background='green',
 )

global crop_box

def _load_world(world_version, world_type, world_url):
    global yolo_world
    yolo_world = _WORLD(world_version, world_url)
    return "World model loaded successfully!"

def _load_sam(sam_version, sam_type, sam_url):
    global sam
    sam = _SAM(sam_version, sam_url)
    return "SAM model loaded successfully!"

def _load_ram(ram_version, ram_type, ram_url, img_size, llm_tag_des):
    global ramp
    ramp = _RAM(ram_version, ram_type, ram_url, img_size, llm_tag_des)
    return "RAM model loaded successfully!"

_load_mapping = {
    'world': _load_world,
    'sam': _load_sam,
    'ram': _load_ram,
}

# Function to handle the model loading with updated parameters.
def load_models(ram_version, ram_type, ram_url, img_size, llm_tag_des,
                world_version, world_type, world_url,
                sam_version, sam_type, sam_url):
    # 检查必要的参数是否提供
    states = []
    states.append(_load_world(world_version, world_type, world_url))
    states.append(_load_sam(sam_version, sam_type, sam_url))
    states.append(_load_ram(ram_version, ram_type, ram_url, img_size, llm_tag_des))

    return '\n'.join(states)

# Recognize function
def recognize(image: np.ndarray, class_threshold: float, llm_tag_des: bool, img_size: int):
    # transform = get_transform(image_size=img_size)
    # image = transform(Image.fromarray(image)).unsqueeze(0).to(device)
    ramp._llm_tag_des(llm_tag_des, class_threshold)
    labels = ramp.infer(image, img_size)
    return labels, copy.deepcopy(labels)

# Detect function
def detect(image: np.ndarray, query: str, confidence_threshold: float,
           nms_threshold: float, max_det: int, use_amp: bool, detect_agnostic: bool) -> Tuple:
    categories = [category.strip() for category in query.split(",")]
    yolo_world.set_classes(categories)  # 类别可以是其他之外的
    ann_img, cls_names, detections = yolo_world.infer(image, score_thr=confidence_threshold,
                                  max_det=max_det, nms_thr=nms_threshold, amp=use_amp, agnostic=detect_agnostic)
    # ann_img = cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB)
    OBJ_CROP.names = cls_names
    if sam is not None:
        sam.set_classes(cls_names)
    return ann_img, detections, gr.update(choices=cls_names)

def crop(image: np.ndarray, detections: sv.Detections, classes: list):
    res_dict = OBJ_CROP.crop_objects(image, detections, classes)
    return res_dict

# Segment function
def segment(image: np.ndarray, detections, masks: bool):
    ann_img, segments, cls_names = sam.infer(image, detections, masks=masks)
    ISO_SEG.names = cls_names
    return ann_img, segments, gr.update(choices=cls_names)


def iso(image: np.ndarray, detections: sv.Detections, classes: list):
    res_dict = OBJ_CROP.crop_objects(image, detections, classes)
    return res_dict


def update_sam(sel):
    # 根据选中的模型名称，返回相应的路径
    file_path = REGISTERED_SAM_MODEL.get(sel, "No path available")
    return file_path


def update_ram(sel):
    # 根据选中的模型名称，返回相应的路径
    file_path = REGISTERED_RAM_MODEL.get(sel, "No path available")
    return file_path


def link_low(state, *params):
    values = REGISTERED_MODEL[state]
    for param in params:
        if param is None:
            return None
        if type(values) is list and param in values:
            return param  # 底层list
        values = values[param]
    if type(values) == dict:
        values = list(values.keys())    # 中层keys
        return gr.update(choices=values, value=None)
    elif type(values) == list:
        return gr.update(choices=values, value=None)  # 下层list
    elif type(values) == str:  # 底层值
        return values

def change_params(pack):
    print(pack)
    k, s = pack.keys()
    key, state = pack[k], pack[s]
    state[k] = key
    return state

# Gradio interface
def interface():
    with gr.Blocks() as app:
        unit_list = []
        load_message = gr.Textbox(label="Load Models", value="Please load models first!")
        with gr.Row():
            # Configuration UI section for the models and inference parameters
            for model, REGISTERED in REGISTERED_MODEL.items():
                param_list = []
                with gr.Column():
                    gr.Markdown(f"### {REGISTERED_NAME[model]} Configuration")
                    model_version = gr.Dropdown(value=config[model]['version'],
                                              choices=list(REGISTERED.keys()),
                                              label='Version')
                    model_type = gr.Dropdown(value=config[model]['type'],
                                           choices=list(REGISTERED[model_version.value].keys()),
                                           label='Type')
                    model_url = gr.State(value=REGISTERED_MODEL[model][model_version.value][model_type.value])

                    @gr.render(inputs=model_url)
                    def render_url(url):
                        if url in REGISTERED_WORLD_MODEL['roboflow']:
                            return None
                        if url in REGISTERED_WORLD_MODEL['mm']:
                            url = os.path.join(MM_WEIGHTS_PATH, f'{url}.pth')
                        if url in REGISTERED_WORLD_MODEL['dino']:
                            url = os.path.join(DINO_WEIGHTS_PATH, f'{url}.pth')
                        if url in REGISTERED_SAM_MODEL['segment_anything_2']:
                            url = os.path.join(SAM2_WEIGHTS_PATH, f'{url}.pt')
                        gr.File(label="Weight File", value=url)

                    model_load = gr.Button(f'Load {REGISTERED_NAME[model]}')

                    model_version.change(
                        fn=link_low,
                        inputs=[gr.State(model), model_version],
                        outputs=[model_type]
                    )
                    model_type.change(
                        fn=link_low,
                        inputs=[gr.State(model), model_version, model_type],
                        outputs=[model_url]
                    )
                    param_list.extend([model_version, model_type, model_url])
                    if model == 'ram':
                        img_size = gr.Dropdown(choices=[224, 384], value=config['ram']['image_size'],
                                               label="RAM Input Size for model")
                        llm_tag_des = gr.Checkbox(label="OpenSet LLM Tag", value=config['ram']['llm_tag_des'])
                        param_list.extend([img_size, llm_tag_des])

                    unit_list.extend(param_list)

                    model_load.click(  # 就地函数
                        fn=_load_mapping[model],
                        inputs=param_list,
                        outputs=load_message
                    )

        load_button = gr.Button("Load Models")

        # Model loading button click event
        load_button.click(
            fn=load_models,
            inputs=unit_list,
            outputs=load_message
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
                detect_det = gr.Slider(minimum=0, maximum=300, value=config['inference']['max_det'], step=1,
                                       label="Max Detections")
                detect_amp = gr.Checkbox(value=config['inference']['use_amp'], label="amp Inference")
                detect_agnostic = gr.Checkbox(value=config['inference']['agnostic_nms'], label="Class Agnostic NMS")

                detect_button = gr.Button("Detect")
                detections = gr.State()  # Store detection results

            with gr.Column():
                segmented_image = gr.Image(type="numpy", label="Segmented Image")
                segment_masks = gr.Checkbox(value=config['inference']['segment_masks'], label="Retina or Multi Masks")
                segment_button = gr.Button("Segment")
                segments = gr.State()  # Store detection results

        with gr.Row():
            with gr.Column():
                crop_label = gr.CheckboxGroup(label="请选择裁剪类别")
                crop_tf = gr.Slider(label='标签厚度', minimum=0, maximum=10, step=1)
                crop_button = gr.Button("Crop")
                crop_res = gr.State({})

                @gr.render(inputs=crop_res)
                def render_crop(res_dict):
                    if len(res_dict) > 0:
                        with gr.Accordion("bbox裁剪结果"):
                            for cls, objs in res_dict.items():
                                with gr.Accordion(label=f"{cls}"):
                                    for i, obj in enumerate(objs):
                                        gr.Image(obj, label=f"{cls} {i}")

            with gr.Column():
                iso_label = gr.CheckboxGroup(label="请选择孤立类别")
                with gr.Row():
                    iso_back_color = gr.ColorPicker(label="孤立背景颜色")
                    iso_back_trans = gr.Checkbox(label="孤立背景颜色透明")
                with gr.Row():
                    iso_crop_color = gr.ColorPicker(label="裁剪背景颜色")
                    iso_crop_trans = gr.Checkbox(label="孤立背景颜色透明")
                iso_is_crop = gr.Checkbox(label="是否裁剪隔离对象")

                iso_button = gr.Button("Iso")
                iso_res = gr.State(())

                @gr.render(inputs=iso_res)
                def render_crop(res):
                    if len(res) > 1:
                        res_dict, whole = res[0], res[1]
                        with gr.Accordion("mask孤立结果"):
                            for cls, objs in res_dict.items():
                                with gr.Accordion(label=f"{cls}"):
                                    for i, obj in enumerate(objs):
                                        gr.Image(obj, label=f"{cls} {i}", type="numpy")
                            with gr.Accordion(label="Whole"):
                                gr.Image(whole, label=f"whole after iso")

        # Recognize function binding
        recognize_button.click(
            fn=recognize,
            inputs=[input_image, class_threshold, llm_tag_des, img_size],
            outputs=[recognize_label, detect_label]
        )

        # Detect function binding
        detect_button.click(
            fn=detect,
            inputs=[input_image, detect_label, detect_ct,
                    detect_nms, detect_det, detect_amp, detect_agnostic],
            outputs=[detected_image, detections, crop_label]
        )

        # Crop function binding
        crop_button.click(
            fn=OBJ_CROP.crop_objects,
            inputs=[input_image, detections,
                    crop_label, crop_tf],
            outputs=[crop_res]
        )

        # Segment function binding
        segment_button.click(
            fn=segment,
            inputs=[input_image, detections, segment_masks],
            outputs=[segmented_image, segments, iso_label]
        )

        # Iso function binding
        iso_button.click(
            fn=ISO_SEG.process,
            inputs=[input_image, segments, iso_label, iso_back_color,
                    iso_back_trans, iso_crop_color, iso_crop_trans, iso_is_crop],
            outputs=[iso_res]
        )

    return app


# Launch application
# interface().launch(server_name='127.0.0.1', share=True, server_port=8080, debug=True)
app = FastAPI()
app = gr.mount_gradio_app(app, interface(), path='')
if __name__ == '__main__':
    os.chdir("F:/Github/YOLO-World-SAM-RAM")  # 设置工作路径
    uvicorn.run(app, access_log=False)