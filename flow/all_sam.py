import copy
import os.path

import gradio as gr
from flow.world_sam import link_low
from model_zoo import REGISTERED_SAM_MODEL, REGISTERED_NAME, _SAM
from sam.FoodSAM.scripts import _infer
from world.ultralytics.solutions.isolate_segment import IsolateSegment
from world.ultralytics.solutions.object_crop import ObjectCropper

sam=None

def link_task(res_dict):
    return gr.update(choices=res_dict.keys())

def link_cls(task, res_dict):
    if task == 'detection':
        flag1, flag2 = True, False
    elif task in ['semantic', 'enhance', 'instance', 'panoptic']:
        flag1, flag2 = False, True
    else:
        flag1, flag2 = False, False
    return (gr.update(choices=res_dict[task]['names'].values()),
            gr.update(visible=flag1),
            gr.update(visible=flag2))

def infer(*args):
    mode, sam_version, keys = args[:3]
    re_path, data_dict = _infer(mode, sam_version, sam, keys,
                     *args[8:],
                     img_path=args[3],
                     model_type=args[4],
                     SAM_checkpoint=args[5],
                     mask_enhance=args[6],
                     device=args[7]
                     )
    ab_path = []
    for path in re_path:
        base = os.path.basename(path).split('.')[0]
        ab_path.append((path, base))

    return ab_path, data_dict


def _load_sam(sam_version, sam_url):
    global sam
    SAM = _SAM(sam_version, sam_url)
    sam = copy.deepcopy(SAM.sam)
    return "SAM model loaded successfully!"


def ALL_SAM(cfg, device):
    OBJ_CROP = ObjectCropper()
    ISO_SEG = IsolateSegment()

    def crop(image, res_dict, task, sel_names, keys, *params):
        res = res_dict[task]
        results, names = res['results'], res['names']
        kwargs = {k: p for k, p in zip(keys, params)}
        if task == 'detection':
            OBJ_CROP.names = names
            crop_res = OBJ_CROP.crop_objects(image, results, sel_names, **kwargs)
        else:
            ISO_SEG.names = names
            if kwargs['_iso_back']:
                kwargs['iso_back'] = None
            if kwargs['_crop_back']:
                kwargs['crop_back'] = None
            del kwargs['_iso_back'], kwargs['_crop_back']
            crop_res = ISO_SEG.process(image, results, sel_names, **kwargs)

        return task, crop_res

    available_models = ['segment_anything', 'segment_anything_hq', 'segment_anything_2']
    with gr.Blocks() as app:  # 不能把gr相关的代码放在函数内，否则会报错
        device = gr.State(value=device)
        load_message = gr.Textbox(label="Load Models", value="Please load models first!")
        gr.Markdown(f"### ALL_SAM Configuration")
        with gr.Row():
            # Configuration UI section for the models and inference parameters
            with gr.Column():
                gr.Markdown(f"### {REGISTERED_NAME['sam']} Configuration")
                sam_version = gr.Dropdown(value=cfg['sam']['version'],
                                          choices=available_models,
                                          label='Version')
                sam_type = gr.Dropdown(value=cfg['sam']['type'],
                                       choices=list(REGISTERED_SAM_MODEL[sam_version.value].keys()),
                                       label='Type')
                sam_url = gr.State(value=REGISTERED_SAM_MODEL[sam_version.value][sam_type.value])
                load_button = gr.Button("load SAM")
                sam_version.change(
                    fn=link_low,
                    inputs=[gr.State('sam'), sam_version],
                    outputs=[sam_type]
                )
                sam_type.change(
                    fn=link_low,
                    inputs=[gr.State('sam'), sam_version, sam_type],
                    outputs=[sam_url]
                )
                gr.Markdown("### Class Options")
                class_params = {
                    'num_class': gr.Number(value=104, label="Number of Classes"),
                    'area_thr': gr.Number(value=0, label="Area Threshold"),
                    'ratio_thr': gr.Slider(0, 1, value=0.5, label="Ratio Threshold"),
                    'top_k': gr.Number(value=80, label="Top K"),
                    'confidence_threshold': gr.Slider(0, 1, value=0.5, label="Confidence Threshold")
                }
                mode = gr.Dropdown(value='semantic',
                                   choices=['semantic', 'object', 'instance', 'panoptic'],
                                   label='Mode')
                enhance = gr.Checkbox(value=False, label='Enhance Mask')

            with gr.Column():
                gr.Markdown("### AMG Options")
                amg_params = {
                    "points_per_side": gr.Number(value=32, label="Points Per Side", step=1),
                    "points_per_batch": gr.Number(value=64, label="Points Per Batch", step=1),
                    "pred_iou_thresh": gr.Number(value=0.88, minimum=0, maximum=1, label="Pred IOU Threshold"),
                    "stability_score_thresh": gr.Number(value=0.95, minimum=0, maximum=1, label="Stability Score Threshold"),
                    "stability_score_offset": gr.Number(value=1.0, minimum=0, maximum=1, label="Stability Score Offset"),
                    "box_nms_thresh": gr.Number(value=0.7, minimum=0, maximum=1, label="Box NMS Threshold"),
                    "crop_n_layers": gr.Number(value=0, step=1, label="Crop N Layers"),
                    "crop_nms_thresh": gr.Number(value=0.7, minimum=0, maximum=1, label="Crop NMS Threshold"),
                    "crop_overlap_ratio": gr.Number(value=0.7, minimum=0, maximum=1, label="Crop Overlap Ratio"),
                    "crop_n_points_downscale_factor": gr.Number(value=1, step=1, label="Crop N Points Downscale Factor"),
                    "min_mask_region_area": gr.Number(value=0, step=1, label="Min Mask Region Area")
                }

            class_inputs = list(class_params.values())
            amg_inputs = list(amg_params.values())  # 获取所有AMG参数
            keys = gr.State(list(class_params.keys()) + list(amg_params.keys()))

        with gr.Row():
            with gr.Column():
                image = gr.Image(type="filepath", label="Image for inference")
                infer_button = gr.Button("Begin Inference")
            with gr.Column():
                images = gr.Gallery(label="Processed Images", type="filepath")

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    task_choice = gr.Dropdown(label='任务选择')
                    cls_choice = gr.CheckboxGroup(label='类别选择')
                with gr.Row():
                    with gr.Column(visible=False) as box_params:
                        crop_params = {}
                        crop_params['crop_tf'] = gr.Slider(label='标签厚度', minimum=0, maximum=10, step=1)
                        crop_keys = gr.State(list(crop_params.keys()))
                        crop_button = gr.Button("开始提取")
                    with gr.Column(visible=False) as mask_params:
                        iso_params = {}
                        with gr.Row():
                            iso_params['iso_back'] = gr.ColorPicker(label="孤立背景颜色",
                                                                    value=ISO_SEG.iso_back)
                            iso_params['_iso_back'] = gr.Checkbox(label="孤立背景无色")
                            iso_params['iso_trans'] = gr.Slider(minimum=0, maximum=1, label="孤立背景透明度")
                        with gr.Row():
                            iso_params['crop_back'] = gr.ColorPicker(label="裁剪背景颜色",
                                                                     value=ISO_SEG.crop_back)
                            iso_params['_crop_back'] = gr.Checkbox(label="裁剪背景无色")
                            iso_params['crop_trans'] = gr.Slider(minimum=0, maximum=1, label="裁剪背景透明度")
                        iso_params['is_cropped'] = gr.Checkbox(label="是否裁剪隔离对象")
                        iso_keys = gr.State(list(iso_params.keys()))
                        iso_button = gr.Button("开始提取")

            with gr.Column():
                crop_res = gr.State(())
                @gr.render(inputs=crop_res)
                def render_crop(result):
                    if len(result) > 0:
                        name, res = result
                        if name != 'detection':
                            res, crop_img = res
                            with gr.Accordion(label="Whole"):
                                gr.Image(crop_img, label=f"whole after iso", type="numpy")
                        if len(res) > 0:
                            for cls, objs in res.items():
                                with gr.Accordion(label=f"{cls}"):
                                    gr.Gallery(objs, label=f"{cls} results")
        res_state = gr.State()
        res_state.change(
            fn=link_task,
            inputs=res_state,
            outputs=task_choice
        )
        task_choice.change(
            fn=link_cls,
            inputs=[task_choice, res_state],
            outputs=[cls_choice, box_params, mask_params]
        )
        crop_button.click(
            fn=crop,
            inputs=[image, res_state, task_choice, cls_choice, crop_keys]
                   + list(crop_params.values()),
            outputs=crop_res
        )
        iso_button.click(
            fn=crop,
            inputs=[image, res_state, task_choice, cls_choice, iso_keys]
                   + list(iso_params.values()),
            outputs=crop_res
        )
        load_button.click(
            fn=_load_sam,
            inputs=[sam_version, sam_url],
            outputs=load_message
        )
        # Model loading button click event
        infer_button.click(
            fn=infer,
            inputs=[mode, sam_version, keys, image, sam_type, sam_url, enhance, device]
                   + class_inputs + amg_inputs,
            outputs=[images, res_state]
        )

    return app
