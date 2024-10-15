import gradio as gr


# 获取配置参数并返回字典
def get_config(num_class, area_thr, ratio_thr, top_k, confidence_threshold, *args):
    config = {
        "num_class": num_class,
        "area_thr": area_thr,
        "ratio_thr": ratio_thr,
        "top_k": top_k,
        "confidence_threshold": confidence_threshold,
    }

    return config


# Gradio UI
def create_app():
    with gr.Blocks() as app:

        other_params = {
            'num_class': gr.Number(value=104, label="Number of Classes"),
            'area_thr': gr.Number(value=0, label="Area Threshold"),
            'ratio_thr': gr.Slider(0, 1, value=0.5, label="Ratio Threshold"),
            'top_k': gr.Number(value=80, label="Top K"),
            'confidence_threshold': gr.Slider(0, 1, value=0.5, label="Confidence Threshold")
        }


        # AMG Parameters Group
        amg_params = {
            "points_per_side": gr.Number(value=None, label="Points Per Side"),
            "points_per_batch": gr.Number(value=None, label="Points Per Batch"),
            "pred_iou_thresh": gr.Number(value=None, label="Pred IOU Threshold"),
            "stability_score_thresh": gr.Number(value=None, label="Stability Score Threshold"),
            "stability_score_offset": gr.Number(value=None, label="Stability Score Offset"),
            "box_nms_thresh": gr.Number(value=None, label="Box NMS Threshold"),
            "crop_n_layers": gr.Number(value=None, label="Crop N Layers"),
            "crop_nms_thresh": gr.Number(value=None, label="Crop NMS Threshold"),
            "crop_overlap_ratio": gr.Number(value=None, label="Crop Overlap Ratio"),
            "crop_n_points_downscale_factor": gr.Number(value=None, label="Crop N Points Downscale Factor"),
            "min_mask_region_area": gr.Number(value=None, label="Min Mask Region Area")
        }

        other_inputs = list(other_params.values())
        amg_inputs = list(amg_params.values())  # 获取所有AMG参数

        # 生成配置按钮
        generate_button = gr.Button("Generate Config")
        config_output = gr.JSON(label="Generated Config")

        # 点击按钮生成配置
        generate_button.click(fn=get_config,
                              inputs=other_inputs + amg_inputs,
                              outputs=config_output)

    return app


app = create_app()
app.launch(share=True)
