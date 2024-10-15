import os

import torch
import uvicorn
import yaml
from fastapi import FastAPI
import gradio as gr
from flow import WORLD_SAM, ALL_SAM

app = FastAPI()

cfg_mapping = {
    'WORLD-SAM': WORLD_SAM,
    'ALL-SAM': ALL_SAM
}

if __name__ == '__main__':
    os.chdir("F:/Github/YOLO-World-SAM-RAM")  # 设置工作路径
    with open('base_config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    interface_list, tab_names = [], []
    for key, value in cfg.items():
        interface_list.append(cfg_mapping[key](value, device))
        tab_names.append(key)
    interface = gr.TabbedInterface(interface_list, tab_names)
    app = gr.mount_gradio_app(app, interface, path='', allowed_paths=['./'])
    uvicorn.run(app, access_log=False)