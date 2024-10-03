import os
import torch
from PIL import Image
from world.groundingdino.models import build_model
from world.groundingdino.util.slconfig import SLConfig
from world.groundingdino.util.utils import clean_state_dict
from world.groundingdino.util.inference import annotate, load_image, predict
import world.groundingdino.datasets.transforms as T

# Use this command for evaluate the Grounding DINO model

def image_transform_grounding(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return init_image, image

def image_transform_grounding_for_vis(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return image

DINO_PRETRAIN_PATH = "world/groundingdino/config/"
DINO_WEIGHTS_PATH = "weights/world/dino/"


# 本模型只接受text，先要执行text2tag任务
class Ground_Dino:
    def  __init__(self, model_id):
        model_config_path = os.path.join(DINO_PRETRAIN_PATH, f'{model_id}.py')
        filename = os.path.join(DINO_WEIGHTS_PATH , f'{model_id}.pth')
        args = SLConfig.fromfile(model_config_path)
        self.model = build_model(args)
        checkpoint = torch.load(filename, map_location='cuda')
        log = self.model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(filename, log))
        _ = self.model.eval()

    def set_classes(self, classes):
        self.names = classes

    def infer(self, input_image, box_threshold, text_threshold):
        init_image = Image.fromarray(input_image).convert("RGB")
        original_size = init_image.size
        _, image_tensor = image_transform_grounding(init_image)
        image_pil: Image = image_transform_grounding_for_vis(init_image)
        # 逐类检测
        all_boxes, all_confs, all_classes = [], [], []
        for grounding_caption in self.names:
            boxes, logits, phrases = predict(self.model, image_tensor, ''.join(grounding_caption), box_threshold, text_threshold,
                                             device='cpu')
            all_boxes.append(boxes)  # 将torch tensor添加到列表中
            all_confs.append(logits)  # 合并list
            all_classes.extend(phrases)  # 合并list
        if all_boxes:
            all_boxes = torch.cat(all_boxes, dim=0)  # 按维度0拼接
        if all_confs:
            all_confs = torch.cat(all_confs, dim=0)  # 按维度0拼接


        return image_pil, (all_boxes, all_confs, all_classes)

    def image_transform_grounding(self, init_image):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image, _ = transform(init_image, None)  # 3, h, w
        return init_image, image

    def image_transform_grounding_for_vis(self, init_image):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
        ])
        image, _ = transform(init_image, None)  # 3, h, w
        return image
