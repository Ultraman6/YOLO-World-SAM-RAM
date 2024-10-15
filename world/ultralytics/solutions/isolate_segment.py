import os
from collections import defaultdict
from typing import List

import torch
from PIL.Image import Image
from PIL.ImageColor import getrgb
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import matplotlib.colors as mcolors

from world.ultralytics.utils import ops

# 定义一个函数，将十六进制颜色转换为 RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_rgb_from_matplotlib(color_name: str) -> tuple:
    """
    使用 matplotlib.colors 将颜色名称或HEX代码转换为RGB.

    Args:
        color_name (str): 颜色名称（如 'red', 'blue'）或HEX代码（如 '#FF0000'）.

    Returns:
        tuple: 转换后的 (R, G, B) 三元组，范围是 0-255.
    """
    try:
        # 获取0-1范围的RGB值
        rgb = mcolors.to_rgb(color_name)
        # 转换为0-255范围内的RGB值
        return tuple(int(c * 255) for c in rgb)
    except ValueError:
        raise ValueError(f"Invalid color name or HEX code: {color_name}")

class IsolateSegment:
    def __init__(self,
                 names=None,
                 isolate_background='#000000',
                 background_transparent=0,
                 crop_background='#000000',
                 crop_transparent=0,
                 is_cropped=False,
                 show=False,
                 save_isolated=False,
                 save_cropped=False,
                 isolate_output_dir=None,
                 crop_output_dir=None):
        """
        初始化 IsolateSegmentation 实例.
        Args:
            save_isolated (bool): 是否保存隔离后的结果.
            save_cropped (bool): 是否保存裁剪后的结果.
            isolate_output_dir (str): 隔离后的结果保存目录.
            crop_output_dir (str): 裁剪后的结果保存目录.
            isolate_background (str): 隔离后的背景类型，'black' 或 'transparent'.
            crop_background (str): 裁剪后的背景类型，'black' 或 'transparent'.
        """
        if names is None:
            names = []
        self.names = names
        self.iso_back = isolate_background  # 隔离时的背景颜色
        self.iso_trans = background_transparent  # 隔离时是否使用透明背景
        self.crop_back = crop_background    # 裁剪时的背景颜色
        self.crop_trans = crop_transparent  # 裁剪时是否使用透明背景
        self.is_cropped = is_cropped  # 是否裁剪隔离对象
        self.save_isolated = save_isolated  # 是否保存隔离后的结果
        self.save_cropped = save_cropped  # 是否保存裁剪后的结果
        self.show = show
        self.isolate_output_dir = isolate_output_dir  # 隔离后的保存路径
        self.crop_output_dir = crop_output_dir  # 裁剪后的保存路径

        # 创建保存目录（如果需要保存）
        if self.save_isolated and self.isolate_output_dir:
            Path(self.isolate_output_dir).mkdir(parents=True, exist_ok=True)
        if self.save_cropped and self.crop_output_dir:
            Path(self.crop_output_dir).mkdir(parents=True, exist_ok=True)

    def isolate_instance(self, img, mask, box):
        """
        根据掩码隔离对象，支持黑色或透明背景.

        Args:
            img (np.ndarray): 原始图片.
            mask (np.ndarray): 实例的掩码.
            box (list or np.ndarray): 边界框 (x1, y1, x2, y2).
        Returns:
            np.ndarray: 隔离后的图像.
        """
        isolated = img.copy()
        if self.iso_back is not None:
            mask3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            isolated = cv2.bitwise_and(mask3ch, isolated)
            background_img = cv2.cvtColor(np.full_like(isolated, hex_to_rgb(self.iso_back)[::-1],
                                                       dtype=np.uint8), cv2.COLOR_RGB2BGR)
            isolated = np.where(mask3ch == 0, background_img, isolated)

        # 将图片转换为 RGBA 格式
        isolated = cv2.cvtColor(isolated, cv2.COLOR_RGB2RGBA)
        alpha_channel = np.where(mask != 0, 255,  # 透明度设置
                                 255 * (1 - self.iso_trans)).astype(np.uint8)
        isolated[:, :, 3] = alpha_channel

        if self.is_cropped:
            x1, y1, x2, y2 = box.astype(np.int32)
            isolated = isolated[y1:y2, x1:x2]
            new_mask = mask[y1:y2, x1:x2]  # 对掩码进行相应裁剪
        else:
            new_mask = mask

        return isolated, new_mask

    def crop_instance(self, img: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
        """
        在原始图像中裁剪掉多个分割部分，并根据背景设置填充被裁剪掉的部分.

        Args:
            img (np.ndarray): 原始图片.
            masks (List[np.ndarray]): 分割掩码列表，每个掩码实例为255，背景为0.

        Returns:
            np.ndarray: 裁剪并填充后的图像.
        """
        # 确保所有的掩码与图像维度一致
        for mask in masks:
            if img.shape[:2] != mask.shape:
                raise ValueError("All masks and image dimensions must match.")

        # 创建一个初始的空白 mask，与图像尺寸相同
        combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # 合并所有的 masks，确保任何实例的 mask 都会被保留为255
        for mask in masks:
            combined_mask = np.maximum(combined_mask, mask)

        # 复制图像，防止对原始图像进行修改
        cropped = img.copy()

        # 处理背景填充
        if self.crop_back is not None:
            # 获取背景颜色并创建背景图像
            background_img = np.full_like(cropped, hex_to_rgb(self.crop_back)[::-1], dtype=np.uint8)
            background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)

            # 创建反掩码（掩码部分为0，背景部分为1）
            mask_inv = (combined_mask == 0).astype(np.uint8)

            # 将反掩码扩展到与图像的通道数一致
            if len(cropped.shape) == 3 and cropped.shape[2] == 3:
                mask_inv_3ch = np.repeat(mask_inv[:, :, np.newaxis], 3, axis=2)
            elif len(cropped.shape) == 3 and cropped.shape[2] == 4:
                mask_inv_3ch = np.repeat(mask_inv[:, :, np.newaxis], 4, axis=2)
            else:
                mask_inv_3ch = mask_inv

            # 使用反掩码，保留原始图像或者用背景图像填充裁剪的部分
            cropped = np.where(mask_inv_3ch == 1, cropped, background_img)

        # 如果图像没有透明通道，添加Alpha通道
        if img.shape[2] != 4:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2RGBA)

        # 使用合并的掩码来设置透明通道
        alpha_channel = np.where(combined_mask == 0, 255, 255 * (1 - self.crop_trans)).astype(np.uint8)
        cropped[:, :, 3] = alpha_channel

        return cropped

    def _set(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


    def process(self, image, results, classes_to_iso=None, **kwargs):

        self._set(**kwargs)
        if type(image) is str:
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res_dict = defaultdict(list)
        boxes, classes, masks = results
        b_masks = []
        # 遍历每个分割的实例
        for idx, (box, cls_idx, mask) in enumerate(zip(boxes, classes, masks)):
            if mask.shape == image.shape[:2]:
                mask = self._xy(mask, image.shape[:-1])
            cname = self.names[int(cls_idx)]
            if classes_to_iso is not None and cname not in classes_to_iso:
                continue
            # 创建掩码
            b_mask = np.zeros(image.shape[:2], np.uint8)
            contour = mask.astype(np.int32).reshape(-1, 1, 2)
            _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
            # 隔离实例
            isolated_img, new_mask = self.isolate_instance(image, b_mask, box)
            b_masks.append(b_mask)
            # 根据边界框裁剪实例
            res_dict[cname].append(isolated_img)

        cropped_img = self.crop_instance(image, b_masks)

        if self.show:
            self._show(res_dict, cropped_img)

        self._save(res_dict, cropped_img)

        return res_dict, cropped_img

    def _xy(self, mask, orig_shape):
        masks = np.array([mask,])
        masks = torch.from_numpy(masks)
        return [
            ops.scale_coords(masks.shape[1:], x, orig_shape, normalize=False)
            for x in ops.masks2segments(masks)
        ][0]

    def annotate_image(self, results):
        """
        Annotates the original image with bounding boxes and class names.
        Args:
            im0 (np.ndarray): Original image to be annotated.
            results (ultralytics YOLO results): YOLO model's detection results.

        Returns:
            np.ndarray: Annotated image.
        """
        boxes = results[0].boxes.xyxy  # Bounding boxes
        classes = results[0].boxes.cls.cpu().numpy()
        masks = results[0].masks
        cls_names = []
        # 遍历每个分割的实例
        for idx, (box, cls, mask) in enumerate(zip(boxes, classes, masks)):
            if cls not in cls_names:
                cls_names.append(cls)

        return results[0].plot(), [self.names[id] for id in cls_names]

    def _save(self, res_dict, cropped_img):
        # 保存隔离结果
        if self.save_isolated and self.isolate_output_dir:
            for cname, imgs in res_dict.items():
                for i, img in enumerate(imgs):
                    output_path = Path(self.isolate_output_dir) / f"{cname}_{i}_isolated.png"
                    cv2.imwrite(str(output_path), img)
                    print(f"Saved isolated image to {output_path}")

        # 保存裁剪结果
        if self.save_cropped and self.crop_output_dir:
            output_path = Path(self.crop_output_dir) / f"cropped.png"
            cv2.imwrite(str(output_path), cropped_img)
            print(f"Saved cropped image to {output_path}")

    def _show(self, res_dict, cropped_img):
        for cname, imgs in res_dict.items():
            for i, img in enumerate(imgs):
                cv2.imshow(f"{cname}_{i}", img)
                cv2.waitKey(0)
        cv2.imshow("cropped", cropped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 使用示例
if __name__ == "__main__":
    # 初始化分割类
    os.chdir("F:/Github/YOLO-World-SAM-RAM")  # 设置工作路径
    # m = YOLO("weights/world/ultralytics/yolov8s-worldv2.pt")
    # m.set_classes(['bus', 'person'])
    m = YOLO("world/ultralytics/solutions/yolov8n-seg.pt")

    segmenter = IsolateSegment(
        m.names,
        background_transparent=True,
        crop_transparent=True,
        save_isolated=True,
        save_cropped=True,
        isolate_output_dir='isolated_results',
        crop_output_dir='cropped_results',
    )

    # 处理图片，假设 res 是外部模型推理后的结果
    img = cv2.imread('world/ultralytics/assets/bus.jpg')
    res = m.predict(img)
    boxes = res[0].boxes.xyxy.cpu().numpy()
    classes = res[0].boxes.cls.cpu().numpy()
    masks = res[0].masks.xy
    confs = res[0].boxes.conf.cpu().numpy()

    segmenter.process(img, (boxes, classes, confs, masks))
