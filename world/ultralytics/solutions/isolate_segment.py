import os
from collections import defaultdict

import torch
from PIL.ImageColor import getrgb
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import matplotlib.colors as mcolors

from world.ultralytics.utils import ops


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
                 isolate_background='black',
                 background_transparent=False,
                 crop_background='black',
                 crop_transparent=False,
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
        self.isolate_background = isolate_background  # 隔离时的背景颜色
        self.isolate_transparent = background_transparent  # 隔离时是否使用透明背景
        self.crop_background = crop_background    # 裁剪时的背景颜色
        self.crop_transparent = crop_transparent  # 裁剪时是否使用透明背景
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

        if self.isolate_transparent:
            # 将图片转换为 RGBA 格式
            isolated = np.dstack([img, mask])
            isolated = cv2.cvtColor(isolated, cv2.COLOR_RGB2RGBA)
        else:
            mask3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            isolated = cv2.bitwise_and(mask3ch, img)
            background_img = cv2.cvtColor(np.full_like(img, self.isolate_background, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            isolated = np.where(mask3ch == 0, background_img, isolated)

        if self.is_cropped:
            x1, y1, x2, y2 = box.astype(np.int32)
            isolated = isolated[y1:y2, x1:x2]
            new_mask = mask[y1:y2, x1:x2]  # 对掩码进行相应裁剪
        else:
            new_mask = mask

        return isolated, new_mask

    def crop_instance(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        在原始图像中裁剪掉分割部分，并根据背景设置填充被裁剪掉的部分.

        Args:
            img (np.ndarray): 原始图片.
            mask (np.ndarray): 分割掩码，实例为1，背景为0.
        Returns:
            np.ndarray: 裁剪并填充后的图像.
        """
        # 确保掩码和图像维度一致
        if img.shape[:2] != mask.shape:
            raise ValueError("Mask and image dimensions must match.")

        # 处理背景填充
        if self.crop_transparent:
            if img.shape[2] != 4:
                # 如果图像不是 RGBA 格式，转换为 RGBA
                cropped = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            else:
                cropped = img.copy()
            # 设置透明度通道，掩码部分透明
            alpha_channel = np.where(mask == 1, 0, 255).astype(np.uint8)
            cropped[:, :, 3] = alpha_channel
        else:
            # 获取背景颜色并创建背景图像
            # background_color = getrgb(self.crop_background)
            background_img = np.full_like(img, self.crop_background, dtype=np.uint8)
            # 创建掩码的反掩码，掩码部分为0，背景部分为1
            mask_inv = (mask == 0).astype(np.uint8)
            # 将掩码扩展到与图像通道数一致
            if len(img.shape) == 3 and img.shape[2] == 3:
                mask_inv_3ch = np.repeat(mask_inv[:, :, np.newaxis], 3, axis=2)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                mask_inv_3ch = np.repeat(mask_inv[:, :, np.newaxis], 4, axis=2)
            else:
                mask_inv_3ch = mask_inv
            # 用掩码选择保留原始图像或背景颜色
            cropped = np.where(mask_inv_3ch == 1, img, background_img)

        return cropped

    def _set(self, back_color, back_trans, crop_color, crop_trans, is_cropped):
        if back_color is not None:
            self.isolate_background = back_color
        if back_trans:
            self.isolate_transparent = back_trans
        if  crop_color is not None:
            self.crop_background = crop_color
        if crop_trans is  not None:
            self.crop_transparent = crop_trans
        if is_cropped is not None:
            self.is_cropped = is_cropped


    def process(self, image, results, classes_to_iso=None, *params):
        """
        主处理函数，隔离实例并进行裁剪.

        Args:
            image (np.ndarray): 输入图片.
            results (ultralytics YOLO results): YOLO model's detection results.

        Returns:

        """
        if params is not None:
            self._set(*params)
        res_dict, cropped_img = defaultdict(list), None
        boxes, classes, confs, masks = results
        masks = self._xy(masks, image.shape[:-1])
        # 遍历每个分割的实例
        for idx, (box, cls_idx, mask) in enumerate(zip(boxes, classes, masks)):
            # class_name = results[0].names[int(cls)]
            cname = self.names[int(cls_idx)]
            if classes_to_iso is not None and cname not in classes_to_iso:
                continue
            # 创建掩码
            b_mask = np.zeros(image.shape[:2], np.uint8)
            contour = mask.astype(np.int32).reshape(-1, 1, 2)
            _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

            # 隔离实例
            isolated_img, new_mask = self.isolate_instance(image, b_mask, box)
            # isolated_img = self._draw(isolated_img, new_mask, cls_idx, device)

            # 根据边界框裁剪实例
            cropped_img = self.crop_instance(image if cropped_img is None else cropped_img, b_mask)
            # isolated_img = cv2.cvtColor(isolated_img, cv2.COLOR_BGRA2RGBA)
            res_dict[cname].append(isolated_img)

        if self.show:
            self._show(res_dict, cropped_img)

        self._save(res_dict, cropped_img)

        return res_dict, cropped_img

    # def _draw(self, img, mask, cls_id, device):
    #     # 获取颜色，并使用 self.tf_color 控制透明度
    #     color = colors(int(cls_id), True)
    #     annotator = Annotator(img)
    #     im_gpu = (
    #             torch.as_tensor(img, dtype=torch.float16, device=device)
    #             .permute(2, 0, 1)
    #             .flip(0)
    #             .contiguous()
    #             / 255
    #     )
    #     print(mask.shape)
    #     annotator.masks(torch.tensor([mask], device=device), [color], im_gpu,
    #                     alpha=self.tf_color, retina_masks=self.hg_res)
    #
    #     return img

    def _xy(self, mask, orig_shape):
        if type(mask) is list:
            return mask
        elif type(mask) is np.ndarray:
            mask = torch.from_numpy(mask)
            return [
                ops.scale_coords(mask.shape[1:], x, orig_shape, normalize=False)
                for x in ops.masks2segments(mask)
            ]
        else:
            raise ValueError("Invalid mask type.")

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
        # isolate_background='transparent',
        # crop_background='transparent',
        save_isolated=True,
        save_cropped=True,
        # show=True,
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
