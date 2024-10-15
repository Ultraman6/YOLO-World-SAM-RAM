import argparse
import os.path as osp
import os
import shutil
import tempfile
from typing import List, Dict, Any

import cv2
import mmcv
import mmengine
import torch
import numpy as np
import sys

from mmengine import build_from_cfg
from mmengine.dataset import DefaultSampler
from mmengine.utils.dl_utils.parrots_wrapper import DataLoader

from mmseg.registry import DATASETS

sys.path.append('.')
from mmcv.image import tensor2imgs
from mmengine.model.wrappers.distributed import DataParallel, MMDistributedDataParallel
from mmengine.runner import load_checkpoint
from mmseg.apis import inference_model, init_model
from mmseg.models import build_segmentor

def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    os.makedirs(os.path.join(path, "sam_mask"), exist_ok=True)
    masks_array = []
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        masks_array.append(mask.copy())
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, "sam_mask" ,filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)

    masks_array = np.stack(masks_array, axis=0)
    np.save(os.path.join(path, "sam_mask" ,"masks.npy"), masks_array)
    metadata_path = os.path.join(path, "sam_metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))
    return


def save_result(img_path,
                result,
                color_list_path,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None,
                vis_save_name='pred_vis.png',
                mask_save_name='pred_mask.png'):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (Tensor): The semantic segmentation results to draw over
            `img`.
        color_list_path: path of (list[list[int]]] | np.ndarray | None): The palette of
            segmentation map.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """
    img = mmcv.imread(img_path)
    img = img.copy()
    seg = result.pred_sem_seg.data.cpu().numpy()[0]
    masks, classes, colors = [], [], []
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    color_list = np.load(color_list_path)  # get id-color
    color_list[0] = [238, 239, 20]  # set special backgrond color

    for label, color in enumerate(color_list):  # seg的第一维即是label的id
        color_seg[seg == label, :] = color_list[label]
        mask = np.zeros((seg.shape[0], seg.shape[1]), dtype=np.uint8)
        mask[seg == label] = 255
        if np.any(mask == 255):
            masks.append(mask)
            classes.append(label)
            colors.append(color)

    # convert to BGR
    # color_seg = color_seg[..., ::-1]
    # 蒙版分割效果
    img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False

    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        vis_path = os.path.join(out_file, vis_save_name)
        mask_path = os.path.join(out_file, mask_save_name)
        mmcv.imwrite(img, vis_path)
        mmcv.imwrite(seg, mask_path)
        return (vis_path, mask_path), (masks, classes)

    if not (show or out_file):
        print('show==False and out_file is not specified, only '
              'result image will be returned')
        return img


def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test(model,
                    data_loader,
                    color_list_path,
                    show=False,
                    out_dir=None,
                    efficient_test=False, ):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'].split('.')[0])
                else:
                    out_file = None

                save_result(
                    img_show,
                    result,
                    color_list_path=color_list_path,
                    show=show,
                    out_file=out_file)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def semantic_predict(config, options, aug_test, checkpoint,
                     color_list_path, img_path, output_dir):
    cfg = mmengine.Config.fromfile(config)
    if options is not None:
        cfg.merge_from_dict(options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    model = init_model(config, checkpoint)
    load_checkpoint(model, checkpoint, map_location='cpu')
    img = mmcv.imread(img_path)
    result = inference_model(model, img)
    return save_result(
        img_path,
        result,
        color_list_path=color_list_path,
        show=False,
        out_file=output_dir)


def sam_predict(t, out, generator, logger):
    # 语义分割与sam无关
    image = cv2.imread(t)
    if image is None:
        logger.error(f"Could not load '{t}' as an image, skipping...")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = generator.generate(image)
    base = os.path.basename(t)
    base = os.path.splitext(base)[0]
    save_base = os.path.join(out, base)
    os.makedirs(save_base, exist_ok=True)
    write_masks_to_folder(masks, str(save_base))
    shutil.copyfile(t, os.path.join(str(save_base), "input.jpg"))



