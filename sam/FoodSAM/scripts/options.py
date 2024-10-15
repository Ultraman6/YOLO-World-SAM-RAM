import argparse
from mmengine.config.config import DictAction


# 本项目仅推理
def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Runs SAM automatic mask generation and  instance segmentation on an input image or directory of images, "
        )
    )

    parser.add_argument(
        "--img_path",
        type=str,
        default='F:\Github\datasets\cook_order\Images/1c6c25ea81ff4fc7bdb67c0241c596651699331487343.jpeg',
        help="Path to a single input image.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default='Output',
        help=(
            "Path to the directory where masks will be output. Output will be a folder"
        ),
    )
    parser.add_argument(
        "--SAM_checkpoint",
        type=str,
        default="F:\Github\YOLO-World-SAM-RAM\weights\sam\segment_anything\sam_vit_h_4b8939.pth",
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument('--semantic_config',
                        default="F:\Github\YOLO-World-SAM-RAM\configs/foodsam\SETR_MLA_768x768_80k_base.py",
                        help='test config file path of mmseg')
    parser.add_argument('--semantic_checkpoint',
                        default="F:\Github\YOLO-World-SAM-RAM\weights\SETR_MLA\iter_80000.pth",
                        help='checkpoint file of mmseg')
    parser.add_argument(
        "--model-type",
        type=str,
        default='vit_h',
        help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    )
    parser.add_argument(
        "--mask-enhance",
        type=bool,
        default=False,
        help="whether to enhance the mask in semantic segmentation",
    )
    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')

    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')

    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')

    parser.add_argument('--color_list_path', type=str,
                        default="F:\Github\YOLO-World-SAM-RAM\configs/foodsam/category_id_files\color_list.npy")

    parser.add_argument(
        "--category_txt",
        default="F:\Github\YOLO-World-SAM-RAM\configs/foodsam\category_id_files/foodseg103_category_id.txt" ,
    )
    parser.add_argument(
        "--num_class",
        default=104,
    )
    parser.add_argument(
        "--area_thr",
        default=0 ,
    )
    parser.add_argument(
        "--ratio_thr",
        default=0.5 ,
    )
    parser.add_argument(
        "--top_k",
        default=80 ,
    )

    parser.add_argument(
        "--detection_config",
        default="F:\Github\YOLO-World-SAM-RAM\configs/foodsam/Unified_learned_OCIM_RS200_6x+2x.yaml",
        metavar="FILE",
        help="path to config file",
        )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "F:\Github\YOLO-World-SAM-RAM\weights/UniDet/Unified_learned_OCIM_RS200_6x+2x.pth"] ,
        nargs=argparse.REMAINDER,
    )

    # parser.add_argument(
    #     "--eval",
    #     action='store_true', help='evaluate the semantic results'
    # )

    amg_settings = parser.add_argument_group("AMG Settings")
    amg_settings.add_argument(
        "--points-per-side",
        type=int,
        default=None,
        help="Generate masks by sampling a grid over the image with this many points to a side.",
    )

    amg_settings.add_argument(
        "--points-per-batch",
        type=int,
        default=None,
        help="How many input points to process simultaneously in one batch.",
    )

    amg_settings.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=None,
        help="Exclude masks with a predicted score from the model that is lower than this threshold.",
    )

    amg_settings.add_argument(
        "--stability-score-thresh",
        type=float,
        default=None,
        help="Exclude masks with a stability score lower than this threshold.",
    )

    amg_settings.add_argument(
        "--stability-score-offset",
        type=float,
        default=None,
        help="Larger values perturb the mask more when measuring stability score.",
    )

    amg_settings.add_argument(
        "--box-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding a duplicate mask.",
    )

    amg_settings.add_argument(
        "--crop-n-layers",
        type=int,
        default=None,
        help=(
            "If >0, mask generation is run on smaller crops of the image to generate more masks. "
            "The value sets how many different scales to crop at."
        ),
    )

    amg_settings.add_argument(
        "--crop-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding duplicate masks across different crops.",
    )

    amg_settings.add_argument(
        "--crop-overlap-ratio",
        type=int,
        default=None,
        help="Larger numbers mean image crops will overlap more.",
    )

    amg_settings.add_argument(
        "--crop-n-points-downscale-factor",
        type=int,
        default=None,
        help="The number of points-per-side in each layer of crop is reduced by this factor.",
    )

    amg_settings.add_argument(
        "--min-mask-region-area",
        type=int,
        default=None,
        help=(
            "Disconnected mask regions or holes with area smaller than this value "
            "in pixels are removed by postprocessing."
        ),
    )
    return parser.parse_args()

def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs

