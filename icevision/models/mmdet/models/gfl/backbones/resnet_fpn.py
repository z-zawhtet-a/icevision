__all__ = [
    "gfl_r50_fpn_1x",
    "gfl_r101_fpn_mstrain_2x",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *


class MMDetGFLBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="gfl", **kwargs)


base_config_path = mmdet_configs_path / "gfl"
base_weights_url = (
    "https://download.openmmlab.com/mmdetection/v2.0/gfl"
)

gfl_r50_fpn_1x = MMDetGFLBackboneConfig(
    config_path=base_config_path / "gfl_r50_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/gfl_r50_fpn_1x_coco/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth",
)

gfl_r101_fpn_mstrain_2x = MMDetGFLBackboneConfig(
    config_path=base_config_path / "gfl_r101_fpn_mstrain_2x_coco.py",
    weights_url=f"{base_weights_url}/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth",
)
