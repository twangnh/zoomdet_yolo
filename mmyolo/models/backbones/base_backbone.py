# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Sequence, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_plugin_layer
from mmdet.utils import ConfigType, OptMultiConfig
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
from torch_cluster import nearest, knn
import time

from mmyolo.registry import MODELS

import random

class NearestInterpolator_Torch(nn.Module):

    def __init__(self, points, values):
        super(NearestInterpolator_Torch, self).__init__()
        self.points = points
        self.values = values

    def __call__(self, points_q):
        nn_torch = nearest(points_q, self.points)
        return self.values[nn_torch]

    # def __call__(self, points_q, n_nearest = 4):
    #     knn_torch = knn(self.points, points_q, n_nearest)
    #     sample_inds = knn_torch[1, :].view(-1, n_nearest)
    #     return self.values[sample_inds].mean(-1)

@MODELS.register_module()
class BaseBackbone(BaseModule, metaclass=ABCMeta):
    """BaseBackbone backbone used in YOLO series.

    .. code:: text

     Backbone model structure diagram
     +-----------+
     |   input   |
     +-----------+
           v
     +-----------+
     |   stem    |
     |   layer   |
     +-----------+
           v
     +-----------+
     |   stage   |
     |  layer 1  |
     +-----------+
           v
     +-----------+
     |   stage   |
     |  layer 2  |
     +-----------+
           v
         ......
           v
     +-----------+
     |   stage   |
     |  layer n  |
     +-----------+
     In P5 model, n=4
     In P6 model, n=5

    Args:
        arch_setting (list): Architecture of BaseBackbone.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        input_channels: Number of input image channels. Defaults to 3.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to None.
        act_cfg (dict): Config dict for activation layer.
            Defaults to None.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch_setting: list,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Sequence[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 plugins: Union[dict, List[dict]] = None,
                 norm_cfg: ConfigType = None,
                 act_cfg: ConfigType = None,
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None,
                 zoomdet_cfg = False):
        super().__init__(init_cfg)
        self.num_stages = len(arch_setting)
        self.arch_setting = arch_setting

        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))

        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('"frozen_stages" must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.input_channels = input_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.widen_factor = widen_factor
        self.deepen_factor = deepen_factor
        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.plugins = plugins

        self.stem = self.build_stem_layer()
        self.layers = ['stem']
        self.zoomdet_cfg = zoomdet_cfg
        if zoomdet_cfg:
            self.offset_conv_layer2 = nn.Conv2d(128, 2, kernel_size=3, padding=1, stride=1, padding_mode='reflect')

        for idx, setting in enumerate(arch_setting):
            stage = []
            stage += self.build_stage_layer(idx, setting)
            if plugins is not None:
                stage += self.make_stage_plugins(plugins, idx, setting)
            self.add_module(f'stage{idx + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{idx + 1}')

        self.force_uniform_saliency=False

        self.size_scaling=3.
        self.saliency_scaling = 1.

    def deformable_grid_generate(self, saliency, rz_shape, device, size_scaling):
        # if self.training:
        #     random_uniform_zoom = random.choice([True, False])
        #     if random_uniform_zoom:
        #         saliency*=0
        # else:
        #     saliency *= 0

        # if self.force_uniform_saliency:
        #     saliency *= 0

        # saliency *= 0
        saliency_h, saliency_w = saliency.shape[2:]

        final_grid = torch.zeros(saliency.shape[0], 2, saliency.shape[2], saliency.shape[3], device=device)
        uniform_grid_y, uniform_grid_x = torch.meshgrid(torch.arange(saliency_h, dtype=torch.float, device=device) + 0.5,
                                                        torch.arange(saliency_w, dtype=torch.float, device=device) + 0.5)

        # uniform_grid = torch.cat((2 * uniform_grid_x.unsqueeze(0) / 100 - 1,
        #                           2 * uniform_grid_y.unsqueeze(0) / 100 - 1)).unsqueeze(0)

        final_grid[:, 0, :, :] = 2 * (uniform_grid_x + saliency[:, 0, :, :]) / saliency.shape[3] - 1
        final_grid[:, 1, :, :] = 2 * (uniform_grid_y + saliency[:, 1, :, :]) / saliency.shape[2] - 1


        saliency_large = F.interpolate(saliency*size_scaling*8., size=rz_shape, mode='bilinear', align_corners=False)
        final_grid_large = torch.zeros(saliency_large.shape[0], 2, saliency_large.shape[2], saliency_large.shape[3], device=device)
        uniform_grid_y_large, uniform_grid_x_large = torch.meshgrid(torch.arange(rz_shape[1], dtype=torch.float, device=device) + 0.5,
                                                                    torch.arange(rz_shape[0], dtype=torch.float, device=device) + 0.5)

        uniform_grid_large = torch.cat((2 * uniform_grid_x_large.unsqueeze(0) / rz_shape[0] - 1,
                                  2 * uniform_grid_y_large.unsqueeze(0) / rz_shape[1] - 1)).unsqueeze(0)

        final_grid_large[:, 0, :, :] = 2 * (uniform_grid_x_large + saliency_large[:, 0, :, :]) / saliency_large.shape[3] - 1
        final_grid_large[:, 1, :, :] = 2 * (uniform_grid_y_large + saliency_large[:, 1, :, :]) / saliency_large.shape[2] - 1

        # final_grid_large = F.interpolate(final_grid, size=rz_shape, mode='bilinear', align_corners=True)
        # uniform_grid_large = F.interpolate(uniform_grid, size=rz_shape, mode='bilinear', align_corners=True)

        # rz_shape = (1600, 1600)
        #
        # uniform_grid_y_large, uniform_grid_x_large = torch.meshgrid(torch.arange(rz_shape[0], dtype=torch.float, device=device) + 0.5,
        #                                                             torch.arange(rz_shape[1], dtype=torch.float, device=device) + 0.5)
        #
        # saliency_large = F.interpolate(saliency, size=rz_shape, mode='bilinear', align_corners=True)
        # final_grid_large_1600 = torch.zeros(saliency_large.shape[0], 2, saliency_large.shape[2], saliency_large.shape[3],
        #                                device=device)
        # final_grid_large_1600[:, 0, :, :] = 2 * (uniform_grid_x_large + saliency_large[:, 0, :, :]) / saliency_large.shape[3] - 1
        # final_grid_large_1600[:, 1, :, :] = 2 * (uniform_grid_y_large + saliency_large[:, 1, :, :]) / saliency_large.shape[2] - 1

        return final_grid.permute(0, 2, 3, 1), uniform_grid_large.permute(0, 2, 3, 1), final_grid_large.permute(0, 2, 3, 1)

    def generate_interpolators(self, grid, scaling=None):
        grid_reorder = grid.permute(3, 0, 1, 2).detach()
        # (N, 800*800)

        u_cor = ((grid_reorder[0, :, :, :] + 1) / 2).view(
            grid_reorder.shape[1], -1)
        v_cor = ((grid_reorder[1, :, :, :] + 1) / 2).view(
            grid_reorder.shape[1], -1)

        ##改到0 H-1范围， 原来是0.5 H-0.5范围
        u_cor, v_cor = u_cor - 1 / grid_reorder.shape[3] / 2., v_cor - 1 / grid_reorder.shape[2] / 2.

        # (800*800)
        x_cor = torch.arange(0, grid_reorder.shape[3], device=grid_reorder.device).unsqueeze(0).expand(
            (grid_reorder.shape[2], grid_reorder.shape[3])).reshape(-1)
        # (N, 800*800)
        x_cor = x_cor.unsqueeze(0).expand(u_cor.shape[0], -1).float()

        y_cor = torch.arange(0, grid_reorder.shape[2], device=grid_reorder.device).unsqueeze(-1).expand(
            (grid_reorder.shape[2], grid_reorder.shape[3])).reshape(-1)
        y_cor = y_cor.unsqueeze(0).expand(u_cor.shape[0], -1).float()

        # x_cor, y_cor = x_cor+0.5, y_cor+0.5
        # v:h, u:w
        # (N, 1, 800*800)
        u_cor, v_cor = u_cor.unsqueeze(1), v_cor.unsqueeze(1)
        # every line in u is (1, u_cor, v_cor)
        # every line in v is (0, u_cor, v_cor)
        u = torch.cat([torch.zeros(u_cor.shape, device=u_cor.device, dtype=u_cor.dtype), u_cor, v_cor], dim=1)
        v = torch.cat([torch.ones(u_cor.shape, device=u_cor.device, dtype=u_cor.dtype), u_cor, v_cor], dim=1)

        # (N, 3, 2*800*800) -> (N, 2*800*800, 3)
        points = torch.cat([u, v], dim=2).transpose(2, 1).float()

        # (N, 2*800*800)
        if scaling is not None:
            values = torch.cat([x_cor/scaling, y_cor/scaling], dim=1)
        else:
            values = torch.cat([x_cor, y_cor], dim=1)

        # range from (0,1)
        # points[:, :, 1:] /= torch.Tensor([rz_shape[1], rz_shape[0]]).to(points.device)

        b, n, dim = points.shape
        # (N, 2*800*800)
        if b == 1:
            batch_inds = torch.zeros((1, n)).to(points.device)
        else:
            batch_inds = torch.arange(b).unsqueeze(1).expand(-1, n).to(points.device)
        # (2*800*800*N, 4), every column means (img_id, u or v, u_cor, v_cor)
        batch_points = torch.cat([batch_inds.view(-1).unsqueeze(1), points.contiguous().view(-1, 3)], dim=1)

        interpolator_all = NearestInterpolator_Torch(batch_points, values.view(-1))
        # end_time = time_synchronized(start_time_2, 'generate_interpolator')

        return interpolator_all


    @abstractmethod
    def build_stem_layer(self):
        """Build a stem layer."""
        pass

    @abstractmethod
    def build_stage_layer(self, stage_idx: int, setting: list):
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        pass

    def make_stage_plugins(self, plugins, stage_idx, setting):
        """Make plugins for backbone ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block``, ``dropout_block``
        into the backbone.


        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True)),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True)),
            ... ]
            >>> model = YOLOv5CSPDarknet()
            >>> stage_plugins = model.make_stage_plugins(plugins, 0, setting)
            >>> assert len(stage_plugins) == 1

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1 -> conv2 -> conv3 -> yyy

        Suppose ``stage_idx=1``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1 -> conv2 -> conv3 -> xxx -> yyy


        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build
                If stages is missing, the plugin would be applied to all
                stages.
            setting (list): The architecture setting of a stage layer.

        Returns:
            list[nn.Module]: Plugins for current stage
        """
        # TODO: It is not general enough to support any channel and needs
        # to be refactored
        in_channels = int(setting[1] * self.widen_factor)
        plugin_layers = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            if stages is None or stages[stage_idx]:
                name, layer = build_plugin_layer(
                    plugin['cfg'], in_channels=in_channels)
                plugin_layers.append(layer)
        return plugin_layers

    def _freeze_stages(self):
        """Freeze the parameters of the specified stage so that they are no
        longer updated."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x: torch.Tensor, zoomDet: bool = False) -> tuple:
        """Forward batch_inputs from the data_preprocessor."""
        outs = []
        if zoomDet:
            for i, layer_name in enumerate(self.layers):
                layer = getattr(self, layer_name)
                x = layer(x)

                if i == 2:
                    saliency_layer2 = self.saliency_scaling * self.offset_conv_layer2(x)
                    saliency = saliency_layer2
                    device = saliency.device
                    feat_size_h, feat_size_w = x.shape[2:]
                    input_size_h, input_size_w = feat_size_h*8, feat_size_w*8
                    grid, uniform_grid_large, grid_large = \
                        self.deformable_grid_generate(saliency, (int(input_size_h*self.size_scaling), int(input_size_w*self.size_scaling)), device, self.size_scaling)

                    x = F.grid_sample(x, grid, padding_mode='reflection')
                if i in self.out_indices:
                    outs.append(x)
            if self.training:
                interpolators = self.generate_interpolators(grid_large,scaling=self.size_scaling)

                return tuple(outs), grid_large, uniform_grid_large, interpolators

            else:
                interpolators = self.generate_interpolators(grid_large,scaling=self.size_scaling)
                return tuple(outs), grid_large, interpolators
                # return tuple(outs), grid_large

        else:
            for i, layer_name in enumerate(self.layers):
                layer = getattr(self, layer_name)
                x = layer(x)
                if i in self.out_indices:
                    outs.append(x)
            return tuple(outs)

