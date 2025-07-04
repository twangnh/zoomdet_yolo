# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
from mmengine.dist import get_world_size
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.optim import OptimWrapper
from typing import List, Optional, Union
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm, _InstanceNorm
from torch.nn import GroupNorm, LayerNorm
import logging
import torch
from mmengine.utils.dl_utils import mmcv_full_available

from mmyolo.registry import (OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS,
                             OPTIMIZERS)


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class YOLOv5OptimizerConstructor:
    """YOLOv5 constructor for optimizers.

    It has the following functions：

        - divides the optimizer parameters into 3 groups:
        Conv, Bias and BN

        - support `weight_decay` parameter adaption based on
        `batch_size_per_gpu`

    Args:
        optim_wrapper_cfg (dict): The config dict of the optimizer wrapper.
            Positional fields are

                - ``type``: class name of the OptimizerWrapper
                - ``optimizer``: The configuration of optimizer.

            Optional fields are

                - any arguments of the corresponding optimizer wrapper type,
                  e.g., accumulative_counts, clip_grad, etc.

            The positional fields of ``optimizer`` are

                - `type`: class name of the optimizer.

            Optional fields are

                - any arguments of the corresponding optimizer type, e.g.,
                  lr, weight_decay, momentum, etc.

        paramwise_cfg (dict, optional): Parameter-wise options. Must include
            `base_total_batch_size` if not None. If the total input batch
            is smaller than `base_total_batch_size`, the `weight_decay`
            parameter will be kept unchanged, otherwise linear scaling.

    Example:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optim_wrapper_cfg = dict(
        >>>     dict(type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01,
        >>>         momentum=0.9, weight_decay=0.0001, batch_size_per_gpu=16))
        >>> paramwise_cfg = dict(base_total_batch_size=64)
        >>> optim_wrapper_builder = YOLOv5OptimizerConstructor(
        >>>     optim_wrapper_cfg, paramwise_cfg)
        >>> optim_wrapper = optim_wrapper_builder(model)
    """

    def __init__(self,
                 optim_wrapper_cfg: dict,
                 paramwise_cfg: Optional[dict] = None):
        if paramwise_cfg is None:
            paramwise_cfg = {'base_total_batch_size': 64}
        self.paramwise_cfg = {} if paramwise_cfg is None else paramwise_cfg
        assert 'base_total_batch_size' in paramwise_cfg

        if not isinstance(optim_wrapper_cfg, dict):
            raise TypeError('optimizer_cfg should be a dict',
                            f'but got {type(optim_wrapper_cfg)}')
        assert 'optimizer' in optim_wrapper_cfg, (
            '`optim_wrapper_cfg` must contain "optimizer" config')

        self.optim_wrapper_cfg = optim_wrapper_cfg
        self.optimizer_cfg = self.optim_wrapper_cfg.pop('optimizer')
        self.base_total_batch_size = paramwise_cfg['base_total_batch_size']
        self.base_lr = self.optimizer_cfg.get('lr', None)
        self.base_wd = self.optimizer_cfg.get('weight_decay', None)

    def add_params(self,
                   params: List[dict],
                   module: nn.Module,
                   prefix: str = '',
                   is_dcn_module: Optional[Union[int, float]] = None) -> None:
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        # get param-wise options
        custom_keys = self.paramwise_cfg.get('custom_keys', {})
        # first sort with alphabet order and then sort with reversed len of str
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bias_lr_mult = self.paramwise_cfg.get('bias_lr_mult', None)
        bias_decay_mult = self.paramwise_cfg.get('bias_decay_mult', None)
        norm_decay_mult = self.paramwise_cfg.get('norm_decay_mult', None)
        dwconv_decay_mult = self.paramwise_cfg.get('dwconv_decay_mult', None)
        flat_decay_mult = self.paramwise_cfg.get('flat_decay_mult', None)
        bypass_duplicate = self.paramwise_cfg.get('bypass_duplicate', False)
        dcn_offset_lr_mult = self.paramwise_cfg.get('dcn_offset_lr_mult', None)

        # special rules for norm layers and depth-wise conv layers
        is_norm = isinstance(module,
                             (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
        is_dwconv = (
                isinstance(module, torch.nn.Conv2d)
                and module.in_channels == module.groups)

        for name, param in module.named_parameters(recurse=False):
            param_group = {'params': [param]}
            if bypass_duplicate and self._is_in(param_group, params):
                print_log(
                    f'{prefix} is duplicate. It is skipped since '
                    f'bypass_duplicate={bypass_duplicate}',
                    logger='current',
                    level=logging.WARNING)
                continue
            if not param.requires_grad:
                print_log((f'{prefix}.{name} is skipped since its '
                           f'requires_grad={param.requires_grad}'),
                          logger='current',
                          level=logging.WARNING)
                continue

            # if the parameter match one of the custom keys, ignore other rules
            is_custom = False
            for key in sorted_keys:
                if key in f'{prefix}.{name}':
                    is_custom = True
                    lr_mult = custom_keys[key].get('lr_mult', 1.)
                    param_group['lr'] = self.base_lr * lr_mult
                    if self.base_wd is not None:
                        decay_mult = custom_keys[key].get('decay_mult', 1.)
                        param_group['weight_decay'] = self.base_wd * decay_mult
                    # add custom settings to param_group
                    for k, v in custom_keys[key].items():
                        param_group[k] = v
                    break

            if not is_custom:
                # bias_lr_mult affects all bias parameters
                # except for norm.bias dcn.conv_offset.bias
                if name == 'bias' and not (
                        is_norm or is_dcn_module) and bias_lr_mult is not None:
                    param_group['lr'] = self.base_lr * bias_lr_mult

                if (prefix.find('conv_offset') != -1 and is_dcn_module
                        and dcn_offset_lr_mult is not None
                        and isinstance(module, torch.nn.Conv2d)):
                    # deal with both dcn_offset's bias & weight
                    param_group['lr'] = self.base_lr * dcn_offset_lr_mult

                # apply weight decay policies
                if self.base_wd is not None:
                    # norm decay
                    if is_norm and norm_decay_mult is not None:
                        param_group[
                            'weight_decay'] = self.base_wd * norm_decay_mult
                    # bias lr and decay
                    elif (name == 'bias' and not is_dcn_module
                          and bias_decay_mult is not None):
                        param_group[
                            'weight_decay'] = self.base_wd * bias_decay_mult
                    # depth-wise conv
                    elif is_dwconv and dwconv_decay_mult is not None:
                        param_group[
                            'weight_decay'] = self.base_wd * dwconv_decay_mult
                    # flatten parameters except dcn offset
                    elif (param.ndim == 1 and not is_dcn_module
                          and flat_decay_mult is not None):
                        param_group[
                            'weight_decay'] = self.base_wd * flat_decay_mult
            params.append(param_group)
            for key, value in param_group.items():
                if key == 'params':
                    continue
                full_name = f'{prefix}.{name}' if prefix else name
                print_log(
                    f'paramwise_options -- {full_name}:{key}={value}',
                    logger='current')

        if mmcv_full_available():
            from mmcv.ops import DeformConv2d, ModulatedDeformConv2d
            is_dcn_module = isinstance(module,
                                       (DeformConv2d, ModulatedDeformConv2d))
        else:
            is_dcn_module = False
        for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            self.add_params(
                params,
                child_mod,
                prefix=child_prefix,
                is_dcn_module=is_dcn_module)

    def __call__(self, model: nn.Module, mask_param) -> OptimWrapper:
        if is_model_wrapper(model):
            model = model.module
        optimizer_cfg = self.optimizer_cfg.copy()
        weight_decay = optimizer_cfg.pop('weight_decay', 0)

        if 'batch_size_per_gpu' in optimizer_cfg:
            batch_size_per_gpu = optimizer_cfg.pop('batch_size_per_gpu')
            # No scaling if total_batch_size is less than
            # base_total_batch_size, otherwise linear scaling.
            total_batch_size = get_world_size() * batch_size_per_gpu
            accumulate = max(
                round(self.base_total_batch_size / total_batch_size), 1)
            scale_factor = total_batch_size * \
                accumulate / self.base_total_batch_size

            if scale_factor != 1:
                weight_decay *= scale_factor
                print_log(f'Scaled weight_decay to {weight_decay}', 'current')

        params_groups = [], [], []

        for name, v in model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                params_groups[2].append(v.bias)
            # Includes SyncBatchNorm
            if isinstance(v, nn.modules.batchnorm._NormBase):
                params_groups[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                params_groups[0].append(v.weight)

        # Note: Make sure bias is in the last parameter group
        optimizer_cfg['params'] = []
        # conv
        optimizer_cfg['params'].append({
            'params': params_groups[0],
            'weight_decay': weight_decay
        })
        # bn
        optimizer_cfg['params'].append({'params': params_groups[1]})
        # bias
        optimizer_cfg['params'].append({'params': params_groups[2]})

        print_log(
            'Optimizer groups: %g .bias, %g conv.weight, %g other' %
            (len(params_groups[2]), len(params_groups[0]), len(
                params_groups[1])), 'current')
        del params_groups

        # optimizer = OPTIMIZERS.build(optimizer_cfg)
        if not self.paramwise_cfg:
            optimizer = OPTIMIZERS.build(optimizer_cfg)
        else:
            # set param-wise lr and weight decay recursively
            params: List = []
            self.add_params(params, model)
            optimizer_cfg['params'] = params
            optimizer = OPTIMIZERS.build(optimizer_cfg)

        optim_wrapper = OPTIM_WRAPPERS.build(
            self.optim_wrapper_cfg, default_args=dict(optimizer=optimizer))

        return optim_wrapper
