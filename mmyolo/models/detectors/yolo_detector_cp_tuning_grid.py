# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.dist import get_world_size
from mmengine.logging import print_log

from mmyolo.registry import MODELS
from typing import List, Tuple, Union
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from .grid_generator import GridGenerator
import torch.nn.functional as F
import cv2
import math
import copy
from .saliency_loss import saliency_loss
from math import floor, ceil
import time

def time_synchronized(t1=None, m=None):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t2 = time.time()
    if m is not None and t1 is not None:
        print('timing {} is {}'.format(m, t2 - t1))
    return t2

def unwarp_bboxes(bboxes, img_shape, grid, ori_shape):
    """Unwarps a tensor of bboxes of shape (n, 4) or (n, 5) according to the grid \
    of shape (h, w, 2) used to warp the corresponding image and the \
    output_shape (H, W, ...)."""
    bboxes = bboxes.clone()
    # image map of unwarped (x,y) coordinates
    img = grid.detach().permute(2, 0, 1).unsqueeze(0)

    warped_height, warped_width = img_shape[0], img_shape[1]
    # xgrid = 2 * (bboxes[:, 0:4:2] / (warped_width - 1)) - 1
    # ygrid = 2 * (bboxes[:, 1:4:2] / (warped_height - 1)) - 1
    xgrid = 2 * (bboxes[:, 0:4:2] / (warped_width - 0)) - 1
    ygrid = 2 * (bboxes[:, 1:4:2] / (warped_height - 0)) - 1
    grid = torch.stack((xgrid, ygrid), dim=2).unsqueeze(0)

    # warped_bboxes has shape (2, num_bboxes, 2)
    warped_bboxes = F.grid_sample(
        img, grid, align_corners=False, padding_mode='reflection').squeeze(0)
    # bboxes[:, 0:4:2] = (warped_bboxes[0] + 1) / 2 * (ori_shape[1] - 1)
    # bboxes[:, 1:4:2] = (warped_bboxes[1] + 1) / 2 * (ori_shape[0] - 1)
    bboxes[:, 0:4:2] = (warped_bboxes[0] + 1) / 2 * (ori_shape[1] - 0)
    bboxes[:, 1:4:2] = (warped_bboxes[1] + 1) / 2 * (ori_shape[0] - 0)

    return bboxes

def warp_bboxes(batch_gt_bboxes, interpolators, ori_shape):
    """Warps a tensor of gt bboxes to the resample image"""
    device = batch_gt_bboxes.device

    for i in range(2):
        gt_xyxy = copy.deepcopy(batch_gt_bboxes[batch_gt_bboxes[:, 0] == i, 2:])
        if gt_xyxy.shape[0] == 0:
            continue
        else:
            gt_xyxy[:, 0] /= ori_shape[1]
            gt_xyxy[:, 2] /= ori_shape[1]
            gt_xyxy[:, 1] /= ori_shape[0]
            gt_xyxy[:, 3] /= ori_shape[0]

            # box transform
            y1new = torch.cat((torch.tensor([i], dtype=torch.int).unsqueeze(0).expand(gt_xyxy.shape[0], 1).to(device),
                               torch.ones(gt_xyxy.shape[0]).unsqueeze(-1).to(device),
                               gt_xyxy[:, :2]), dim=-1)
            x1new = torch.cat((torch.tensor([i], dtype=torch.int).unsqueeze(0).expand(gt_xyxy.shape[0], 1).to(device),
                               torch.zeros(gt_xyxy.shape[0]).unsqueeze(-1).to(device),
                               gt_xyxy[:, :2]), dim=-1)
            y2new = torch.cat((torch.tensor([i], dtype=torch.int).unsqueeze(0).expand(gt_xyxy.shape[0], 1).to(device),
                               torch.ones(gt_xyxy.shape[0]).unsqueeze(-1).to(device),
                               gt_xyxy[:, 2:]), dim=-1)
            x2new = torch.cat((torch.tensor([i], dtype=torch.int).unsqueeze(0).expand(gt_xyxy.shape[0], 1).to(device),
                               torch.zeros(gt_xyxy.shape[0]).unsqueeze(-1).to(device),
                               gt_xyxy[:, 2:]), dim=-1)

            gt_sampled_xyxy = torch.stack([interpolators(x1new).to(device),
                                           interpolators(y1new).to(device),
                                           interpolators(x2new).to(device),
                                           interpolators(y2new).to(device)], dim=-1)

            x_equal_mask = gt_sampled_xyxy[:, 0] == gt_sampled_xyxy[:, 2]
            y_equal_mask = gt_sampled_xyxy[:, 1] == gt_sampled_xyxy[:, 3]

            gt_sampled_xyxy[x_equal_mask, 2] += 1
            gt_sampled_xyxy[y_equal_mask, 3] += 1

            batch_gt_bboxes[batch_gt_bboxes[:, 0] == i, 2:] = gt_sampled_xyxy

    return batch_gt_bboxes[:, 2:]

def warp_bboxes_test(batch_gt_bboxes, interpolators, ori_shape):
    """Warps a tensor of gt bboxes to the resample image"""
    device = batch_gt_bboxes[0].device

    transformed_gt_bboxes = []

    gt_xyxy = copy.deepcopy(batch_gt_bboxes)
    if gt_xyxy.shape[0] == 0:
        transformed_gt_bboxes.append(gt_xyxy)
    else:
        gt_xyxy[:, 0] /= ori_shape[1]
        gt_xyxy[:, 2] /= ori_shape[1]
        gt_xyxy[:, 1] /= ori_shape[0]
        gt_xyxy[:, 3] /= ori_shape[0]

        # box transform
        y1new = torch.cat((torch.tensor([0], dtype=torch.int).unsqueeze(0).expand(gt_xyxy.shape[0], 1).to(device),
                           torch.ones(gt_xyxy.shape[0]).unsqueeze(-1).to(device),
                           gt_xyxy[:, :2]), dim=-1)
        x1new = torch.cat((torch.tensor([0], dtype=torch.int).unsqueeze(0).expand(gt_xyxy.shape[0], 1).to(device),
                           torch.zeros(gt_xyxy.shape[0]).unsqueeze(-1).to(device),
                           gt_xyxy[:, :2]), dim=-1)
        y2new = torch.cat((torch.tensor([0], dtype=torch.int).unsqueeze(0).expand(gt_xyxy.shape[0], 1).to(device),
                           torch.ones(gt_xyxy.shape[0]).unsqueeze(-1).to(device),
                           gt_xyxy[:, 2:]), dim=-1)
        x2new = torch.cat((torch.tensor([0], dtype=torch.int).unsqueeze(0).expand(gt_xyxy.shape[0], 1).to(device),
                           torch.zeros(gt_xyxy.shape[0]).unsqueeze(-1).to(device),
                           gt_xyxy[:, 2:]), dim=-1)

        gt_sampled_xyxy = torch.stack([interpolators(x1new).to(device),
                                       interpolators(y1new).to(device),
                                       interpolators(x2new).to(device),
                                       interpolators(y2new).to(device)], dim=-1)

        x_larger_mask = gt_sampled_xyxy[:, 0] >= gt_sampled_xyxy[:, 2]
        y_larger_mask = gt_sampled_xyxy[:, 1] >= gt_sampled_xyxy[:, 3]

        if x_larger_mask.any() or y_larger_mask.any():
            print('except y x larger mask')

        transformed_gt_bboxes.append(gt_sampled_xyxy)

    return transformed_gt_bboxes

def compute_rz_shape(ori_shapes):
    rz_width = 0
    rz_height = 0
    for ori_shape in ori_shapes:
        scale_factor = min(1333 / max(ori_shape[0], ori_shape[1]), 800 / min(ori_shape[0], ori_shape[1]))
        rz_height = max(rz_height, int(ori_shape[0] * float(scale_factor) + 0.5))
        rz_width = max(rz_width, int(ori_shape[1] * float(scale_factor) + 0.5))

    # 将最大宽度和最大高度调整为32的倍数
    rz_width = math.ceil(rz_width / 32) * 32
    rz_height = math.ceil(rz_height / 32) * 32

    return tuple([rz_height, rz_width])

def invert_nonseparable_grid(grid, input_shape):
    grid = grid.detach().clone()
    device = grid.device
    _, _, H, W = input_shape
    B, grid_H, grid_W, _ = grid.shape
    assert B == input_shape[0]

    eps = 1e-8
    grid[:, :, :, 0] = (grid[:, :, :, 0] + 1) / 2 * (W - 1)
    grid[:, :, :, 1] = (grid[:, :, :, 1] + 1) / 2 * (H - 1)
    # grid now ranges from 0 to ([H or W] - 1)
    # TODO: implement batch operations
    uniform_grid_y, uniform_grid_x = torch.meshgrid(torch.arange(H, device=device),
                                                    torch.arange(W, device=device))

    uniform_grid_x = uniform_grid_x.float()
    uniform_grid_y = uniform_grid_y.float()
    uniform_grid = torch.cat((uniform_grid_x.unsqueeze(0) / (W - 1) * (grid_W - 1),
                              uniform_grid_y.unsqueeze(0) / (H - 1) * (grid_H - 1))).unsqueeze(0)
    inverse_grid = uniform_grid.repeat(B, 1, 1, 1).permute(0, 2, 3, 1)

    # inverse_grid = 2 * max(H, W) * torch.ones(
    #     (B, H, W, 2), dtype=torch.float32, device=device)
    for b in range(B):
        # each of these is ((grid_H - 1)*(grid_W - 1)) x 2
        p00 = grid[b,  :-1,   :-1, :].contiguous().view(-1, 2)  # noqa: 203
        p10 = grid[b, 1:  ,   :-1, :].contiguous().view(-1, 2)  # noqa: 203
        p01 = grid[b,  :-1,  1:  , :].contiguous().view(-1, 2)  # noqa: 203
        p11 = grid[b, 1:  ,  1:  , :].contiguous().view(-1, 2)  # noqa: 203

        ref = torch.floor(p00).type(torch.int)
        v00 = p00 - ref
        v10 = p10 - ref
        v01 = p01 - ref
        v11 = p11 - ref

        min_x = int(floor(min(v00[:, 0].min(), v10[:, 0].min()) - eps))
        max_x = int(ceil(max(v01[:, 0].max(), v11[:, 0].max()) + eps))
        min_y = int(floor(min(v00[:, 1].min(), v01[:, 1].min()) - eps))
        max_y = int(ceil(max(v10[:, 1].max(), v11[:, 1].max()) + eps))

        pts = torch.cartesian_prod(
            torch.arange(min_x, max_x + 1, device=device),
            torch.arange(min_y, max_y + 1, device=device),
        ).T

        # each of these is  ((grid_H - 1)*(grid_W - 1)) x 2
        vb = v10 - v00
        vc = v01 - v00
        vd = v00 - v10 - v01 + v11

        vx = pts.permute(1, 0).unsqueeze(0)  # 1 x (x_range*y_range) x 2
        Ma = v00.unsqueeze(1) - vx  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range) x 2

        vc_cross_vd = (vc[:, 0] * vd[:, 1] - vc[:, 1] * vd[:, 0]).unsqueeze(1)  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x 1
        vc_cross_vb = (vc[:, 0] * vb[:, 1] - vc[:, 1] * vb[:, 0]).unsqueeze(1)  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x 1
        Ma_cross_vd = (Ma[:, :, 0] * vd[:, 1].unsqueeze(1) - Ma[:, :, 1] * vd[:, 0].unsqueeze(1))  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range)
        Ma_cross_vb = (Ma[:, :, 0] * vb[:, 1].unsqueeze(1) - Ma[:, :, 1] * vb[:, 0].unsqueeze(1))  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range)

        qf_a = vc_cross_vd.expand(*Ma_cross_vd.shape)
        qf_b = vc_cross_vb + Ma_cross_vd
        qf_c = Ma_cross_vb

        mu_neg = -1 * torch.ones_like(Ma_cross_vd)
        mu_pos = -1 * torch.ones_like(Ma_cross_vd)
        mu_linear = -1 * torch.ones_like(Ma_cross_vd)

        nzie = (qf_a.abs() > 1e-10).expand(*Ma_cross_vd.shape)

        disc = (qf_b[nzie]**2 - 4 * qf_a[nzie] * qf_c[nzie]) ** 0.5
        mu_pos[nzie] = (-qf_b[nzie] + disc) / (2 * qf_a[nzie])
        mu_neg[nzie] = (-qf_b[nzie] - disc) / (2 * qf_a[nzie])
        mu_linear[~nzie] = qf_c[~nzie] / qf_b[~nzie]

        mu_pos_valid = torch.logical_and(mu_pos >= 0, mu_pos <= 1)
        mu_neg_valid = torch.logical_and(mu_neg >= 0, mu_neg <= 1)
        mu_linear_valid = torch.logical_and(mu_linear >= 0, mu_linear <= 1)

        mu = -1 * torch.ones_like(Ma_cross_vd)
        mu[mu_pos_valid] = mu_pos[mu_pos_valid]
        mu[mu_neg_valid] = mu_neg[mu_neg_valid]
        mu[mu_linear_valid] = mu_linear[mu_linear_valid]

        lmbda = -1 * (Ma[:, :, 1] + mu * vc[:, 1:2]) / (vb[:, 1:2] + vd[:, 1:2] * mu)  # noqa: E501

        unwarped_pts = torch.stack((lmbda, mu), dim=0)

        good_indices = torch.logical_and(
            torch.logical_and(-eps <= unwarped_pts[0],
                              unwarped_pts[0] <= 1+eps),
            torch.logical_and(-eps <= unwarped_pts[1],
                              unwarped_pts[1] <= 1+eps),
        )  # ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range)
        nonzero_good_indices = good_indices.nonzero()
        inverse_j = pts[0, nonzero_good_indices[:, 1]] + ref[nonzero_good_indices[:, 0], 0]  # noqa: E501
        inverse_i = pts[1, nonzero_good_indices[:, 1]] + ref[nonzero_good_indices[:, 0], 1]  # noqa: E501
        # TODO: is replacing this with reshape operations on good_indices faster? # noqa: E501
        j = nonzero_good_indices[:, 0] % (grid_W - 1)
        i = nonzero_good_indices[:, 0] // (grid_W - 1)
        grid_mappings = torch.stack(
            (j + unwarped_pts[1, good_indices], i + unwarped_pts[0, good_indices]),  # noqa: E501
            dim=1
        )
        in_bounds = torch.logical_and(
            torch.logical_and(0 <= inverse_i, inverse_i < H),
            torch.logical_and(0 <= inverse_j, inverse_j < W),
        )
        inverse_grid[b, inverse_i[in_bounds], inverse_j[in_bounds], :] = grid_mappings[in_bounds, :]  # noqa: E501

    inverse_grid[..., 0] = (inverse_grid[..., 0]) / (grid_W - 1) * 2.0 - 1.0  # noqa: E501
    inverse_grid[..., 1] = (inverse_grid[..., 1]) / (grid_H - 1) * 2.0 - 1.0  # noqa: E501
    return inverse_grid

def calculate_iou(box1, box2):
    x1_inter = torch.max(box1[:, 0], box2[:, 0])
    y1_inter = torch.max(box1[:, 1], box2[:, 1])
    x2_inter = torch.min(box1[:, 2], box2[:, 2])
    y2_inter = torch.min(box1[:, 3], box2[:, 3])

    intersection_area = torch.clamp(x2_inter - x1_inter + 1, min=0) * torch.clamp(y2_inter - y1_inter + 1, min=0)

    area_box1 = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    area_box2 = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

    union_area = area_box1 + area_box2 - intersection_area

    iou = intersection_area / union_area
    return torch.sum(iou), box1.shape[0]

def compute_IOU(transformed_gt_bboxes, unwarp_gt_bboxes, test_cfg):
    h = transformed_gt_bboxes[:, 2] - transformed_gt_bboxes[:, 0]
    w = transformed_gt_bboxes[:, 3] - transformed_gt_bboxes[:, 1]
    area_before = w * h

    small_indice = area_before < 1024
    medium_indice = (area_before > 1024) & (area_before < 9216)
    large_indice = area_before > 9216

    iou_small_sum, small_sum = calculate_iou(transformed_gt_bboxes[small_indice], unwarp_gt_bboxes[small_indice])
    iou_medium_sum, medium_sum = calculate_iou(transformed_gt_bboxes[medium_indice], unwarp_gt_bboxes[medium_indice])
    iou_large_sum, large_sum = calculate_iou(transformed_gt_bboxes[large_indice], unwarp_gt_bboxes[large_indice])
    iou_sum, _ = calculate_iou(transformed_gt_bboxes, unwarp_gt_bboxes)

    test_cfg['iou_sum']['small_sum'] += iou_small_sum.cpu().item()
    test_cfg['iou_sum']['medium_sum'] += iou_medium_sum.cpu().item()
    test_cfg['iou_sum']['large_sum'] += iou_large_sum.cpu().item()
    test_cfg['iou_sum']['sum'] += iou_sum.cpu().item()

    test_cfg['iou_box_num_sum']['image_sum']+=1
    test_cfg['iou_box_num_sum']['small_sum'] += small_sum
    test_cfg['iou_box_num_sum']['medium_sum'] += medium_sum
    test_cfg['iou_box_num_sum']['large_sum'] += large_sum
    test_cfg['iou_box_num_sum']['sum'] += transformed_gt_bboxes.shape[0]

def compute_zoom_in_multiple(batch_gt_bboxes, interpolators, ori_shapes, test_cfg):
    warp_batch_gt_bboxes = warp_bboxes_test(batch_gt_bboxes, interpolators, ori_shapes)

    h = (batch_gt_bboxes[:, 2] - batch_gt_bboxes[:, 0]) / 2.
    w = (batch_gt_bboxes[:, 3] - batch_gt_bboxes[:, 1]) / 2.
    area_before = w * h

    small_indice = area_before < 1024
    medium_indice = (area_before > 1024) & (area_before < 9216)
    large_indice = area_before > 9216

    area_before_small = area_before[small_indice]
    area_before_medium = area_before[medium_indice]
    area_before_large = area_before[large_indice]

    h1 = warp_batch_gt_bboxes[0][:, 2] - warp_batch_gt_bboxes[0][:, 0]
    w1 = warp_batch_gt_bboxes[0][:, 3] - warp_batch_gt_bboxes[0][:, 1]
    area_after = w1 * h1

    area_after_small = area_after[small_indice]
    area_after_medium = area_after[medium_indice]
    area_after_large = area_after[large_indice]
    max_mutliple = float(torch.max(area_after / area_before).cpu().item())

    test_cfg['multiple_sum']['img_sum'] += 1
    test_cfg['multiple_sum']['sum'] += float(torch.sum(area_after / area_before).cpu().item())
    test_cfg['multiple_sum']['small_sum'] += float(torch.sum(area_after_small / area_before_small).cpu().item())
    test_cfg['multiple_sum']['medium_sum'] += float(torch.sum(area_after_medium / area_before_medium).cpu().item())
    test_cfg['multiple_sum']['large_sum'] += float(torch.sum(area_after_large / area_before_large).cpu().item())
    if max_mutliple > test_cfg['multiple_sum']['max_multiple']:
        test_cfg['multiple_sum']['max_multiple'] = max_mutliple

    test_cfg['box_num_sum']['sum'] += batch_gt_bboxes[0].shape[0]
    test_cfg['box_num_sum']['small_sum'] += area_before_small.shape[0]
    test_cfg['box_num_sum']['medium_sum'] += area_before_medium.shape[0]
    test_cfg['box_num_sum']['large_sum'] += area_before_large.shape[0]
    test_cfg['box_num_sum']['multiple_more_than_2'] += torch.sum((area_after / area_before) > 2).item()

    if test_cfg['multiple_sum']['img_sum'] == 1547:
        print("global multiple is:", test_cfg['multiple_sum']['sum'] / test_cfg['box_num_sum']['sum'])
        print("small multiple is:", test_cfg['multiple_sum']['small_sum'] / test_cfg['box_num_sum']['small_sum'])
        print("medium multiple is:", test_cfg['multiple_sum']['medium_sum'] / test_cfg['box_num_sum']['medium_sum'])
        print("large multiple is:", test_cfg['multiple_sum']['large_sum'] / test_cfg['box_num_sum']['large_sum'])
        print("max multiple is:", test_cfg['multiple_sum']['max_multiple'])
        print("more than 2 multiple sum:", test_cfg['box_num_sum']['multiple_more_than_2'])


@MODELS.register_module()
class SaliencyYOLODetector(SingleStageDetector):
    r"""Implementation of YOLO Series

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLO. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLO. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
        use_syncbn (bool): whether to use SyncBatchNorm. Defaults to True.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_syncbn: bool = True):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # self.grid_generator = GridGenerator()

        # TODO： Waiting for mmengine support
        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log('Using SyncBatchNorm()', 'current')

    # def to(self, device):
    #     super().to(device)
    #     self.grid_generator.to(device)
    #
    #     return self# Assuming MyClass has a 'to' method that appropriately moves its data to the device

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        device = batch_inputs[0].device

        ori_shape = (800, 800)
        rz_shape = (800, 800)
        batch_gt_bboxes = copy.deepcopy(batch_data_samples['bboxes_labels'])

        # lr_inputs = F.interpolate(batch_inputs, size=rz_shape,
        #                           mode='bilinear', align_corners=True)
        #
        # grid, uniform_grid, interpolators, saliency_list = self.grid_generator(lr_inputs, rz_shape)

        # start_time = time_synchronized()

        # end_time = time_synchronized(start_time, 'grid_sample')

        # inverse_grid = invert_nonseparable_grid(grid, (batch_inputs.shape))

        # ################### 可视化 ###########################
        # img_visualize = (batch_inputs[0] * 255.).byte().contiguous()
        # image = img_visualize.permute(1, 2, 0).cpu().numpy()
        # # 保存绘制好的图像到指定文件夹
        # output_path = '/root/autodl-tmp/img_ori.jpg'
        # cv2.imwrite(output_path, image)
        # # img = cv2.imread('/root/autodl-tmp/img_ori.jpg')
        # # for i in range(0, inverse_grid.shape[1], 4):
        # #     for j in range(0, inverse_grid.shape[2], 4):
        # #         x = (inverse_grid[0, i, j, 0] + 1) / 2 * batch_inputs.shape[3]
        # #         y = (inverse_grid[0, i, j, 1] + 1) / 2 * batch_inputs.shape[2]
        # #         cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)  # 绘制红色的点，半径为1
        # # # 保存处理后的图像
        # # # img = cv2.resize(img, tuple([batch_inputs.shape[3], batch_inputs.shape[2]]))
        # # cv2.imwrite("/root/autodl-tmp/sample_location.jpg", img)
        # #

        # #
        # image = cv2.imread('/root/autodl-tmp/img_resampled.jpg')
        # # 假设box_list是包含检测框的列表，每个框都是一个元组 (x1, y1, x2, y2)
        # box_list = batch_transformed_gt_bboxes.cpu().numpy()
        # # 遍历每个检测框并在图像上绘制
        # for box in box_list:
        #     x1, y1, x2, y2 = box
        #     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度
        # # 保存绘制好的图像到指定文件夹
        # output_path = '/root/autodl-tmp/img_resampled_with_gt.jpg'
        # cv2.imwrite(output_path, image)

        # image = cv2.imread('/root/autodl-tmp/img_resampled.jpg')
        # # 假设box_list是包含检测框的列表，每个框都是一个元组 (x1, y1, x2, y2)
        # box_list = batch_transformed_gt_bboxes_3.detach().cpu().numpy()
        # # 遍历每个检测框并在图像上绘制
        # for box in box_list:
        #     x1, y1, x2, y2 = box
        #     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度
        # # 保存绘制好的图像到指定文件夹
        # output_path = '/root/autodl-tmp/img_resampled_with_gt_2.jpg'
        # cv2.imwrite(output_path, image)
        #
        # image = cv2.imread('/root/autodl-tmp/img_resampled.jpg')
        # # 假设box_list是包含检测框的列表，每个框都是一个元组 (x1, y1, x2, y2)
        # box_list = batch_transformed_gt_bboxes_5.detach().cpu().numpy()
        # # 遍历每个检测框并在图像上绘制
        # for box in box_list:
        #     x1, y1, x2, y2 = box
        #     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度
        # # 保存绘制好的图像到指定文件夹
        # output_path = '/root/autodl-tmp/img_resampled_with_gt_3.jpg'
        # cv2.imwrite(output_path, image)
        # #################### 可视化 ###########################

        x, grid, uniform_grid, interpolators = self.extract_feat(batch_inputs)

        # batch_sampled_inputs = F.grid_sample(batch_inputs, grid, padding_mode='reflection')
        # img_visualize = (batch_sampled_inputs[0] * 255.).byte().contiguous()
        # image = img_visualize.permute(1, 2, 0).cpu().numpy()
        # # 保存绘制好的图像到指定文件夹
        # output_path = '/root/autodl-tmp/img_resampled.jpg'
        # cv2.imwrite(output_path, image)

        batch_transformed_gt_bboxes = warp_bboxes(batch_gt_bboxes, interpolators, ori_shape)

        batch_data_samples['bboxes_labels'][:, 2:] = batch_transformed_gt_bboxes  # (img_id, x1, y1, x2, y2)

        losses = self.bbox_head.loss(x, batch_data_samples)
        #
        # x_2 = self.extract_feat(batch_sampled_inputs.detach())
        # losses_detector = self.bbox_head.loss(x_2, batch_data_samples)

        mag_weight = self.train_cfg['saliency_loss_weight']['weight']
        M = self.train_cfg['saliency_loss_weight']['M']
        b = self.train_cfg['saliency_loss_weight']['b']
        a = self.train_cfg['saliency_loss_weight']['a']
        saliency_losses = saliency_loss(grid, uniform_grid, batch_gt_bboxes,
                                        ori_shape, mag_weight, M, b, a)

        losses.update(saliency_losses)
        # losses = {
        #     'loss_cls': losses_saliency['loss_cls'],
        #     'loss_bbox': losses_detector['loss_bbox'],
        #     'loss_dfl': losses_detector['loss_dfl'],
        #     'loss_saliency': saliency_losses['saliency_loss']
        # }

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        rz_shape = (800, 800)
        batch_data_samples[0].real_shape = rz_shape
        # batch_gt_bboxes = copy.deepcopy(batch_data_samples[0]._gt_instances.bboxes)

        # lr_inputs = F.interpolate(batch_inputs, size=rz_shape,
        #                           mode='bilinear', align_corners=True)
        # end_time_1 = time_synchronized(start_time_1, 'interpolate time')

        # torch.cuda.synchronize()
        # start_time = time.perf_counter()
        # grid = self.grid_generator(lr_inputs, rz_shape)
        # torch.cuda.synchronize()
        # elapsed = time.perf_counter() - start_time
        # print("grid_generator time:", elapsed)

        # # compute zoom-in multiple
        # rz_gt_bboxes = batch_data_samples[0]._gt_instances.bboxes
        # compute_zoom_in_multiple(rz_gt_bboxes, batch_gt_bboxes, interpolators, (batch_inputs.shape[2:]), self.test_cfg)

        # compute_zoom_in_multiple(batch_gt_bboxes, interpolators, ori_shape, self.test_cfg)
        # start_time = time_synchronized()
        # batch_sampled_inputs = F.grid_sample(batch_inputs, grid, padding_mode='reflection')
        # # end_time = time_synchronized(start_time, 'grid sample time')

        # torch.cuda.synchronize()
        # start_time = time_synchronized()
        x, grid, interpolators, interpolators_1600, grid_large_1600 = self.extract_feat(batch_inputs)
        # end_time = time_synchronized(start_time, 'extract_feat')
        # batch_sampled_inputs = F.grid_sample(batch_inputs, grid, padding_mode='reflection')

        batch_gt_bboxes = batch_data_samples[0].gt_instances.bboxes
        batch_gt_bboxes = torch.cat([torch.zeros(batch_gt_bboxes.shape[0], 2).to(batch_gt_bboxes.device), batch_gt_bboxes], dim=1)
        batch_input_shape = (800, 800)
        if len(batch_gt_bboxes)>0:
            batch_transformed_gt_bboxes = warp_bboxes(batch_gt_bboxes.clone(), interpolators, batch_input_shape)
            unwarp_transformed_bboxes = unwarp_bboxes(batch_transformed_gt_bboxes, (800, 800), grid[0], batch_input_shape)

            batch_transformed_gt_bboxes_1 = warp_bboxes(batch_gt_bboxes.clone(), interpolators_1600, batch_input_shape)
            unwarp_transformed_bboxes_1 = unwarp_bboxes(batch_transformed_gt_bboxes_1, (4800, 4800), grid_large_1600[0],batch_input_shape)

            # unwarp_transformed_bboxes = batch_transformed_gt_bboxes[0]
            # compute_IOU(batch_gt_bboxes[0], unwarp_transformed_bboxes, self.test_cfg)
            # if (unwarp_transformed_bboxes - batch_transformed_gt_bboxes[0]).sum() != 0:
            #     print('xx')
            compute_IOU(batch_gt_bboxes[:, 2:], unwarp_transformed_bboxes_1, self.test_cfg)
        else:
            self.test_cfg['iou_box_num_sum']['image_sum'] += 1


        # start_time_1 = time_synchronized()
        results_list = self.bbox_head.predict(
            x, batch_data_samples, grid=grid, rescale=True)
        # end_time_1 = time_synchronized(start_time_1, 'predict')
        # results_list_2 = self.bbox_head.predict(
        #     x, batch_data_samples, grid=grid, rescale=False)
        # torch.cuda.synchronize()
        # elapsed = time.perf_counter() - start_time
        # print("yolo time:", elapsed)

        # #################### 可视化 ###########################
        # img_visualize = (batch_inputs[0] * 255.).byte().contiguous()
        # image = img_visualize.permute(1, 2, 0).cpu().numpy()
        # # image = cv2.resize(image, (rz_shape[1], rz_shape[0]))
        # # 保存绘制好的图像到指定文件夹
        # output_path = '/root/autodl-tmp/img_ori.jpg'
        # cv2.imwrite(output_path, image)

        # ##################### 绘制采样点 ##########################
        # img = cv2.imread('/root/autodl-tmp/img_ori.jpg')
        # for i in range(0, grid.shape[1], 2):
        #     for j in range(0, grid.shape[2], 2):
        #         x = (grid[0, i, j, 0] + 1) / 2 * 800
        #         y = (grid[0, i, j, 1] + 1) / 2 * 800
        #         cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)  # 绘制红色的点，半径为1
        # # 保存处理后的图像
        # # img = cv2.resize(img, tuple([batch_inputs.shape[3], batch_inputs.shape[2]]))
        # cv2.imwrite("/root/autodl-tmp/sample_location.jpg", img)


        # image = cv2.imread('/root/autodl-tmp/img_ori.jpg')
        # # 假设box_list是包含检测框的列表，每个框都是一个元组 (x1, y1, x2, y2)
        # box_list = batch_data_samples[0]._gt_instances.bboxes.cpu().numpy()
        # # 遍历每个检测框并在图像上绘制
        # for box in box_list:
        #     x1, y1, x2, y2 = box
        #     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度
        # # 保存绘制好的图像到指定文件夹
        # output_path = '/root/autodl-tmp/img_ori_with_gt.jpg'
        # cv2.imwrite(output_path, image)

        # img_visualize = (batch_sampled_inputs[0] * 255.).byte().contiguous()
        # image = img_visualize.permute(1, 2, 0).cpu().numpy()
        # # 保存绘制好的图像到指定文件夹
        # output_path = '/root/autodl-tmp/img_resampled.jpg'
        # cv2.imwrite(output_path, image)
        #
        # image = cv2.imread(batch_data_samples[0].img_path)
        # # 假设box_list是包含检测框的列表，每个框都是一个元组 (x1, y1, x2, y2)
        # box_list = results_list[0]['bboxes'].cpu().numpy()
        # # 遍历每个检测框并在图像上绘制
        # for box in box_list:
        #     x1, y1, x2, y2 = box
        #     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度
        # # 保存绘制好的图像到指定文件夹
        # output_path = '/root/autodl-tmp/img_ori_predict.jpg'
        # cv2.imwrite(output_path, image)
        #
        #
        # image = cv2.imread('/root/autodl-tmp/img_resampled.jpg')
        # # 假设box_list是包含检测框的列表，每个框都是一个元组 (x1, y1, x2, y2)
        # box_list = results_list_2[0]['bboxes'].cpu().numpy()
        # # 遍历每个检测框并在图像上绘制
        # for box in box_list:
        #     x1, y1, x2, y2 = box
        #     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度
        # # 保存绘制好的图像到指定文件夹
        # output_path = '/root/autodl-tmp/img_resampled_predict.jpg'
        # cv2.imwrite(output_path, image)

        # results_list[0]['bboxes'] = unwarp_bboxes(results_list[0]['bboxes'], rz_shape,
        #                                           grid[0], ori_shape)
        # pad_param = batch_data_samples[0].pad_param
        # results_list[0]['bboxes'] -= torch.tensor([pad_param[2], pad_param[0],
        #                                            pad_param[2], pad_param[0]]).cuda()
        # results_list[0]['bboxes'][:, 0:4:2] /= batch_data_samples[0].scale_factor[1]
        # results_list[0]['bboxes'][:, 1:4:2] /= batch_data_samples[0].scale_factor[0]

        # img_id = batch_data_samples[0].img_id
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)

        return batch_data_samples


@MODELS.register_module()
class YOLODetector(SingleStageDetector):
    r"""Implementation of YOLO Series

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLO. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLO. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
        use_syncbn (bool): whether to use SyncBatchNorm. Defaults to True.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_syncbn: bool = True):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # TODO： Waiting for mmengine support
        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log('Using SyncBatchNorm()', 'current')


