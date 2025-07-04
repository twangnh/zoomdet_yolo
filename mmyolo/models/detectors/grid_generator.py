import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .saliency_network import saliency_network_resnet18, fov_simple, resnetUnet, Unet, saliency_network_yolov8cspdarknet
import random
from torch_cluster import nearest, knn
from torchvision import transforms
import cv2
import time
from math import floor, ceil

def time_synchronized(t1=None, m=None):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t2 = time.time()
    if m is not None and t1 is not None:
        print('timing {} is {}'.format(m, t2 - t1))
    return t2

def make1DGaussian(size, fwhm=3, center=None):
    """ Make a 1D gaussian kernel.

    size is the length of the kernel,
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, size, 1, dtype=np.float)

    if center is None:
        center = size // 2

    return np.exp(-4*np.log(2) * (x-center)**2 / fwhm**2)

def make2DGaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

class NearestInterpolator_Torch(nn.Module):

    def __init__(self, points, values):
        super(NearestInterpolator_Torch, self).__init__()
        self.points = points
        self.values = values

    def __call__(self, points_q):
        nn_torch = nearest(points_q, self.points)
        return self.values[nn_torch]

    # def __call__(self, points_q, n_nearest = 2):
        # knn_torch = knn(self.points, points_q, n_nearest)
        # sample_inds = knn_torch[1, :].view(-1, n_nearest)
        # return self.values[sample_inds].mean(-1)

# class GridGenerator(nn.Module):
#     def __init__(self):
#         super(GridGenerator, self).__init__()
#         self.using_gt = False
#         self.saliency_network = saliency_network_resnet18()
#         self.uniform_grid_y, self.uniform_grid_x = torch.meshgrid(torch.arange(100, dtype=torch.float) + 0.5,
#                                                         torch.arange(100, dtype=torch.float) + 0.5)
#
#         self.uniform_grid = torch.cat((2 * self.uniform_grid_x.unsqueeze(0) / 100 - 1,
#                                   2 * self.uniform_grid_y.unsqueeze(0) / 100 - 1)).unsqueeze(0)
#
#     def to(self, device):
#         super(GridGenerator, self).to(device)
#         self.saliency_network.to(device)
#         self.uniform_grid_y = self.uniform_grid_y.to(device)
#         self.uniform_grid_x = self.uniform_grid_x.to(device)
#         self.uniform_grid = self.uniform_grid.to(device)
#
#         return self
#
#
#     def deformable_grid_generate(self, saliency, rz_shape, device):
#         if self.training:
#             final_grid = torch.zeros(saliency.shape[0], 2, saliency.shape[2], saliency.shape[3],  device=device)
#             # saliency = F.interpolate(saliency, size=rz_shape, mode='bilinear', align_corners=True)
#             # final_grid[:, 0, :, :] = (uniform_grid_x + saliency[:, 0, :, :] * offset_confidence) / (saliency.shape[3] - 1)
#             # final_grid[:, 1, :, :] = (uniform_grid_y + saliency[:, 1, :, :] * offset_confidence) / (saliency.shape[2] - 1)
#             # final_grid[:, 0, :, :] = (uniform_grid_x + saliency[:, 0, :, :]) / (saliency.shape[3] - 1)
#             # final_grid[:, 1, :, :] = (uniform_grid_y + saliency[:, 1, :, :]) / (saliency.shape[2] - 1)
#             final_grid[:, 0, :, :] = 2 * (self.uniform_grid_x + saliency[:, 0, :, :]) / saliency.shape[3] - 1
#             final_grid[:, 1, :, :] = 2 * (self.uniform_grid_y + saliency[:, 1, :, :]) / saliency.shape[2] - 1
#             final_grid = F.interpolate(final_grid, size=rz_shape, mode='bilinear', align_corners=True)
#
#             return final_grid.permute(0, 2, 3, 1), self.uniform_grid.permute(0, 2, 3, 1)
#
#         else:
#             final_grid = torch.zeros(saliency.shape[0], 2, saliency.shape[2], saliency.shape[3], device=device)
#             # saliency = F.interpolate(saliency, size=rz_shape, mode='bilinear', align_corners=True)
#             final_grid[:, 0, :, :] = 2 * (self.uniform_grid_x + saliency[:, 0, :, :]) / saliency.shape[3] - 1
#
#             final_grid[:, 1, :, :] = 2 * (self.uniform_grid_y + saliency[:, 1, :, :]) / saliency.shape[2] - 1
#             final_grid = F.interpolate(final_grid, size=rz_shape, mode='bilinear', align_corners=True)
#             return final_grid.permute(0, 2, 3, 1)
#
#
#     def forward(self, x, rz_shape):
#         # torch.cuda.synchronize()
#         # start_time = time.perf_counter()
#         saliency = self.saliency_network(x)
#         # saliency = torch.zeros((1, 2, 100, 100)).to('cuda')
#         # torch.cuda.synchronize()
#         # elapsed = time.perf_counter() - start_time
#         # print("saliency_net time:", elapsed)
#
#         device = x.device
#
#         if self.training:
#             grid, uniform_grid = self.deformable_grid_generate(saliency, rz_shape, device)
#
#             # start_time_2 = time_synchronized()
#             # (N, output_shape, output_shape, 2) to (2, N, output_shape, output_shape)
#             grid_reorder = grid.permute(3, 0, 1, 2).detach()
#             # (N, 800*800)
#
#             u_cor = ((grid_reorder[0, :, :, :] + 1) / 2).view(
#                 grid_reorder.shape[1], -1)
#             v_cor = ((grid_reorder[1, :, :, :] + 1) / 2).view(
#                 grid_reorder.shape[1], -1)
#
#             ##改到0 H-1范围， 原来是0.5 H-0.5范围
#             u_cor, v_cor = u_cor-1/grid_reorder.shape[3]/2., v_cor-1/grid_reorder.shape[2]/2.
#
#             # (800*800)
#             x_cor = torch.arange(0, grid_reorder.shape[3], device=grid_reorder.device).unsqueeze(0).expand(
#                 (grid_reorder.shape[2], grid_reorder.shape[3])).reshape(-1)
#             # (N, 800*800)
#             x_cor = x_cor.unsqueeze(0).expand(u_cor.shape[0], -1).float()
#
#             y_cor = torch.arange(0, grid_reorder.shape[2], device=grid_reorder.device).unsqueeze(-1).expand(
#                 (grid_reorder.shape[2], grid_reorder.shape[3])).reshape(-1)
#             y_cor = y_cor.unsqueeze(0).expand(u_cor.shape[0], -1).float()
#
#             # x_cor, y_cor = x_cor+0.5, y_cor+0.5
#             # v:h, u:w
#             # (N, 1, 800*800)
#             u_cor, v_cor = u_cor.unsqueeze(1), v_cor.unsqueeze(1)
#             # every line in u is (1, u_cor, v_cor)
#             # every line in v is (0, u_cor, v_cor)
#             u = torch.cat([torch.zeros(u_cor.shape, device=u_cor.device, dtype=u_cor.dtype), u_cor, v_cor], dim=1)
#             v = torch.cat([torch.ones(u_cor.shape, device=u_cor.device, dtype=u_cor.dtype), u_cor, v_cor], dim=1)
#
#             # (N, 3, 2*800*800) -> (N, 2*800*800, 3)
#             points = torch.cat([u, v], dim=2).transpose(2, 1).float()
#
#             # range from (0,1)
#             # y_cor /= grid.shape[1]
#             # x_cor /= grid.shape[2]
#
#             # (N, 2*800*800)
#             values = torch.cat([x_cor, y_cor], dim=1)
#
#             # range from (0,1)
#             # points[:, :, 1:] /= torch.Tensor([rz_shape[1], rz_shape[0]]).to(points.device)
#
#             b, n, dim = points.shape
#             # (N, 2*800*800)
#             if b == 1:
#                 batch_inds = torch.zeros((1, n)).to(points.device)
#             else:
#                 batch_inds = torch.arange(b).unsqueeze(1).expand(-1, n).to(points.device)
#             # (2*800*800*N, 4), every column means (img_id, u or v, u_cor, v_cor)
#             batch_points = torch.cat([batch_inds.view(-1).unsqueeze(1), points.contiguous().view(-1, 3)], dim=1)
#
#             interpolator_all = NearestInterpolator_Torch(batch_points, values.view(-1))
#             # end_time = time_synchronized(start_time_2, 'generate_interpolator')
#
#             return grid, uniform_grid, interpolator_all, saliency
#
#         else:
#             # torch.cuda.synchronize()
#             # start_time = time.perf_counter()
#             grid = self.deformable_grid_generate(saliency, rz_shape, device)
#             # torch.cuda.synchronize()
#             # elapsed = time.perf_counter() - start_time
#             # print("saliency_to_grid time:", elapsed)
#
#             return grid
