# From the great work https://github.com/Junyi42/GeoAware-SC

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


class Correlation2Displacement(nn.Module):
    def __init__(self,
                 setting=1,
                 window_size=0,
                 feature_size=16,
                 calc_soft_argmax=True,
                 beta=0.02):
        super(Correlation2Displacement, self).__init__()

        self.setting = setting
        self.window_size = window_size
        self.final_corr_size = feature_size
        self.x_normal = np.linspace(-1, 1, self.final_corr_size)
        self.x_normal = nn.Parameter(torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False))
        self.y_normal = np.linspace(-1, 1, self.final_corr_size)
        self.y_normal = nn.Parameter(torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False))

        # 1-d indices for generating Gaussian kernels
        self.x = np.linspace(0, self.final_corr_size - 1, self.final_corr_size)
        self.x = torch.tensor(self.x, dtype=torch.float, requires_grad=False)
        self.y = np.linspace(0, self.final_corr_size - 1, self.final_corr_size)
        self.y = torch.tensor(self.y, dtype=torch.float, requires_grad=False)
        # count model parameters
        num_params = 0
        for p in self.parameters():
            num_params += p.numel()
        self.calc_soft_argmax = calc_soft_argmax
        self.beta = beta

    def softmax_with_temperature(self, x, d=1):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M  # subtract maximum value for stability
        exp_x = torch.exp(x / self.beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    def soft_argmax(self, corr):
        r'''Modified for input shape (B, Q, H, W)'''

        B, Q, H, W = corr.shape

        # Apply softmax over spatial dimension H*W for each query
        corr = corr.view(B, Q, -1)  # [B, Q, H*W]
        corr = self.softmax_with_temperature(corr, d=2)  # softmax over spatial
        corr = corr.view(B, Q, H, W)  # [B, Q, H, W]

        # Get normalized coordinate grids
        y_normal = self.y_normal.to(corr.dtype).view(1, 1, H, 1).expand(B, Q, H, W)  # [B, Q, H, W]
        x_normal = self.x_normal.to(corr.dtype).view(1, 1, 1, W).expand(B, Q, H, W)  # [B, Q, H, W]

        # Weighted sum of coordinates
        grid_x = (corr * x_normal).sum(dim=(2, 3), keepdim=True)  # [B, Q, 1, 1]
        grid_y = (corr * y_normal).sum(dim=(2, 3), keepdim=True)  # [B, Q, 1, 1]

        return grid_x, grid_y  # still normalized in [-1, 1]


    def unnormalise_and_convert_mapping_to_flow(self, map):
        # here map is normalised to -1;1
        # we put it back to 0,W-1, then convert it to flow
        B, C, H, W = map.size()
        mapping = torch.zeros_like(map)
        # mesh grid
        mapping[:, 0, :, :] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0  # unormalise
        mapping[:, 1, :, :] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0  # unormalise

        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if mapping.is_cuda:
            grid = grid.cuda()

        flow = mapping - grid
        return flow

    def mutual_nn_filter(self, correlation_matrix):
        r"""Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18)"""
        corr_src_max = torch.max(correlation_matrix, dim=3, keepdim=True)[0]
        corr_trg_max = torch.max(correlation_matrix, dim=2, keepdim=True)[0]
        corr_src_max[corr_src_max == 0] += 1e-30
        corr_trg_max[corr_trg_max == 0] += 1e-30

        corr_src = correlation_matrix / corr_src_max
        corr_trg = correlation_matrix / corr_trg_max

        return correlation_matrix * (corr_src * corr_trg)

    def apply_gaussian_kernel(self, corr, sigma=5):
        b, hw, h, w = corr.size()

        idx = corr.max(dim=1)[1]  # b x h x w    get maximum value along channel
        idx_y = (idx // w).view(b, 1, 1, h, w).float()
        idx_x = (idx % w).view(b, 1, 1, h, w).float()

        x = self.x.view(1, 1, w, 1, 1).expand(b, 1, w, h, w).to(corr.device)
        y = self.y.view(1, h, 1, 1, 1).expand(b, h, 1, h, w).to(corr.device)

        gauss_kernel = torch.exp(-((x - idx_x) ** 2 + (y - idx_y) ** 2) / (2 * sigma ** 2))
        gauss_kernel = gauss_kernel.view(b, hw, h, w)

        return gauss_kernel * corr

    def forward(self, x):
        # calc correleation map
        # x = self.process_corr_map(x)
        if self.window_size > 0:  # zero out the corr_map outside the window
            B, H, W, _ = x.shape
            assert B == 1
            corr = x.view(H * W, H * W)
            # get the argmax
            max_index_flatten = torch.argmax(corr, dim=-1)
            max_index_x = max_index_flatten % (H * W)  # (H_s * W_s, )
            max_index_y = max_index_flatten // (H * W)  # (H_s * W_s, )
            corr = corr.view(-1, H * W, H * W)
            window_mask = torch.zeros_like(corr)  # (H_s * W_s, H_t, W_t)
            for i in range(-self.window_size, self.window_size + 1):
                for j in range(-self.window_size, self.window_size + 1):
                    # make sure the index is within the range
                    clamped_y = torch.clamp(max_index_y + i, 0, H * W - 1)  # (H_s * W_s, )
                    clamped_x = torch.clamp(max_index_x + j, 0, H * W - 1)  # (H_s * W_s, )
                    window_mask[torch.arange(corr.shape[0]), clamped_y, clamped_x] = 1
            corr = corr * window_mask
            x = corr.view(1, H, W, -1)

        if self.window_size < 0:  # gaussian kernel
            x = self.apply_gaussian_kernel(x.permute(0, 3, 1, 2), sigma=-self.window_size)
            x = x.permute(0, 2, 3, 1)

        # take soft argmax
        if self.calc_soft_argmax:
            grid_x, grid_y = self.soft_argmax(x)
            x = torch.cat((grid_x.squeeze(-1), grid_y.squeeze(-1)), dim=2)  # (B, Q, 2)
        else:
            x = self.argmax(x)  # (B, Q, 2)
        return x # (B, Q, 2)

    def argmax(self, x):

        B, Q, H, W = x.shape
        # Flatten H and W dimensions
        flattened = x.view(B, Q, -1)

        # Get the argmax index along the flattened dimension
        argmax_indices = torch.argmax(flattened, dim=-1)  # [B, Q]

        # Convert the flattened index to 2D coordinates (row, col)
        argmax_y = argmax_indices // W  # Row index
        argmax_x = argmax_indices % W  # Column index

        # Combine into a single tensor of shape [B, Q, 2]
        argmax_positions = torch.stack((argmax_x, argmax_y), dim=-1)

        # Convert to float and normalize to [-1, 1]
        argmax_positions = argmax_positions.float() / (W - 1) * 2 - 1

        return argmax_positions  # [B, Q, 2]

    def process_corr_map(self, x):
        """
        input: corr_map (B, (C), H, W, H2, W2)
        output: processed corr_map (B, H, W, H2*W2)
        """
        if len(x.shape) == 5:
            x = x.unsqueeze(1)
        B, C, H, W, H2, W2 = x.shape
        x = x.view(B, H, W, -1)
        return x


if __name__ == '__main__':
    # usage
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    model = Correlation2Displacement(60 * 60, 8)
    model.to('cuda')
    correlation_map = torch.randn((1, 60, 60, 60, 60))
    correlation_map = correlation_map.to('cuda')
    displacement_map = model(correlation_map)
    print(displacement_map.shape)  # Expected: torch.Size([1, 60, 60, 2])