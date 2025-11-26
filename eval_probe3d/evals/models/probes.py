import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import torch.nn.functional as F
import xformers.ops as xops


class SurfaceNormalHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        head_type="multiscale",
        uncertainty_aware=False,
        hidden_dim=512,
        kernel_size=1,
    ):
        super().__init__()

        self.uncertainty_aware = uncertainty_aware
        output_dim = 4 if uncertainty_aware else 3

        self.kernel_size = kernel_size

        assert head_type in ["linear", "multiscale", "dpt", "mv-dpt", "mv-dptv2"]
        name = f"snorm_{head_type}_k{kernel_size}"
        self.name = f"{name}_UA" if uncertainty_aware else name

        if head_type == "linear":
            self.head = Linear(feat_dim, output_dim, kernel_size)
        elif head_type == "multiscale":
            self.head = MultiscaleHead(feat_dim, output_dim, hidden_dim, kernel_size)
        elif head_type == "dpt":
            self.head = DPT(feat_dim, output_dim, hidden_dim, kernel_size)
        elif head_type == "mv-dpt":
            self.head = MultiViewDPT(feat_dim, output_dim, hidden_dim, kernel_size)
        elif head_type == "mv-dptv2":
            self.head = MultiViewDPTV2(feat_dim, output_dim, hidden_dim, kernel_size)
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

    def forward(self, feats):
        return self.head(feats)


class DepthHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        head_type="multiscale",
        min_depth=0.001,
        max_depth=10,
        prediction_type="bindepth",
        hidden_dim=512,
        kernel_size=1,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.name = f"{prediction_type}_{head_type}_k{kernel_size}"

        if prediction_type == "bindepth":
            output_dim = 256
            self.predict = DepthBinPrediction(min_depth, max_depth, n_bins=output_dim)
        elif prediction_type == "sigdepth":
            output_dim = 1
            self.predict = DepthSigmoidPrediction(min_depth, max_depth)
        else:
            raise ValueError()

        if head_type == "linear":
            self.head = Linear(feat_dim, output_dim, kernel_size)
        elif head_type == "multiscale":
            self.head = MultiscaleHead(feat_dim, output_dim, hidden_dim, kernel_size)
        elif head_type == "dpt":
            self.head = DPT(feat_dim, output_dim, hidden_dim, kernel_size)
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

    def forward(self, feats):
        """Prediction each pixel."""
        feats = self.head(feats)
        depth = self.predict(feats)
        return depth


class DepthBinPrediction(nn.Module):
    def __init__(
        self,
        min_depth=0.001,
        max_depth=10,
        n_bins=256,
        bins_strategy="UD",
        norm_strategy="linear",
    ):
        super().__init__()
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.norm_strategy = norm_strategy
        self.bins_strategy = bins_strategy

    def forward(self, prob):
        if self.bins_strategy == "UD":
            bins = torch.linspace(
                self.min_depth, self.max_depth, self.n_bins, device=prob.device
            )
        elif self.bins_strategy == "SID":
            bins = torch.logspace(
                self.min_depth, self.max_depth, self.n_bins, device=prob.device
            )

        # following Adabins, default linear
        if self.norm_strategy == "linear":
            prob = torch.relu(prob)
            eps = 0.1
            prob = prob + eps
            prob = prob / prob.sum(dim=1, keepdim=True)
        elif self.norm_strategy == "softmax":
            prob = torch.softmax(prob, dim=1)
        elif self.norm_strategy == "sigmoid":
            prob = torch.sigmoid(prob)
            prob = prob / prob.sum(dim=1, keepdim=True)

        depth = torch.einsum("ikhw,k->ihw", [prob, bins])
        depth = depth.unsqueeze(dim=1)
        return depth


class DepthSigmoidPrediction(nn.Module):
    def __init__(self, min_depth=0.001, max_depth=10):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, pred):
        depth = pred.sigmoid()
        depth = self.min_depth + depth * (self.max_depth - self.min_depth)
        return depth


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, kernel_size, with_skip=True):
        super().__init__()
        self.with_skip = with_skip
        if self.with_skip:
            self.resConfUnit1 = ResidualConvUnit(features, kernel_size)

        self.resConfUnit2 = ResidualConvUnit(features, kernel_size)

    def forward(self, x, skip_x=None):
        if skip_x is not None:
            assert self.with_skip and skip_x.shape == x.shape
            x = self.resConfUnit1(x) + skip_x

        x = self.resConfUnit2(x)
        return x


class ResidualConvUnit(nn.Module):
    def __init__(self, features, kernel_size):
        super().__init__()
        assert kernel_size % 1 == 0, "Kernel size needs to be odd"
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x) + x


class DPT(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_dim=512, kernel_size=3):
        super().__init__()
        assert len(input_dims) == 4
        self.conv_0 = nn.Conv2d(input_dims[0], hidden_dim, 1, padding=0)
        self.conv_1 = nn.Conv2d(input_dims[1], hidden_dim, 1, padding=0)
        self.conv_2 = nn.Conv2d(input_dims[2], hidden_dim, 1, padding=0)
        self.conv_3 = nn.Conv2d(input_dims[3], hidden_dim, 1, padding=0)

        self.ref_0 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_1 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_2 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_3 = FeatureFusionBlock(hidden_dim, kernel_size, with_skip=False)

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
        )

    def forward(self, feats):
        """Prediction each pixel."""
        assert len(feats) == 4

        feats[0] = self.conv_0(feats[0])
        feats[1] = self.conv_1(feats[1])
        feats[2] = self.conv_2(feats[2])
        feats[3] = self.conv_3(feats[3])

        feats = [interpolate(x, scale_factor=2) for x in feats]

        out = self.ref_3(feats[3], None)
        out = self.ref_2(feats[2], out)
        out = self.ref_1(feats[1], out)
        out = self.ref_0(feats[0], out)

        out = interpolate(out, scale_factor=4)
        out = self.out_conv(out)
        out = interpolate(out, scale_factor=2)
        return out


def make_conv(input_dim, hidden_dim, output_dim, num_layers, kernel_size=1):
    if num_layers == 1:
        conv = nn.Conv2d(input_dim, output_dim, kernel_size)
    else:
        assert num_layers > 1
        modules = [nn.Conv2d(input_dim, hidden_dim, kernel_size), nn.ReLU(inplace=True)]
        for i in range(num_layers - 2):
            modules.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size))
            modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Conv2d(hidden_dim, output_dim, kernel_size))
        conv = nn.Sequential(*modules)

    return conv


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=1):
        super().__init__()
        if type(input_dim) is not int:
            input_dim = input_dim[-1]

        assert type(input_dim) is int
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, padding=padding)

    def forward(self, feats):
        if type(feats) is list:
            feats = feats[-1]
            # feats = torch.cat(feats, dim=1)
        # print(feats.shape)
        feats = interpolate(feats, scale_factor=4, mode="bilinear")
        return self.conv(feats)


class MultiscaleHead(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_dim=512, kernel_size=1):
        super().__init__()

        self.convs = nn.ModuleList(
            [make_conv(in_d, None, hidden_dim, 1, kernel_size) for in_d in input_dims]
        )
        interm_dim = len(input_dims) * hidden_dim
        self.conv_mid = make_conv(interm_dim, hidden_dim, hidden_dim, 3, kernel_size)
        self.conv_out = make_conv(hidden_dim, hidden_dim, output_dim, 2, kernel_size)

    def forward(self, feats):
        num_feats = len(feats)
        feats = [self.convs[i](feats[i]) for i in range(num_feats)]

        h, w = feats[-1].shape[-2:]
        feats = [interpolate(feat, (h, w), mode="bilinear") for feat in feats]
        feats = torch.cat(feats, dim=1).relu()

        # upsample
        feats = interpolate(feats, scale_factor=2, mode="bilinear")
        feats = self.conv_mid(feats).relu()
        feats = interpolate(feats, scale_factor=4, mode="bilinear")
        return self.conv_out(feats)


class MultiViewDPT(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_dim=512, kernel_size=3):
        super().__init__()
        assert len(input_dims) == 4

        # print(input_dims, hidden_dim, kernel_size)

        self.cross_attn_0 = CrossViewAttention(input_dims[0], num_heads=8)
        self.cross_attn_1 = CrossViewAttention(input_dims[1], num_heads=8)
        self.cross_attn_2 = CrossViewAttention(input_dims[2], num_heads=8)
        self.cross_attn_3 = CrossViewAttention(input_dims[3], num_heads=8)

        # Shared projection layers (process all views together)
        self.conv_0 = nn.Conv2d(input_dims[0]*2, hidden_dim, 1, padding=0)
        self.conv_1 = nn.Conv2d(input_dims[1]*2, hidden_dim, 1, padding=0)
        self.conv_2 = nn.Conv2d(input_dims[2]*2, hidden_dim, 1, padding=0)
        self.conv_3 = nn.Conv2d(input_dims[3]*2, hidden_dim, 1, padding=0)

        # Cross-view attention at each scale
        # self.cross_attn_0 = CrossViewAttention(hidden_dim, num_heads=8)
        # self.cross_attn_1 = CrossViewAttention(hidden_dim, num_heads=8)
        # self.cross_attn_2 = CrossViewAttention(hidden_dim, num_heads=4)
        # self.cross_attn_3 = CrossViewAttention(hidden_dim, num_heads=4)

        # Refinement blocks (shared across views)
        self.ref_0 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_1 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_2 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_3 = FeatureFusionBlock(hidden_dim, kernel_size, with_skip=False)

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
        )

    def forward(self, feats):
        """
        Args:
            feats: List of 4 tensors, each [B, N, C, H, W]
        Returns:
            output: [B, N, output_dim, H_final, W_final]
        """
        assert len(feats) == 4
        B, N = feats[0].shape[:2]

        feats[0] = torch.cat([feats[0], self.cross_attn_0(feats[0])], dim=2)
        feats[1] = torch.cat([feats[1], self.cross_attn_1(feats[1])], dim=2)
        feats[2] = torch.cat([feats[2], self.cross_attn_2(feats[2])], dim=2)
        feats[3] = torch.cat([feats[3], self.cross_attn_3(feats[3])], dim=2)

        # feats[0] = self.cross_attn_0(feats[0])
        # feats[1] = self.cross_attn_1(feats[1])
        # feats[2] = self.cross_attn_2(feats[2])
        # feats[3] = self.cross_attn_3(feats[3])



        # Project features: reshape to [B*N, C, H, W], apply conv, reshape back
        feats[0] = self._apply_conv(self.conv_0, feats[0])  # [B, N, hidden_dim, H, W]
        feats[1] = self._apply_conv(self.conv_1, feats[1])
        feats[2] = self._apply_conv(self.conv_2, feats[2])
        feats[3] = self._apply_conv(self.conv_3, feats[3])

        # Upsample 2x
        feats = [self._interpolate(x, scale_factor=2) for x in feats]

        # Apply cross-view attention at each scale
        # feats[0] = self.cross_attn_0(feats[0])
        # feats[1] = self.cross_attn_1(feats[1])
        # feats[2] = self.cross_attn_2(feats[2])
        # feats[3] = self.cross_attn_3(feats[3])

        # Refinement (process each view independently)
        out = self._apply_refinement(feats)

        out = self._interpolate(out, scale_factor=4)
        out = self._apply_conv(self.out_conv, out)
        out = self._interpolate(out, scale_factor=2)

        return out  # [B, N, output_dim, H, W]

    def _apply_conv(self, conv_module, x):
        """Apply conv to [B, N, C, H, W] by reshaping to [B*N, C, H, W]"""
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = conv_module(x)
        _, C_out, H_out, W_out = x.shape
        x = x.view(B, N, C_out, H_out, W_out)
        return x

    def _interpolate(self, x, scale_factor):
        """Interpolate [B, N, C, H, W]"""
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = interpolate(x, scale_factor=scale_factor)
        _, _, H_new, W_new = x.shape
        x = x.view(B, N, C, H_new, W_new)
        return x

    def _apply_refinement(self, feats):
        """Apply refinement blocks to each view"""
        B, N = feats[0].shape[:2]

        # Reshape all features to [B*N, C, H, W] for refinement
        feats = [f.view(B * N, *f.shape[2:]) for f in feats]

        out = self.ref_3(feats[3], None)
        out = self.ref_2(feats[2], out)
        out = self.ref_1(feats[1], out)
        out = self.ref_0(feats[0], out)

        # Reshape back to [B, N, C, H, W]
        _, C, H, W = out.shape
        out = out.view(B, N, C, H, W)
        return out




class CrossViewAttention(nn.Module):
    """Memory-efficient cross-view attention using xformers"""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: [B, N, C, H, W]
        Returns:
            out: [B, N, C, H, W] with cross-view information
        """
        B, N, C, H, W = x.shape

        # Reshape: [B, N, C, H, W] -> [B, N*H*W, C]
        x = x.permute(0, 1, 3, 4, 2).reshape(B, N * H * W, C)

        # Self-attention across all N*H*W tokens
        residual = x
        x = self.norm(x)

        # QKV projection
        qkv = self.qkv(x).reshape(B, N * H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)  # [3, B, N*H*W, num_heads, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Memory-efficient attention with xformers
        x = xops.memory_efficient_attention(q, k, v, attn_bias=None)
        # Output: [B, N*H*W, num_heads, head_dim]

        x = x.reshape(B, N * H * W, C)
        x = self.proj(x) + residual

        # Reshape back: [B, N*H*W, C] -> [B, N, C, H, W]
        x = x.reshape(B, N, H, W, C).permute(0, 1, 4, 2, 3)
        return x


class CrossViewAttentionWithPosEmb(nn.Module):
    """Memory-efficient cross-view attention with learnable view positional embeddings"""

    def __init__(self, dim, num_heads=8, max_views=16):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
        # Learnable positional embeddings for each view
        self.view_pos_emb = nn.Parameter(torch.randn(1, max_views, dim) * 0.02)
        self.max_views = max_views

    def forward(self, x):
        """
        Args:
            x: [B, N, C, H, W]
        Returns:
            out: [B, N, C, H, W] with cross-view information
        """
        B, N, C, H, W = x.shape
        assert N <= self.max_views, f"Number of views {N} exceeds max_views {self.max_views}"

        # Reshape: [B, N, C, H, W] -> [B, N, H*W, C]
        x = x.permute(0, 1, 3, 4, 2).reshape(B, N, H * W, C)
        
        # Add view positional embeddings
        # view_pos_emb: [1, N, C] -> broadcast to [B, N, H*W, C]
        view_emb = self.view_pos_emb[:, :N, :].unsqueeze(2)  # [1, N, 1, C]
        x = x + view_emb  # Broadcast add
        
        # Reshape for attention: [B, N, H*W, C] -> [B, N*H*W, C]
        x = x.reshape(B, N * H * W, C)

        # Self-attention across all N*H*W tokens
        residual = x
        x = self.norm(x)

        # QKV projection
        qkv = self.qkv(x).reshape(B, N * H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)  # [3, B, N*H*W, num_heads, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Memory-efficient attention with xformers
        x = xops.memory_efficient_attention(q, k, v, attn_bias=None)
        # Output: [B, N*H*W, num_heads, head_dim]

        x = x.reshape(B, N * H * W, C)
        x = self.proj(x) + residual

        # Reshape back: [B, N*H*W, C] -> [B, N, C, H, W]
        x = x.reshape(B, N, H, W, C).permute(0, 1, 4, 2, 3)
        return x


class MultiViewDPTV2(nn.Module):
    """MultiView DPT with positional embeddings to differentiate between views"""
    
    def __init__(self, input_dims, output_dim, hidden_dim=512, kernel_size=3, max_views=16):
        super().__init__()
        assert len(input_dims) == 4

        # Cross-view attention with positional embeddings at each scale
        self.cross_attn_0 = CrossViewAttentionWithPosEmb(input_dims[0], num_heads=8, max_views=max_views)
        self.cross_attn_1 = CrossViewAttentionWithPosEmb(input_dims[1], num_heads=8, max_views=max_views)
        self.cross_attn_2 = CrossViewAttentionWithPosEmb(input_dims[2], num_heads=8, max_views=max_views)
        self.cross_attn_3 = CrossViewAttentionWithPosEmb(input_dims[3], num_heads=8, max_views=max_views)

        # Shared projection layers (process all views together)
        self.conv_0 = nn.Conv2d(input_dims[0]*2, hidden_dim, 1, padding=0)
        self.conv_1 = nn.Conv2d(input_dims[1]*2, hidden_dim, 1, padding=0)
        self.conv_2 = nn.Conv2d(input_dims[2]*2, hidden_dim, 1, padding=0)
        self.conv_3 = nn.Conv2d(input_dims[3]*2, hidden_dim, 1, padding=0)

        # Refinement blocks (shared across views)
        self.ref_0 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_1 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_2 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_3 = FeatureFusionBlock(hidden_dim, kernel_size, with_skip=False)

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
        )

    def forward(self, feats):
        """
        Args:
            feats: List of 4 tensors, each [B, N, C, H, W]
        Returns:
            output: [B, N, output_dim, H_final, W_final]
        """
        assert len(feats) == 4
        B, N = feats[0].shape[:2]

        # Apply cross-view attention with positional embeddings
        feats[0] = torch.cat([feats[0], self.cross_attn_0(feats[0])], dim=2)
        feats[1] = torch.cat([feats[1], self.cross_attn_1(feats[1])], dim=2)
        feats[2] = torch.cat([feats[2], self.cross_attn_2(feats[2])], dim=2)
        feats[3] = torch.cat([feats[3], self.cross_attn_3(feats[3])], dim=2)

        # Project features: reshape to [B*N, C, H, W], apply conv, reshape back
        feats[0] = self._apply_conv(self.conv_0, feats[0])  # [B, N, hidden_dim, H, W]
        feats[1] = self._apply_conv(self.conv_1, feats[1])
        feats[2] = self._apply_conv(self.conv_2, feats[2])
        feats[3] = self._apply_conv(self.conv_3, feats[3])

        # Upsample 2x
        feats = [self._interpolate(x, scale_factor=2) for x in feats]

        # Refinement (process each view independently)
        out = self._apply_refinement(feats)

        out = self._interpolate(out, scale_factor=4)
        out = self._apply_conv(self.out_conv, out)
        out = self._interpolate(out, scale_factor=2)

        return out  # [B, N, output_dim, H, W]

    def _apply_conv(self, conv_module, x):
        """Apply conv to [B, N, C, H, W] by reshaping to [B*N, C, H, W]"""
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = conv_module(x)
        _, C_out, H_out, W_out = x.shape
        x = x.view(B, N, C_out, H_out, W_out)
        return x

    def _interpolate(self, x, scale_factor):
        """Interpolate [B, N, C, H, W]"""
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = interpolate(x, scale_factor=scale_factor)
        _, _, H_new, W_new = x.shape
        x = x.view(B, N, C, H_new, W_new)
        return x

    def _apply_refinement(self, feats):
        """Apply refinement blocks to each view"""
        B, N = feats[0].shape[:2]

        # Reshape all features to [B*N, C, H, W] for refinement
        feats = [f.view(B * N, *f.shape[2:]) for f in feats]

        out = self.ref_3(feats[3], None)
        out = self.ref_2(feats[2], out)
        out = self.ref_1(feats[1], out)
        out = self.ref_0(feats[0], out)

        # Reshape back to [B, N, C, H, W]
        _, C, H, W = out.shape
        out = out.view(B, N, C, H, W)
        return out