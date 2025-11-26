import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple

# Simple MLP definition for use_ffn=True
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EpipolarCrossAttention(nn.Module):
    """
    Cross-View Epipolar Attention with memory-efficient two-pass algorithm.

    Two-pass algorithm:
    - Pass 1: Build logits by sampling along epipolar lines in chunks
    - Pass 2: Apply softmax and weighted aggregation of values

    Memory management:
    - fp32 for logits buffer (numerical stability)
    - fp16/bf16 for Q/K/V and sampled tensors
    - Reuse buffers: logits -> weights after softmax
    - Chunking along sample axis to control memory usage
    - Optional tiling for very large spatial dimensions
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 attn_heads: int = 8,
                 num_samples: int = 32,
                 n_chunk: int = 8,
                 tile_size: Optional[Tuple[int, int]] = None,
                 use_ffn: bool = False,
                 kv_dim_ratio: float = 0.5,
                 dropout: float = 0.0):
        """
        Args:
            input_channels: Input feature dimension
            output_channels: Output feature dimension
            attn_heads: Number of attention heads
            num_samples: Number of samples along epipolar line (N)
            n_chunk: Chunk size for processing samples (controls memory usage)
            tile_size: Optional (h, w) tile size for processing large images
            use_ffn: Whether to use feed-forward network
            kv_dim_ratio: Ratio of key/value dimension to input_channels
            dropout: Dropout probability
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.attn_heads = attn_heads
        self.num_samples = num_samples
        self.n_chunk = n_chunk
        self.tile_size = tile_size
        self.use_ffn = use_ffn

        # Attention dimensions
        self.d_k = input_channels // attn_heads
        self.d_v = int(input_channels * kv_dim_ratio) // attn_heads

        # Projection layers
        self.q_proj = nn.Linear(input_channels, attn_heads * self.d_k)
        self.k_proj = nn.Linear(input_channels, attn_heads * self.d_k)
        self.v_proj = nn.Linear(input_channels, attn_heads * self.d_v)
        self.out_proj = nn.Linear(attn_heads * self.d_v, output_channels)

        # Optional FFN
        if use_ffn:
            mlp_ratio = 4.0
            self.ffn = Mlp(
                in_features=output_channels,
                hidden_features=int(output_channels * mlp_ratio),
                act_layer=nn.GELU,
                drop=dropout,
                bias=True,
            )
        else:
            self.ffn = nn.Identity()

        # Initialize zero_conv for residual-like behavior
        self.zero_conv = nn.Conv1d(output_channels, output_channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def compute_epipolar_grids_fundamental(self, ref_int, ref_ext, src_int, src_ext, N, H, W, device, depth_min=0.1, depth_max=10.0):
        """
        Optimized, batch-friendly computation of epipolar sampling grids using the fundamental matrix.
        Args:
            ref_int, src_int: (N, 3, 3) camera intrinsics for reference and source views
            ref_ext, src_ext: (N, 4, 4) camera extrinsics (world-to-camera) for reference and source views
            N: number of (ref, src) pairs
            H, W: image height and width
            device: torch device
        Returns:
            grid: (N, num_samples, H, W, 2) sampling grids for torch.grid_sample
        """
        def skew_batch(t):
            O = t.new_zeros(t.shape[0])
            tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
            return torch.stack([
                torch.stack([O, -tz,  ty], dim=1),
                torch.stack([tz,  O, -tx], dim=1),
                torch.stack([-ty, tx,  O], dim=1)
            ], dim=1)  # (N, 3, 3)

        # Ensure extrinsics are (N, 4, 4) homogeneous matrices
        def reshape_spatial(x):
            if x.dim() == 5:
                x = x.reshape(-1, *x.shape[-2:])  # (B*N, ...)
            return x

        # Always flatten to (B*N, ...) for batch ops
        ref_ext = reshape_spatial(ref_ext)
        src_ext = reshape_spatial(src_ext)
        ref_int = reshape_spatial(ref_int)
        src_int = reshape_spatial(src_int)
        # Compute relative pose and fundamental matrix in batch
        R1 = ref_ext[:, :3, :3]
        t1 = ref_ext[:, :3, 3]
        R2 = src_ext[:, :3, :3]
        t2 = src_ext[:, :3, 3]
        R = torch.bmm(R2, R1.transpose(1, 2))
        t = t2 - torch.bmm(R, t1.unsqueeze(2)).squeeze(2)
        E = torch.bmm(skew_batch(t), R)
        K1_inv = torch.inverse(ref_int)
        K2_inv = torch.inverse(src_int)
        F = torch.bmm(torch.bmm(K2_inv.transpose(1, 2), E), K1_inv)

        # Meshgrid for all pixels (H*W, 3)
        y_ref, x_ref = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
        ones = torch.ones_like(x_ref)
        pix_ref_homo = torch.stack([x_ref, y_ref, ones], dim=-1).reshape(-1, 3).t()  # (3, H*W)
        pix_ref_homo = pix_ref_homo.unsqueeze(0).expand(N, 3, H*W)

        # Epipolar lines in batch: (N, 3, H*W)
        l2 = torch.bmm(F, pix_ref_homo)
        a, b, c = l2[:, 0, :], l2[:, 1, :], l2[:, 2, :]
        eps = 1e-8

        # Intersections with borders (vectorized)
        y0 = -c / (b + eps)
        yW = -(a * (W-1) + c) / (b + eps)
        x0 = -c / (a + eps)
        xH = -(b * (H-1) + c) / (a + eps)

        # Validity masks
        valid_y0 = (y0 >= 0) & (y0 < H)
        valid_yW = (yW >= 0) & (yW < H)
        valid_x0 = (x0 >= 0) & (x0 < W)
        valid_xH = (xH >= 0) & (xH < W)

        # Stack all intersection points and masks
        pts = torch.stack([
            torch.stack([torch.zeros_like(y0), y0], dim=-1),
            torch.stack([(W-1)*torch.ones_like(yW), yW], dim=-1),
            torch.stack([x0, torch.zeros_like(x0)], dim=-1),
            torch.stack([xH, (H-1)*torch.ones_like(xH)], dim=-1)
        ], dim=2)  # (N, H*W, 4, 2)
        valids = torch.stack([valid_y0, valid_yW, valid_x0, valid_xH], dim=2)  # (N, H*W, 4)

        # Find the first two valid intersections for each line
        # Use topk on boolean mask to get indices of valid points
        valid_int = valids.int()
        valid_sum = valid_int.sum(dim=2)
        # Fallback: center if <2 valid
        center = torch.tensor([(W-1)/2, (H-1)/2], device=device, dtype=pts.dtype)
        center = center.view(1, 1, 2).expand(N, H*W, 2)

        # Get indices of valid points
        idxs = torch.arange(4, device=device).view(1, 1, 4)
        idxs = idxs.expand(N, H*W, 4)
        # Mask invalids to high value so they sort last
        masked_idxs = torch.where(valids, idxs, torch.full_like(idxs, 99))
        sorted_idxs = torch.argsort(masked_idxs, dim=2)
        # Gather first two valid indices
        first_idx = sorted_idxs[:, :, 0]
        second_idx = sorted_idxs[:, :, 1]
        # Gather points
        p1 = torch.gather(pts, 2, first_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)).squeeze(2)
        p2 = torch.gather(pts, 2, second_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)).squeeze(2)
        # If not enough valid, use center
        not_enough = valid_sum < 2
        # Always get (N, H*W, 2) after where
        p1 = torch.where(not_enough.unsqueeze(-1), center, p1)
        p2 = torch.where(not_enough.unsqueeze(-1), center, p2)

        # Uniformly sample along the segment
        samples = torch.linspace(0, 1, self.num_samples, device=device, dtype=pts.dtype).view(1, 1, self.num_samples, 1)
        p1 = p1.unsqueeze(2)  # (N, H*W, 1, 2)
        p2 = p2.unsqueeze(2)  # (N, H*W, 1, 2)
        pts_sampled = p1 + (p2 - p1) * samples  # (N, H*W, num_samples, 2)
        pts_sampled = pts_sampled.permute(0, 2, 1, 3)  # (N, num_samples, H*W, 2)

        # Normalize to [-1,1] for grid_sample
        norm_x = 2 * (pts_sampled[..., 0] / (W-1)) - 1
        norm_y = 2 * (pts_sampled[..., 1] / (H-1)) - 1
        grid = torch.stack([norm_x, norm_y], dim=-1)
        grid = grid.view(N, self.num_samples, H, W, 2)
        return grid

    def forward(self, x: torch.Tensor, x_shape: Tuple[int, int, int, int],
                intrinsics: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
        """
        Two-pass epipolar cross attention.

        Args:
            x: (B*S, H*W, C) input features
            x_shape: (B, S, H, W) original shape
            intrinsics: (B, S, 3, 3) camera intrinsics
            extrinsics: (B, S, 4, 4) camera extrinsics

        Returns:
            out: (B*S, H*W, output_channels) attended features
        """
        B, S, H, W = x_shape
        device = x.device

        # Reshape input to (B, S, C, H, W)
        x_reshaped = rearrange(x, '(b s) (h w) c -> b s c h w', b=B, s=S, h=H, w=W)

        # Fold heads into batch dimension: B' = B * Hh
        B_prime = B * self.attn_heads

        # Project to Q, K, V with folded heads
        # First rearrange to (B, S, H, W, C) for linear projection
        x_for_proj = rearrange(x_reshaped, 'b s c h w -> b s h w c')
        
        # Q: (B, S, H, W, C) -> (B, S, H, W, Hh*d_k) -> (B, S, Hh, d_k, H, W)
        Q = self.q_proj(x_for_proj)  # (B, S, H, W, Hh*d_k)
        Q = rearrange(Q, 'b s h w (hh dk) -> b s hh dk h w', hh=self.attn_heads, dk=self.d_k)
        Q = rearrange(Q, 'b s hh dk h w -> (b hh) s dk h w', hh=self.attn_heads)

        # K, V: similar but with d_v for V
        K = self.k_proj(x_for_proj)  # (B, S, H, W, Hh*d_k)
        K = rearrange(K, 'b s h w (hh dk) -> b s hh dk h w', hh=self.attn_heads, dk=self.d_k)
        K = rearrange(K, 'b s hh dk h w -> (b hh) s dk h w', hh=self.attn_heads)

        V = self.v_proj(x_for_proj)  # (B, S, H, W, Hh*d_v)
        V = rearrange(V, 'b s h w (hh dv) -> b s hh dv h w', hh=self.attn_heads, dv=self.d_v)
        V = rearrange(V, 'b s hh dv h w -> (b hh) s dv h w', hh=self.attn_heads)

        # Handle tiling if specified
        if self.tile_size is not None:
            tile_h, tile_w = self.tile_size
            # For simplicity, implement single tile version first
            # TODO: Add multi-tile support
            pass

        # Initialize output accumulator
        out_accumulator = torch.zeros((B, S, self.attn_heads * self.d_v, H, W), device=device, dtype=V.dtype)

        # Process each reference view
        for ref_idx in range(S):
            # Get Q for this reference view: (B', d_k, H, W)
            Q_ref = Q[:, ref_idx]  # (B', d_k, H, W)

            # Flatten spatial dimensions for easier processing
            Q_ref_flat = rearrange(Q_ref, 'bp dk h w -> bp dk (h w)')
            hw = H * W

            # Logits buffer: (B, Hh, (S-1)*N, H, W) - fp32 for stability
            num_targets = S - 1
            logits_buffer = torch.zeros((B, self.attn_heads, num_targets * self.num_samples, H, W),
                                      device=device, dtype=torch.float32)

            # Pass 1: Build logits
            target_idx = 0
            for src_idx in range(S):
                if src_idx == ref_idx:
                    continue

                # Get K for this source view: (B', d_k, H, W)
                K_src = K[:, src_idx]  # (B', d_k, H, W)
                K_src_flat = rearrange(K_src, 'bp dk h w -> bp dk (h w)')

                # Compute epipolar grids for this (ref, src) pair
                ref_int = intrinsics[:, ref_idx]  # (B, 3, 3)
                ref_ext = extrinsics[:, ref_idx]  # (B, 4, 4)
                src_int = intrinsics[:, src_idx]  # (B, 3, 3)
                src_ext = extrinsics[:, src_idx]  # (B, 4, 4)

                # Compute grids: (B, num_samples, H, W, 2)
                grids = self.compute_epipolar_grids_fundamental(
                    ref_int, ref_ext, src_int, src_ext, B, H, W, device)

                # Process in chunks along sample dimension
                for chunk_start in range(0, self.num_samples, self.n_chunk):
                    chunk_end = min(chunk_start + self.n_chunk, self.num_samples)
                    chunk_size = chunk_end - chunk_start

                    # Sample K along epipolar lines: (B, d_k, H, W, n_chunk)
                    grids_chunk = grids[:, chunk_start:chunk_end]  # (B, n_chunk, H, W, 2)
                    
                    # Sample from K_src: (B', d_k, H, W) -> sampled (B', d_k, n_chunk, H, W)
                    K_src_expanded = K_src.unsqueeze(2).expand(-1, -1, chunk_size, -1, -1)  # (B', d_k, n_chunk, H, W)
                    
                    # Reshape for grid_sample: flatten batch and chunk dims
                    K_src_flat = rearrange(K_src_expanded, 'b d_k nc h w -> (b nc) d_k h w')
                    grids_flat = rearrange(grids_chunk, 'b nc h w c -> (b nc) h w c')
                    
                    # Expand grids to match K_src batch size
                    grids_expanded = grids_flat.repeat(B_prime // B, 1, 1, 1)
                    
                    sampled_K_flat = F.grid_sample(
                        K_src_flat, grids_expanded,
                        align_corners=True, mode='bilinear', padding_mode='zeros'
                    )  # ((B' * nc), d_k, h, w)
                    
                    # Reshape back and flatten spatial dims
                    sampled_K = rearrange(sampled_K_flat, '(b nc) d_k h w -> b d_k nc (h w)', b=B_prime, nc=chunk_size)

                    # Reshape for dot product: (B', d_k, n_chunk, H*W)
                    sampled_K = sampled_K.view(B_prime, self.d_k, chunk_size, hw)

                    # Q_ref: (B', d_k, H*W) -> (B', d_k, 1, H*W)
                    Q_expanded = Q_ref_flat.unsqueeze(2)  # (B', d_k, 1, H*W)

                    # Dot product: (B', 1, n_chunk, H*W) -> (B', n_chunk, H*W)
                    # Compute attention logits: Q @ K^T
                    # Q_expanded: (B', d_q, n_chunk, H*W)
                    # sampled_K: (B', d_k, n_chunk, H*W)
                    
                    # Reshape for matmul: (B', n_chunk, H*W, d_q) @ (B', n_chunk, d_k, H*W)
                    Q_reshaped = rearrange(Q_expanded, 'b dq nc hw -> b nc hw dq')
                    K_reshaped = rearrange(sampled_K, 'b dk nc hw -> b nc hw dk')
                    
                    chunk_logits = torch.matmul(Q_reshaped, K_reshaped.transpose(-2, -1))  # (B', n_chunk, H*W, H*W)
                    
                    # Sum over query positions to get per-sample weights: (B', n_chunk, H*W)
                    chunk_weights = chunk_logits.sum(dim=2)  # (B', n_chunk, H*W)
                    
                    # Reshape back to spatial: (B', n_chunk, H, W)
                    chunk_weights = rearrange(chunk_weights, 'bp nc (h w) -> bp nc h w', h=H, w=W)

                    # Split back to (B, Hh, n_chunk, H, W) and store in logits buffer
                    chunk_weights = rearrange(chunk_weights, '(b hh) nc h w -> b hh nc h w', b=B, hh=self.attn_heads)

                    # Store in logits buffer at correct slice
                    global_chunk_start = chunk_start + target_idx * self.num_samples
                    logits_buffer[:, :, global_chunk_start:global_chunk_start+chunk_size] = chunk_weights

                target_idx += 1

            # Pass 2: Weighted aggregation
            # Apply softmax over sample dimension (in-place to save memory)
            weights_buffer = logits_buffer  # Reuse buffer
            weights_buffer = F.softmax(weights_buffer, dim=2)  # Over (S-1)*N dimension

            # Accumulator for this reference view: (B', d_v, H, W)
            R_ref = torch.zeros((B_prime, self.d_v, H, W), device=device, dtype=V.dtype)

            # Process each target view for aggregation
            target_idx = 0
            for src_idx in range(S):
                if src_idx == ref_idx:
                    continue

                # Get V for this source view: (B', d_v, H, W)
                V_src = V[:, src_idx]  # (B', d_v, H, W)

                # Compute same epipolar grids
                ref_int = intrinsics[:, ref_idx]
                ref_ext = extrinsics[:, ref_idx]
                src_int = intrinsics[:, src_idx]
                src_ext = extrinsics[:, src_idx]

                grids = self.compute_epipolar_grids_fundamental(
                    ref_int, ref_ext, src_int, src_ext, B, H, W, device)

                # Process in chunks
                for chunk_start in range(0, self.num_samples, self.n_chunk):
                    chunk_end = min(chunk_start + self.n_chunk, self.num_samples)
                    chunk_size = chunk_end - chunk_start

                    # Sample V along epipolar lines
                    grids_chunk = grids[:, chunk_start:chunk_end]  # (B, n_chunk, H, W, 2)
                    
                    V_src_expanded = V_src.unsqueeze(2).expand(-1, -1, chunk_size, -1, -1)  # (B', d_v, n_chunk, H, W)
                    
                    # Reshape for grid_sample: flatten batch and chunk dims
                    V_src_flat = rearrange(V_src_expanded, 'b dv nc h w -> (b nc) dv h w')
                    grids_flat = rearrange(grids_chunk, 'b nc h w c -> (b nc) h w c')
                    
                    # Expand grids to match V_src batch size
                    grids_expanded = grids_flat.repeat(B_prime // B, 1, 1, 1)
                    
                    sampled_V_flat = F.grid_sample(
                        V_src_flat, grids_expanded,
                        align_corners=True, mode='bilinear', padding_mode='zeros'
                    )  # ((B' * nc), d_v, h, w)
                    
                    # Reshape back and flatten spatial dims
                    sampled_V = rearrange(sampled_V_flat, '(b nc) dv h w -> b dv nc (h w)', b=B_prime, nc=chunk_size)

                    # Get corresponding weights: (B, Hh, n_chunk, H, W)
                    global_chunk_start = chunk_start + target_idx * self.num_samples
                    weights_chunk = weights_buffer[:, :, global_chunk_start:global_chunk_start+chunk_size]

                    # Reshape weights: (B, Hh, n_chunk, H, W) -> (B', n_chunk, H*W)
                    weights_chunk = rearrange(weights_chunk, 'b hh nc h w -> (b hh) nc (h w)', hh=self.attn_heads)

                    # Apply weights to sampled values: (B', d_v, n_chunk, H*W)
                    weighted_V = sampled_V * weights_chunk.unsqueeze(1)

                    # Sum over chunk dimension: (B', d_v, H*W)
                    weighted_sum = weighted_V.sum(dim=2)

                    # Reshape back to spatial and accumulate: (B', d_v, H, W)
                    weighted_sum = rearrange(weighted_sum, 'bp dv (h w) -> bp dv h w', h=H, w=W)
                    R_ref += weighted_sum

                target_idx += 1

            # Reshape R_ref: (B', d_v, H, W) -> (B, Hh*d_v, H, W)
            R_ref = rearrange(R_ref, '(b hh) dv h w -> b (hh dv) h w', b=B, hh=self.attn_heads)
            
            # Add to output accumulator at correct view index
            out_accumulator[:, ref_idx] = R_ref

        # Apply output projection: (B, S, attn_heads*d_v, H, W) -> (B, S, output_channels, H, W)
        # Flatten spatial dims for linear layer: (B, S, attn_heads*d_v, H, W) -> (B*S, H*W, attn_heads*d_v)
        out_accumulator = rearrange(out_accumulator, 'b s c h w -> (b s) (h w) c')
        out_features = self.out_proj(out_accumulator)  # (B*S, H*W, output_channels)
        out_features = rearrange(out_features, '(b s) (h w) c -> b s c h w', b=B, s=S, h=H, w=W)

        # Apply FFN if specified
        if self.use_ffn:
            # Flatten spatial for FFN
            out_flat = rearrange(out_features, 'b s c h w -> (b s) c (h w)')
            out_flat = self.ffn(out_flat)
            out_features = rearrange(out_flat, '(b s) c (h w) -> b s c h w', b=B, s=S, h=H, w=W)

        # Final residual connection
        out_features = out_features + x_reshaped

        # Apply zero_conv for final projection
        out_flat = rearrange(out_features, 'b s c h w -> (b s) c (h w)')
        out_final = self.zero_conv(out_flat)
        out_final = rearrange(out_final, '(b s) c (h w) -> (b s) (h w) c', b=B, s=S, h=H, w=W)

        return out_final


if __name__ == "__main__":
    # Test EpipolarCrossAttention with random values
    B, S, H, W = 2, 3, 32, 32  # Smaller batch for testing
    input_channels = 256
    output_channels = 256
    num_samples = 16  # Smaller for testing
    attn_heads = 8
    n_chunk = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create random input tensors
    x = torch.randn(B * S, H * W, input_channels, device=device)
    x_shape = (B, S, H, W)
    intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(B, S, 1, 1)
    extrinsics = torch.eye(4, device=device).unsqueeze(0).repeat(B, S, 1, 1)

    # Instantiate the module
    epipolar_attn = EpipolarCrossAttention(
        input_channels=input_channels,
        output_channels=output_channels,
        attn_heads=attn_heads,
        num_samples=num_samples,
        n_chunk=n_chunk,
        use_ffn=False,
    ).to(device)

    # Run forward pass
    print("Running epipolar attention...")
    with torch.no_grad():
        out_feats = epipolar_attn(x, x_shape, intrinsics, extrinsics)
    print("Input shape:", x.shape)
    print("Output shape:", out_feats.shape)  # Should be (B*S, H*W, output_channels)
    print("Test completed successfully!")
