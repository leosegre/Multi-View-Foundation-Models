from einops import rearrange
import torch
from torch import nn
from kornia.feature.dedode.transformer.layers import Mlp
from timm.layers import Attention

class DinoSpatialAdapter(nn.Module):
  def __init__(
    self,
    input_channels,
    output_channels,
    num_heads=8,
    parametric_spatial_conv=False,
    spatial_reduction=True,
    use_ffn=False
  ):
    super().__init__()
    self.norm = nn.LayerNorm(input_channels)
    # self.spatial_attn = nn.MultiheadAttention(input_channels, num_heads, batch_first=True)
    self.spatial_attn = Attention(
        dim=input_channels,
        num_heads=num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0
    )
    self.zero_conv = nn.Conv1d(input_channels, output_channels, kernel_size=1)
    nn.init.zeros_(self.zero_conv.weight)
    nn.init.zeros_(self.zero_conv.bias)
    self.fix_spatial_attn = True
    self.spatial_reduction = spatial_reduction
    self.fix_res_bug = True
    if spatial_reduction:
      if parametric_spatial_conv:
        self.spatial_downsample = nn.Sequential(
          nn.MaxPool2d(kernel_size=2),
          nn.Conv2d(input_channels, input_channels, 3, padding=1),
          nn.BatchNorm2d(input_channels),
          nn.ReLU(inplace=True)
        )
        self.spatial_upsample = nn.Sequential(
          nn.Conv2d(input_channels, input_channels, 3, padding=1),
          nn.BatchNorm2d(input_channels),
          nn.ReLU(inplace=True)
        )
      else:
        self.spatial_downsample = nn.MaxPool2d(kernel_size=2)
        self.spatial_upsample = nn.Identity()
    if use_ffn:
      mlp_ratio = 4.0
      self.ffn = Mlp(
            in_features=input_channels,
            hidden_features=int(input_channels * mlp_ratio),
            act_layer=nn.GELU,
            drop=0.0,
            bias=True,
        )
    else:
        self.ffn = nn.Identity()

  def forward(self, x, x_shape, registers=None):
    B, S, H, W = x_shape

    # If no registers are provided, preserve original behavior exactly.
    if registers is None:
      x = rearrange(x, '(b s) (h w) c -> (b s) c h w', s=S, b=B, h=H, w=W)
      if self.spatial_reduction:
        x_res = self.spatial_downsample(x)
      else:
        x_res = x
      h, w = x_res.shape[-2], x_res.shape[-1]
      # Registers shape is [B*S, n_registers, C]
      x_res = rearrange(x_res, '(b s) c h w -> b (h w s) c', s=S, b=B)
      x_res = self.norm(x_res)
      x_res = self.spatial_attn(x_res)
      x_res = self.ffn(x_res)
      x_res = rearrange(x_res, 'b (h w s) c -> (b s) c h w', s=S, b=B, h=h, w=w)
      if self.spatial_reduction:
        x_res = nn.functional.interpolate(x_res, size=(H, W), mode='bilinear', align_corners=True)
        x_res = self.spatial_upsample(x_res)

      if self.fix_res_bug:
        x = x_res
      else:
        x = x + x_res
      x = rearrange(x, '(b s) c h w -> (b s) c (h w)', s=S, b=B, h=H, w=W)
    else:
      # register-aware path
      if x.shape[-1] != registers.shape[-1]:
        # Pad with zeros where Plucker embeddings are missing
        padding = torch.zeros((*registers.shape[:-1], x.shape[-1] - registers.shape[-1]), device=registers.device)
        registers = torch.cat([registers, padding], dim=-1)
      x = rearrange(x, '(b s) (h w) c -> (b s) c h w', s=S, b=B, h=H, w=W)
      if self.spatial_reduction:
        x_down = self.spatial_downsample(x)
      else:
        x_down = x
      h, w = x_down.shape[-2], x_down.shape[-1]

      # Prepare tokens: (b, t, s, c) where t = h*w
      x_res = rearrange(x_down, '(b s) c h w -> b (h w) s c', s=S, b=B)

      # Registers incoming shape: (B*S, n_registers, C)
      # reshape to (B, n_registers, S, C) so we can concat along token dim
      regs = rearrange(registers, '(b s) n c -> b n s c', s=S, b=B)
      x_res = torch.cat([x_res, regs], dim=1)  # concat on t axis

      # merge token and view dims for attention: (B, t*s, C)
      x_res = rearrange(x_res, 'b t s c -> b (t s) c')
      x_res = self.norm(x_res)
      x_res = self.spatial_attn(x_res)
      x_res = self.ffn(x_res)

      # recover (B, t, s, C)
      x_res = rearrange(x_res, 'b (t s) c -> b t s c', s=S, b=B)

      # split tokens back into spatial tokens and register tokens along t
      n_regs = regs.shape[1]
      x_tokens, regs_res = torch.split(x_res, [h * w, n_regs], dim=1)
      # x_tokens -> (B*S, C, h, w)
      x_tokens = rearrange(x_tokens, 'b (h w) s c -> (b s) c h w', s=S, b=B, h=h, w=w)
      # regs_res -> (B*S, n_regs, C)
      regs_res = rearrange(regs_res, 'b n s c -> (b s) n c', s=S, b=B)

      # upsample back to original spatial resolution and add residual
      if self.spatial_reduction:
        x_tokens = nn.functional.interpolate(x_tokens, size=(H, W), mode='bilinear', align_corners=True)
        x_tokens = self.spatial_upsample(x_tokens)
      
      if self.fix_res_bug:
        x = x_tokens
        registers = regs_res
      else:
        x = x + x_tokens
        registers = registers + regs_res
      x = rearrange(x, '(b s) c h w -> (b s) c (h w)', s=S, b=B, h=H, w=W)
    # elif self.fix_spatial_attn:
    #   x = rearrange(x, '(b s) hw c -> b (s hw) c', s=S, b=B)
    #   x = self.norm(x)
    #   attn_output = self.spatial_attn(x)
    #   ffn = self.ffn(attn_output)
    #   x = x + ffn
    #   x = rearrange(x, 'b (s hw) c -> (b s) c hw', s=S, b=B)
    # else:
    #   x = rearrange(x, '(b s) hw c -> (b hw) s c', s=S, b=B)
    #   x = self.norm(x)
    #   attn_output = self.spatial_attn(x)
    #   ffn = self.ffn(attn_output)
    #   x = x + ffn
    #   x = rearrange(x, '(b hw) s c -> (b s) c hw', s=S, b=B)

    # Zero Convolution
    x = self.zero_conv(x)
    x = x.transpose(1, 2)  # (B*S, C, H*W) --> (B*S, H*W, C)
    
    if registers is not None:
      # registers currently (B*S, n_regs, C) -> conv1d expects (N, C, L)
      regs = registers.transpose(1, 2)  # (B*S, C, n_regs)
      regs = self.zero_conv(regs)
      regs = regs.transpose(1, 2)  # (B*S, n_regs, out_C)
      return x, regs
    
    return x