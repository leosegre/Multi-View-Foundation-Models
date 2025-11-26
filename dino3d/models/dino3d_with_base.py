from dino3d.models.utils.utils import get_dino3d_model
from dino3d.models.base_dino import DINO
from torch import nn
import torch
import torch.nn.functional as F
from dino3d.checkpointing import CheckPoint
from einops import rearrange
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


class DINO3DWITHBASE(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        print(model_args)
        self.dino3d = get_dino3d_model(model_args)
        base_model_output = 'dense-cls' if model_args.use_cls_token else 'dense'
        self.base_model = DINO(
                        model_name=model_args.model_name,
                        output=base_model_output,
                        return_multilayer=False,
                        stride=model_args.stride,
                        )

    def forward(self, x, pe):
        dino3d_output = self.dino3d(x, pe)
        B, N, C, H, W = x.shape
        x = rearrange(x, "b n c h w -> (b n) c h w")
        base_output = self.base_model(x)
        base_output = rearrange(base_output, "(b n) c h w -> b n c h w", b=B, n=N)
        # Combine outputs from dino3d and base model
        combined_output = torch.cat([base_output, dino3d_output], -3)

        return combined_output
    
    def forward_no_plucker(self, x, pe):
        dino3d_output = self.dino3d(x, torch.zeros_like(pe))
        B, N, C, H, W = x.shape
        x = rearrange(x, "b n c h w -> (b n) c h w")
        base_output = self.base_model(x)
        base_output = rearrange(base_output, "(b n) c h w -> b n c h w", b=B, n=N)
        # Combine outputs from dino3d and base model
        combined_output = torch.cat([base_output, dino3d_output], -3)

        return combined_output

    def load_3d_checkpoint(self, checkpoint_dir):
        print(f"Loading model from {checkpoint_dir}")
        checkpointer = CheckPoint(checkpoint_dir)
        self.dino3d = checkpointer.load_model(self.dino3d)

class DINO3DDOUBLEBASE(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        base_model_output = 'dense-cls' if model_args.use_cls_token else 'dense'
        self.base_model_output = base_model_output
        self.base_model = DINO(
                        model_name=model_args.model_name,
                        output=base_model_output,
                        return_multilayer=False,
                        stride=model_args.stride,
                        )
        # self.base_model = DINO(
        #                 model_name=model_args.model_name,
        #                 output=model_args.model_output_type,
        #                 return_multilayer=False,
        #                 stride=model_args.stride,
        #                 )

        # self.base_model = build_2d_model(model_args.backbone_type)

    def forward(self, x, pe):
        # return_prefix_tokens = False
        # return_class_token = False
        # norm = True
        B, N, C, H, W = x.shape
        x = rearrange(x, "b n c h w -> (b n) c h w")
        raw_vit_feats = self.base_model(x)
        # vit_outputs = self.base_model.get_intermediate_layers(
        #     x,
        #     n=[len(self.base_model.blocks) - 1],
        #     reshape=True,
        #     return_prefix_tokens=return_prefix_tokens,
        #     return_class_token=return_class_token,
        #     norm=norm,
        # )  # [b, 384, 37, 37]
        # vit_outputs = (
        #     vit_outputs[-1]
        #     if return_prefix_tokens or return_class_token
        #     else vit_outputs
        # )
        # raw_vit_feats = vit_outputs[0].permute(0, 2, 3, 1).detach()
        # raw_vit_feats = vit_outputs[0].detach()
        base_output = rearrange(raw_vit_feats, "(b n) c h w -> b n c h w", b=B, n=N)
        # Combine outputs from dino3d and base model
        if self.base_model_output == 'dense-cls':
            base_output_channels = base_output.shape[2]
            combined_output = torch.cat([base_output[:,:,:base_output_channels // 2], base_output], -3)
        else:
            combined_output = torch.cat([base_output, base_output], -3)

        return combined_output


class DINO3DWITHBASE_DEBUG(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.dino3d = get_dino3d_model(model_args)
        base_model_output = 'dense-cls' if model_args.use_cls_token else 'dense'
        self.base_model = DINO(
                        model_name=model_args.model_name,
                        output=base_model_output,
                        return_multilayer=False,
                        stride=model_args.stride,
                        )

    def forward(self, x, pe):
        print(x.mean(), x.std(), x.min(), x.max())
        print(pe.mean(), pe.std(), pe.min(), pe.max())
        with torch.cuda.amp.autocast(dtype=torch.float16):
            dino3d_output_plucker = self.dino3d(x, pe)
            B, N, C, H, W = x.shape
            x_2d = rearrange(x, "b n c h w -> (b n) c h w")
            base_output = self.base_model(x_2d)
            base_output = rearrange(base_output, "(b n) c h w -> b n c h w", b=B, n=N)
            # Combine outputs from dino3d and base model
            combined_output_plucker = torch.cat([base_output, dino3d_output_plucker], -3)
            
            dino3d_output = self.dino3d(x, torch.zeros_like(pe))
            # Combine outputs from dino3d and base model
            combined_output = torch.cat([base_output, dino3d_output], -3)
        for view_idx in range(N):
            diff = (combined_output[0, view_idx] - combined_output_plucker[0, view_idx]).abs()
            diff_stats = {
                'mean': float(diff.mean()),
                'var': float(diff.var()),
                'min': float(diff.min()),
                'max': float(diff.max())
            }
            print(f"View {view_idx} | Diff stats (forward - forward_no_plucker): {diff_stats}")
        # Visualize the N views of x (each one in a new row), 
        # along with the PCA of: [combined_output_plucker, combined_output, dino3d_output_plucker, dino3d_output]
        
        # Use the first batch for visualization
        batch_idx = 0
        
        # Fit PCA for combined outputs (768 channels)
        all_feat_flat_combined = []
        C_base = base_output[batch_idx, 0].shape[0]
        for i in range(N):
            # combined_output_plucker
            feats = combined_output_plucker[batch_idx, i].cpu().numpy()
            C_feat, H_feat, W_feat = feats.shape
            feat_flat = feats.reshape(C_feat, -1).T
            all_feat_flat_combined.append(feat_flat)
            # combined_output
            feats = combined_output[batch_idx, i].cpu().numpy()
            feat_flat = feats.reshape(C_feat, -1).T
            all_feat_flat_combined.append(feat_flat)
        all_feat_flat_combined = np.concatenate(all_feat_flat_combined, axis=0)
        pca_combined = PCA(n_components=3)
        pca_combined.fit(all_feat_flat_combined)
        
        # Fit PCA for single outputs (384 channels)
        all_feat_flat_single = []
        C_dino = dino3d_output_plucker[batch_idx, 0].shape[0]
        for i in range(N):
            # dino3d_output_plucker
            feats = dino3d_output_plucker[batch_idx, i].cpu().numpy()
            feat_flat = feats.reshape(C_dino, -1).T
            all_feat_flat_single.append(feat_flat)
            # dino3d_output
            feats = dino3d_output[batch_idx, i].cpu().numpy()
            feat_flat = feats.reshape(C_dino, -1).T
            all_feat_flat_single.append(feat_flat)
            # base_output
            feats = base_output[batch_idx, i].cpu().numpy()
            feat_flat = feats.reshape(C_base, -1).T
            all_feat_flat_single.append(feat_flat)
        all_feat_flat_single = np.concatenate(all_feat_flat_single, axis=0)
        pca_single = PCA(n_components=3)
        pca_single.fit(all_feat_flat_single)
        
        # Fit PCA for pe (6 channels) - REMOVED: Now visualizing channels directly
        
        fig, axes = plt.subplots(N, 8, figsize=(32, 4 * N))
        combined_pca_imgs = []
        single_pca_imgs = []
        pe_pca_imgs = []
        for i in range(N):
            # Original image
            img = x[batch_idx, i].permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'View {i} Original')
            axes[i, 0].axis('off')
            
            # PCA for combined_output_plucker
            feats = combined_output_plucker[batch_idx, i].cpu().numpy()
            feat_flat = feats.reshape(C_feat, -1).T
            pca_img_plucker = pca_combined.transform(feat_flat).reshape(H_feat, W_feat, 3)
            
            # PCA for combined_output
            feats = combined_output[batch_idx, i].cpu().numpy()
            feat_flat = feats.reshape(C_feat, -1).T
            pca_img_combined = pca_combined.transform(feat_flat).reshape(H_feat, W_feat, 3)
            
            # Global normalization for combined group
            all_pca_combined = np.concatenate([pca_img_plucker.flatten(), pca_img_combined.flatten()])
            global_min_combined = all_pca_combined.min()
            global_max_combined = all_pca_combined.max()
            pca_img_plucker = (pca_img_plucker - global_min_combined) / (global_max_combined - global_min_combined + 1e-8)
            pca_img_combined = (pca_img_combined - global_min_combined) / (global_max_combined - global_min_combined + 1e-8)
            
            axes[i, 1].imshow(pca_img_plucker)
            axes[i, 1].set_title(f'View {i} Combined Plucker')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pca_img_combined)
            axes[i, 2].set_title(f'View {i} Combined No Plucker')
            axes[i, 2].axis('off')
            
            # PCA for dino3d_output_plucker
            feats = dino3d_output_plucker[batch_idx, i].cpu().numpy()
            feat_flat = feats.reshape(C_dino, -1).T
            pca_img_dino_plucker = pca_single.transform(feat_flat).reshape(H_feat, W_feat, 3)
            
            # PCA for dino3d_output
            feats = dino3d_output[batch_idx, i].cpu().numpy()
            feat_flat = feats.reshape(C_dino, -1).T
            pca_img_dino = pca_single.transform(feat_flat).reshape(H_feat, W_feat, 3)
            
            # PCA for base_output
            feats = base_output[batch_idx, i].cpu().numpy()
            feat_flat = feats.reshape(C_base, -1).T
            pca_img_base = pca_single.transform(feat_flat).reshape(H_feat, W_feat, 3)
            
            # Plucker coordinates - first 3 channels
            pe_view = pe[batch_idx, i].cpu().numpy()
            pe_first3 = pe_view[:3]  # First 3 channels
            pe_first3_norm = (pe_first3 - pe_first3.min()) / (pe_first3.max() - pe_first3.min() + 1e-8)
            pe_first3_rgb = np.stack([pe_first3_norm[0], pe_first3_norm[1], pe_first3_norm[2]], axis=-1)
            
            # Plucker coordinates - last 3 channels
            pe_last3 = pe_view[3:]  # Last 3 channels
            pe_last3_norm = (pe_last3 - pe_last3.min()) / (pe_last3.max() - pe_last3.min() + 1e-8)
            pe_last3_rgb = np.stack([pe_last3_norm[0], pe_last3_norm[1], pe_last3_norm[2]], axis=-1)
            
            # Global normalization for single group
            all_pca_single = np.concatenate([pca_img_dino_plucker.flatten(), pca_img_dino.flatten(), pca_img_base.flatten()])
            global_min_single = all_pca_single.min()
            global_max_single = all_pca_single.max()
            pca_img_dino_plucker = (pca_img_dino_plucker - global_min_single) / (global_max_single - global_min_single + 1e-8)
            pca_img_dino = (pca_img_dino - global_min_single) / (global_max_single - global_min_single + 1e-8)
            pca_img_base = (pca_img_base - global_min_single) / (global_max_single - global_min_single + 1e-8)
            
            axes[i, 3].imshow(pca_img_dino_plucker)
            axes[i, 3].set_title(f'View {i} DINO3D Plucker')
            axes[i, 3].axis('off')
            
            axes[i, 4].imshow(pca_img_dino)
            axes[i, 4].set_title(f'View {i} DINO3D No Plucker')
            axes[i, 4].axis('off')
            
            axes[i, 5].imshow(pca_img_base)
            axes[i, 5].set_title(f'View {i} Base Output')
            axes[i, 5].axis('off')
            
            axes[i, 6].imshow(pe_first3_rgb)
            axes[i, 6].set_title(f'View {i} Plucker First 3')
            axes[i, 6].axis('off')
            
            axes[i, 7].imshow(pe_last3_rgb)
            axes[i, 7].set_title(f'View {i} Plucker Last 3')
            axes[i, 7].axis('off')
        
        plt.tight_layout()
        plt.show()

        return combined_output_plucker
    

    def load_3d_checkpoint(self, checkpoint_dir):
        print(f"Loading model from {checkpoint_dir}")
        checkpointer = CheckPoint(checkpoint_dir)
        self.dino3d = checkpointer.load_model(self.dino3d)