import torch
from einops import rearrange


def get_feature_cos_sims(fs, ft):
    assert fs.shape == ft.shape

    fs_n = fs.norm(dim=2) # b x s x h x w
    ft_n = ft.norm(dim=2) # b x s x h x w
    return torch.einsum("bschw,bschw->bshw", fs, ft) / (fs_n * ft_n)


def get_emb_norm_regularization_loss(features, base_features):
    refined_emb_norm = features.norm(dim=2) # b x s x c x h x w
    dino_emb_norm = base_features.norm(dim=2) # b x s x c x h x w
    norm_ratio = refined_emb_norm / dino_emb_norm
    return (norm_ratio - 1).abs().mean()

def get_emb_angle_regularization_loss(features, base_features):
    cos_sims = get_feature_cos_sims(features, base_features)
    return (1 - cos_sims).mean()

def get_reg_loss(features, base_features):
    """
    Calculate the regularization loss based on the specified type.

    Args:
        features (torch.Tensor): The features to be regularized.
        base_features (torch.Tensor): The base features for comparison.

    Returns:
        torch.Tensor: The computed regularization loss.
    """
    B, S, C, H1, W1 = features.shape
    _, _, _, H2, W2 = base_features.shape
    if H1 != H2 or W1 != W2:
        features = rearrange(features, 'b s c h w -> (b s) c h w')
        features = torch.nn.functional.interpolate(
            features, size=(H2, W2), mode='bilinear', align_corners=False
        )
        features = rearrange(features, '(b s) c h w -> b s c h w', b=B, s=S)
    norm_loss = get_emb_norm_regularization_loss(features, base_features)
    angle_loss = get_emb_angle_regularization_loss(features, base_features)
    # Combine the two losses
    return norm_loss + angle_loss
    