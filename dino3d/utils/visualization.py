import torch
from einops import rearrange
from sklearn.decomposition import PCA


def merged_pca(features_3d,
               features_2d,
               ):

    N1, C, H, W = features_3d.shape
    N2, C, H2, W2 = features_2d.shape
    # Check H and W dimensions
    if H != H2 or W != W2:
        features_2d = torch.nn.functional.interpolate(
            features_2d, size=(H, W), mode='bilinear', align_corners=False
        )
    features = torch.cat([features_3d, features_2d], dim=0).detach().cpu().float()
    # Reshape features to 2D
    num_samples, num_features = features.shape[0], features.shape[1]
    reshaped_features = rearrange(features, "n c h w -> (n h w) c")

    # Perform PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(reshaped_features)
    pca_result = torch.from_numpy(pca_result)
    pca_result = rearrange(pca_result, "(n h w) c -> n c h w", n=num_samples, h=H, w=W)

    features_3d = pca_result[:N1]
    features_2d = pca_result[N1:]

    # Normalize the features to [0, 255]
    features_3d = (features_3d - features_3d.min()) / (features_3d.max() - features_3d.min()) * 255
    features_2d = (features_2d - features_2d.min()) / (features_2d.max() - features_2d.min()) * 255
    features_3d = features_3d.type(torch.uint8)
    features_2d = features_2d.type(torch.uint8)

    return features_3d, features_2d