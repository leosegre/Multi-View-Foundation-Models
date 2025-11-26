import torch
import torch.nn as nn
import torch.nn.functional as F

from dino3d.losses.contrastive_loss import ContrastiveLoss
from dino3d.losses.CosineSimLoss import CosineSimilarityLoss


class ConsistencyLoss(nn.Module):
    def __init__(self,
                 correlation_model,
                 loss_type='Huber',
                 optim_strategy='location',
                 ):
        super(ConsistencyLoss, self).__init__()
        self.correlation_model = correlation_model.to("cuda" if torch.cuda.is_available() else "cpu")
        if loss_type not in ['Huber', 'L1', 'L2']:
            raise ValueError('Invalid loss type. Must be either L1 or L2.')
        if loss_type == 'Huber':
            self.dense_loss = torch.nn.HuberLoss(delta=1/32)
        elif loss_type == 'L1':
            self.dense_loss = torch.nn.L1Loss()
        else:
            self.dense_loss = torch.nn.MSELoss()
        if optim_strategy == 'location':
            self.sparse_loss = ContrastiveLoss()
        elif optim_strategy == 'similarity':
            self.sparse_loss = CosineSimilarityLoss()
        else:
            raise ValueError('Invalid optimization strategy. Must be either location or cosine.')
        self.optim_strategy = optim_strategy

    def forward(self, features, image_ids, pairwise_correspondence, images):

        query_features_img_ids = self.get_batch_idx(image_ids, pairwise_correspondence[..., 0])
        target_features_img_ids = self.get_batch_idx(image_ids, pairwise_correspondence[..., 3])

        sample_locations = self.normalize_coords(pairwise_correspondence[..., 1:3])
        query_features = self.get_query_features(features, query_features_img_ids, sample_locations) # [batch, num_query, c]

        target_locations = self.normalize_coords(pairwise_correspondence[..., 4:6])
        target_sparse_features = self.get_query_features(features, target_features_img_ids, target_locations)

        # Create a batch index tensor
        B, Q = query_features.shape[:2]
        batch_indices = torch.arange(B).unsqueeze(1).expand(B, Q)
        target_features = features[batch_indices, target_features_img_ids]

        correlation_maps = self.calculate_correlation_map(query_features, target_features) # [B, H, W, Q]
        predicted_coords = self.correlation_model(correlation_maps)
        gt_coords = pairwise_correspondence[..., 4:]
        gt_coords = self.normalize_coords(gt_coords)

        dense_loss = self.dense_loss(predicted_coords, gt_coords.to(predicted_coords.dtype))
        sparse_loss = self.sparse_loss(query_features, target_sparse_features)

        return dense_loss, sparse_loss


    def get_query_features(self, features, query_features_idx, locations):
        """
            Args:
                features: Tensor of shape [B, N, C, H, W]
                query_features_idx: LongTensor of shape [B, Q], values in 0..N-1
                locations: Tensor of shape [B, Q, 2]

            Returns:
                sampled_features: [B, Q, C]
        """

        # Use interpolation to get the feature at the given location

        B, N, C, H, W = features.shape
        Q = locations.shape[1]

        # Step 1: Flatten features to [B*N, C, H, W]
        features = features.view(B * N, C, H, W)

        # Step 2: Repeat locations for all N images in the batch
        # [B, Q, 2] -> [B, N, Q, 2] → flatten to [B*N, Q, 2]
        locations_exp = locations.unsqueeze(1).expand(B, N, Q, 2).reshape(B * N, Q, 2)

        # Step 3: Prepare grid for grid_sample → [B*N, Q, 1, 2]
        grid = locations_exp.view(B * N, Q, 1, 2)

        # Step 4: Sample all features from all images
        sampled = F.grid_sample(features, grid.to(features.dtype), mode='bilinear', padding_mode='border')  # [B*N, C, Q, 1]
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # [B*N, Q, C]

        # Step 5: Gather the correct samples using query_features_idx
        sampled = sampled.view(B, N, Q, C)  # [B, N, Q, C]
        gather_idx = query_features_idx.unsqueeze(-1).unsqueeze(1).expand(B, 1, Q, C)
        sampled_features = torch.gather(sampled, dim=1, index=gather_idx)  # [B, 1, Q, C]
        sampled_features = sampled_features.squeeze(1)  # [B, Q, C]

        return sampled_features


    def calculate_correlation_map(self, query_features, target_features):
        # Implement the correlation map calculation
        return torch.cosine_similarity(query_features.unsqueeze(-1).unsqueeze(-1), target_features, dim=2)

    def get_batch_idx(self, image_ids, idx):
        batch_size = idx.shape[0]
        query_features_img_ids = torch.argwhere(image_ids[:, None] == idx[..., None])[..., -1].reshape(batch_size, -1)
        return query_features_img_ids

    def normalize_coords(self, coords):
        # Normalize the coordinates to be between -1 and 1
        return coords * 2 - 1