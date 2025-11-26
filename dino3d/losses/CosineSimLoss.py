import torch

class CosineSimilarityLoss(torch.nn.Module):
    """
    Cosine Similarity Loss
    """

    def __init__(self, reduction='mean'):
        super(CosineSimilarityLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x1, x2, dim=1):
        cosine_similarity = torch.nn.functional.cosine_similarity(x1, x2, dim=dim)
        loss = 1 - cosine_similarity

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss