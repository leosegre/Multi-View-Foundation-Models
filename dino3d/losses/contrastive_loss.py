import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def get_logits(self, img1, img2):
        # Compute base logits
        denominator = img1.norm(dim=-1) * img2.norm(dim=-1)
        logits_per_image =  img1 @ img2.transpose(1, 2) / denominator.unsqueeze(1)
        logits_per_text =  img2 @ img1.transpose(1, 2) / denominator.unsqueeze(1)

        return logits_per_image, logits_per_text

    def forward(self, x1, x2):
        total_loss = 0

        device = x1.device
        logits_x1, logits_x2 = self.get_logits(x1, x2)
        labels = torch.arange(logits_x1.shape[-1], device=device).unsqueeze(0).expand(logits_x1.shape[0], -1)
        total_loss += (
                              F.cross_entropy(logits_x1, labels) +
                              F.cross_entropy(logits_x2, labels)
                      ) / 2
        return total_loss
