import torch
import torch.nn as nn
import torch.nn.functional as F


class KLD_Loss(nn.Module):
    def __init__(self, weight: float = 1.):
        super(KLD_Loss, self).__init__()
        self.weight = weight

    def forward(self, output_mu, output_logVar):
        batch_size = output_mu.size(0)
        
        KL_loss = 0.5* (torch.sum(-1 - output_logVar + torch.pow(output_mu, 2) + torch.exp(output_logVar)))
        return self.weight * KL_loss


