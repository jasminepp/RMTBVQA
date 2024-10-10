import torch
import torch.nn as nn
from .gather import GatherLayer

class Contrastive_loss(nn.Module):
    def __init__(self, batch_size, temperature, device, world_size):
        super(Contrastive_loss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.world_size = world_size
    
    def forward(self, z_i, z_j, z_i_patch, z_j_patch,  z_i_predict, z_i_T, quality_label,content_label):
        
        N1 = 2*z_i_patch.shape[0] * self.world_size
        N2 = 2*z_i.shape[0] * self.world_size
        
        #Quality_loss
        z_quality = torch.cat((z_i_patch, z_j_patch), dim=0)
        quality_labels = torch.cat((quality_label, quality_label),dim=0)

        z_quality = nn.functional.normalize(z_quality, p=2, dim=1)
        sim = torch.mm(z_quality, z_quality.T) / self.temperature
        quality_labels = quality_labels.cpu()
        
        positive_mask = torch.mm(quality_labels.to_sparse(), quality_labels.T)
        positive_mask = torch.clamp(positive_mask, min=0, max=1)
        positive_mask = positive_mask.fill_diagonal_(0).to(sim.device)

        zero_diag = torch.ones((N1, N1)).fill_diagonal_(0).to(sim.device)
        positive_sum = torch.sum(positive_mask, dim=1)
        denominator = torch.sum(torch.exp(sim)*zero_diag,dim=1)
        quality_loss = torch.mean(torch.log(denominator) - \
                          (torch.sum(sim * positive_mask, dim=1)/positive_sum))

        
        # Content_loss 
        z_content1 = torch.cat((z_i_predict, z_i_T), dim=0)
        content_labels = torch.cat((content_label, content_label),dim=0)
        content_labels = content_labels.cpu()

        if self.world_size > 1:
            z_content1 = torch.cat(GatherLayer.apply( z_content1), dim=0)
            content_labels = torch.cat(GatherLayer.apply(content_labels), dim=0)
        
        # calculate similarity and divide by temperature parameter
        z_content1 = nn.functional.normalize(z_content1, p=2, dim=1)
        sim1 = torch.mm(z_content1, z_content1.T) / self.temperature
        zero_diag = torch.ones((N2, N2)).fill_diagonal_(0).to(sim1.device)
        positive_mask = torch.mm(content_labels.to_sparse(), content_labels.T)
        positive_mask = positive_mask.fill_diagonal_(0).to(sim1.device)
        positive_sum = torch.sum(positive_mask, dim=1)
        denominator1 = torch.sum(torch.exp(sim1)*zero_diag,dim=1)
        content_loss =torch.mean( torch.log(denominator1) - (torch.sum(sim1 * positive_mask, dim=1)/positive_sum) )

        loss = quality_loss + self.coef1 * content_loss
        return loss

