import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ClipModel(nn.Module):
    def __init__(self, args):
        super(ClipModel, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, image_embedding,  text_embedding):
        image_features = torch.mean(image_embedding, 1)
        text_features = torch.mean(text_embedding, 1)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        sim_targets = torch.zeros(logits_per_image.size()).to(image_features.device)
        sim_targets.fill_diagonal_(1)
        loss_image = -torch.sum(F.log_softmax(logits_per_image, dim=1)*sim_targets,dim=1).mean()
        loss_text = -torch.sum(F.log_softmax(logits_per_text, dim=1)*sim_targets,dim=1).mean()
        loss = (loss_image + loss_text)/2
        return loss, loss_image, loss_text