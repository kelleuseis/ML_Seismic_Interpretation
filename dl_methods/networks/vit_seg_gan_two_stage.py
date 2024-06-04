import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_seg_modeling import VisionTransformer as ViT_seg
from .vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

class TSViT_seg(nn.Module):
    def __init__(self):
        super(TSViT_seg, self).__init__()
        
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 3
        config_vit.n_skip = 3
        img_size = 224
        vit_patches_size = 16
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        
        self.stage1 = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes)
        self.stage1.load_from(weights=np.load('model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'))
        
        config_vit.n_classes = 8
        
        self.stage2 = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes)
        self.stage2.load_from(weights=np.load('model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'))
        
    
    def forward(self, inputs):
        inputs_2 = inputs.clone()
        
        outputs_1_logits = self.stage1(inputs)
        outputs_1 = F.log_softmax(outputs_1_logits, dim=1)
        _, fault_channel_pred = torch.max(outputs_1, 1)
        
        non_background_mask = outputs_1_logits.argmax(dim=1) != 2

        outputs_2_logits = self.stage2(inputs_2)
        non_background_mask = non_background_mask.unsqueeze(1).repeat(1, outputs_2_logits.shape[1], 1, 1)
        min_logits_per_class, _ = torch.min(outputs_2_logits.view(outputs_2_logits.size(0), outputs_2_logits.size(1), -1), dim=2)

        outputs_1_logits_faults = torch.zeros([outputs_1_logits.shape[0], 1, outputs_1_logits.shape[2], outputs_1_logits.shape[3]]).cuda()
        outputs_1_logits_channels = torch.zeros([outputs_1_logits.shape[0], 1, outputs_1_logits.shape[2], outputs_1_logits.shape[3]]).cuda()
        outputs_2_logits_facies = torch.zeros([outputs_2_logits.shape[0], 8, outputs_2_logits.shape[2], outputs_2_logits.shape[3]]).cuda()
        outputs_2_logits_facies[non_background_mask] = (
            min_logits_per_class[:, :, None, None].expand_as(outputs_2_logits_facies)[non_background_mask] +\
            outputs_2_logits_facies[non_background_mask]
        ) / 4
        
        outputs_1_logits_faults = outputs_1_logits[:, 1:2, :, :]
        outputs_1_logits_channels = outputs_1_logits[:, 0:1, :, :]
        outputs_2_logits_facies = outputs_2_logits[:, 1:, :, :]
        
        outputs_logits = torch.cat([outputs_1_logits_channels, outputs_1_logits_faults, outputs_2_logits_facies], 1)
        
        return outputs_1_logits, outputs_2_logits, outputs_logits
    
    
class Discriminator(nn.Module):
    '''Patch Discriminator from VQGAN repository (https://github.com/dome272/VQGAN-pytorch/tree/main)'''
    def __init__(self, pred_channels=1, num_filters_last=64, n_layers=3):
        super(Discriminator, self).__init__()

        layers = [nn.Conv2d(pred_channels, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)