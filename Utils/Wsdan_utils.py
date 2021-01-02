"""Utils
Created: Nov 11,2019 - Yuchong Gu
Revised: Dec 03,2019 - Yuchong Gu
"""
import torch
import random
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook as tqdm

class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)

def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])

            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)


def visualize_attention(model, dataset, device, visualize_save_path, batch_size=16):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)
    ToPILImage = transforms.ToPILImage()
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    fakehaimgs = []
    realhaimgs = []

    for i, (inputs, labels) in tqdm(enumerate(dataloader),total=len(dataloader)):
        inputs = inputs.to(device)
        preds, _, attention_maps = model(inputs)
        attention_maps = F.upsample_bilinear(attention_maps, size=(inputs.size(2), inputs.size(3)))
        attention_maps = torch.sqrt(attention_maps.cpu() / attention_maps.max().item())

        heat_attention_maps = generate_heatmap(attention_maps)
        raw_image = inputs.cpu() * STD + MEAN
        heat_attention_image = raw_image * 0.5 + heat_attention_maps * 0.5
        raw_attention_image = raw_image * attention_maps

    for batch_idx in range(inputs.size(0)):
        rimg = ToPILImage(raw_image[batch_idx])
        raimg = ToPILImage(raw_attention_image[batch_idx])
        haimg = ToPILImage(heat_attention_image[batch_idx])
        rimg.save(os.path.join(visualize_save_path, '%03d_raw.jpg' % (i * batch_size + batch_idx)))
        raimg.save(os.path.join(visualize_save_path, '%03d_raw_atten.jpg' % (i * batch_size + batch_idx)))
        haimg.save(os.path.join(visualize_save_path, '%03d_heat_atten.jpg' % (i * batch_size + batch_idx)))
        if labels[batch_idx] == 0:
          fakehaimgs.append(haimg)
        else: 
          realhaimgs.append(haimg)

    _, axes = plt.subplots(nrows=2, ncols=5, figsize=(24, 10))
    for idx, image in enumerate(fakehaimgs[:5], start=0):
        axes.ravel()[idx].imshow(image)
        axes.ravel()[idx].axis('off')
        axes.ravel()[idx].set_title("Label:Fake")
    plt.tight_layout()

    for idx, image in enumerate(realhaimgs[:5], start=5):
        axes.ravel()[idx].imshow(image)
        axes.ravel()[idx].axis('off')
        axes.ravel()[idx].set_title("Label:Real")
    plt.tight_layout()

def generate_heatmap(attention_maps):
    heat_attention_maps = []
    heat_attention_maps.append(attention_maps[:, 0, ...])  # R
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
                               (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
    heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
    return torch.stack(heat_attention_maps, dim=1)
