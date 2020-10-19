from tqdm import tqdm
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import time
from os import path

import torch
from torch.nn import functional as F
from torch.nn.functional import conv2d
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import torchvision
from torchvision import transforms


def patch_mean(images, patch_shape):
    channels, *patch_size = patch_shape
    dimensions = len(patch_size)
    padding = tuple(side // 2 for side in patch_size)

    conv = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]

    # Convolution with these weights will effectively compute the channel-wise means
    patch_elements = torch.Tensor(patch_size).prod().item()
    weights = torch.full((channels, channels, *patch_size), fill_value=1 / patch_elements)
    weights = weights.to(images.device)

    # Make convolution operate on single channels
    channel_selector = torch.eye(channels).bool()
    weights[~channel_selector] = 0

    result = conv(images, weights, stride=1, padding=padding, bias=None)
    return result


def patch_std(image, patch_shape):
    return (patch_mean(image ** 2, patch_shape) - patch_mean(image, patch_shape) ** 2).sqrt()


def channel_normalize(template):
    reshaped_template = template.clone().view(template.shape[0], -1)
    reshaped_template.sub_(reshaped_template.mean(dim=-1, keepdim=True))
    reshaped_template.div_(reshaped_template.std(dim=-1, keepdim=True, unbiased=False) + 10e-10)

    return reshaped_template.view_as(template)


class NCC(torch.nn.Module):

    def __init__(self, template, keep_channels=False):
        super().__init__()

        self.keep_channels = keep_channels

        channels, *template_shape = template.shape
        dimensions = len(template_shape)
        self.padding = tuple(side // 2 for side in template_shape)

        self.conv_f = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]
        self.normalized_template = channel_normalize(template)
        ones = template.dim() * (1,)
        self.normalized_template = self.normalized_template.repeat(channels, *ones)
        # Make convolution operate on single channels
        channel_selector = torch.eye(channels).bool()
        self.normalized_template[~channel_selector] = 0
        # Reweight so that output is averaged
        patch_elements = torch.Tensor(template_shape).prod().item()
        self.normalized_template.div_(patch_elements)

    def forward(self, image, stride):
        result = self.conv_f(image, self.normalized_template, bias=None, stride=stride, padding=self.padding)

        std = patch_std(image, self.normalized_template.shape[1:])

        result.div_(std + 10e-10)
        if not self.keep_channels:
            result = result.mean(dim=1)

        # remove nan values due to sqrt of negative value
        result[result != result] = 0
        result[result != result] = result.min()

        return result


def calc_corr(model, track_l, ref_l, tracks, refs, device, stride, rot, start, end, scorefile):


    trans = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    score_mat = np.zeros((len(np.sort(track_l)), (len(np.sort(ref_l)))), dtype='float64')

    calc_time = time.time()

    if rot == False:

        for x, t in enumerate(tqdm(np.sort(track_l))):

            template = Image.open(path.join(tracks, t))
            template_t = model(trans(template).unsqueeze(0).to(device))[0]
            template_t2 = template_t[:, 3:template_t.shape[1] - 3, 3:template_t.shape[2] - 3]
            ncc = NCC(template_t2)

            for y, r in enumerate(np.sort(ref_l)):
                image = Image.open(path.join(refs, r))
                image_t = model(trans(image).unsqueeze(0).to(device))
                image_t2 = image_t[:, :, 3:image_t.shape[2] - 3, 3:image_t.shape[3] - 3].to(device)

                ncc_response = ncc(image_t2, stride)
                score_mat[x][y] = np.amax(ncc_response.cpu().data.numpy())

    else:

        for x, t in enumerate(tqdm(np.sort(track_l))):

            template = Image.open(path.join(tracks, t))
            template_t = model(trans(template).unsqueeze(0).to(device))[0]
            template_t2 = template_t[:, 3:template_t.shape[1] - 3, 3:template_t.shape[2] - 3]
            ncc = NCC(template_t2)

            for y, r in enumerate(np.sort(ref_l)):

                img = Image.open(path.join(refs, r))
                for i in range(start, end):
                    if i == start:
                        img2 = img.rotate(i, expand=1, fillcolor='white')
                        img2 = trans(img2).unsqueeze(0)
                        img_batch = torch.zeros(abs(start - end), 3, img2.shape[2], img2.shape[3])
                    else:
                        img2 = img.rotate(i)
                        img2 = trans(img2).unsqueeze(0)
                        img_batch[i][:, :img2[0].shape[1], :img2[0].shape[2]] = img2[0]

                img_t_batch = model(img_batch.to(device))

                ncc_response = ncc(img_t_batch)

                cc = 0

                for i in range(abs(start - end) - 1):
                    if cc < torch.max(ncc_response[i]).item():
                        cc = torch.max(ncc_response[i]).item()

                score_mat[x][y] = cc

    elapsed = time.time() - calc_time
    print("elapsed time:", elapsed)
    np.save(scorefile, score_mat)
    print("score_mat saved")
    return score_mat