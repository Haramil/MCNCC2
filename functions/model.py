import torch
from torch import nn
import torchvision.models as models

def create_model(avgpool_bool):

    device = torch.device('cuda:0')

    googlenet = models.googlenet(pretrained=True)
    model = nn.Sequential(*list(googlenet.children())[0:4])

    if avgpool_bool:
        model = nn.Sequential(model, nn.AvgPool2d(2, stride=1))

    model.to(device)
    model.eval()

    return device, model