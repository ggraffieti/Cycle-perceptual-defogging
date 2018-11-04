import torch
from torch import nn
from torchvision import models


class PerceptualModel:
    def __init__(self, device):
        self.vgg16 = models.vgg16(pretrained=True).features.to(device).eval()
        self.pool2Features = []
        self.pool5Features = []
        # don't accumulate gradient
        # for param in self.vgg16.parameters():
        #    param.require_grad = False
        self.vgg16[9].register_forward_hook(self.hook)
        self.transform = Transform(device).to(device)
        self.loss_function = nn.MSELoss(size_average=True, reduce=True)

    def hook(self, model, input, output):
        self.pool2Features.append(output.data)

    def perceptual_loss(self, original_image, cycle_image):
        """
        Calculate the perceptual loss of the 2 given images.
        The original and cycle image have to be in the range [-1, 1].
        """
        self.pool2Features.clear()
        self.pool5Features.clear()
        self.pool5Features.append(self.vgg16(self.transform(original_image)))
        self.pool5Features.append(self.vgg16(self.transform(cycle_image)))
        l1 = self.loss_function(self.pool2Features[0].data, self.pool2Features[1].data)
        l2 = self.loss_function(self.pool5Features[0].data, self.pool5Features[1].data)
        return l1 + l2


class Transform(nn.Module):
    def __init__(self, device, cnn_normalization_mean=torch.Tensor([0.485, 0.456, 0.406]),
                 cnn_normalization_std=torch.Tensor([0.229, 0.224, 0.225])):
        super(Transform, self).__init__()
        self.mean = torch.Tensor(cnn_normalization_mean).to(device).view(-1, 1, 1)
        self.std = torch.Tensor(cnn_normalization_std).to(device).view(-1, 1, 1)

    def forward(self, img):
        img = (img + 1) / 2
        return (img - self.mean) / self.std
