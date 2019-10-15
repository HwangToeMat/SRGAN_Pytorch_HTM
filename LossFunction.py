import torch
from torch import nn
from torchvision.models.vgg import vgg16
import torch.backends.cudnn as cudnn

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        vgg_loss = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in vgg_loss.parameters():
            param.requires_grad = False
        self.vgg_loss = vgg_loss
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = nn.BCELoss()

    def forward(self, fake_rate, SR, HR):
        # MSE Loss
        MSE_loss = self.mse_loss(SR, HR)
        # VGG Loss
        VGG_loss = self.mse_loss(self.vgg_loss(SR), self.vgg_loss(HR))
        # Adversarial Loss
        Adversarial_loss = self.cross_entropy(fake_rate,torch.ones(fake_rate.size(0)).cuda())
        return MSE_loss + 6e-3 * VGG_loss + 1e-3 * Adversarial_loss

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.cross_entropy = nn.BCELoss()

    def forward(self, fake_rate, real_rate):
        # Fake_img Correct Rate
        Fake_img_CR = self.cross_entropy(fake_rate,torch.zeros(fake_rate.size(0)).cuda())
        # Real_img Correct Rate
        Real_img_CR = self.cross_entropy(real_rate,torch.ones(real_rate.size(0)).cuda())
        return Fake_img_CR + Real_img_CR
