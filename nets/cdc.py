import torch
import torch.nn as nn
import torch.nn.functional as F

class CentralDifferenceConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False, theta=0.7):
        super().__init__()
        self.theta = theta
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.kernel_size = kernel_size

    def forward(self, x):
        out_normal = self.conv(x)
        if self.theta == 0:
            return out_normal

        # Central difference
        kernel_diff = self.conv.weight.sum(dim=[2, 3], keepdim=True)
        out_diff = F.conv2d(x, kernel_diff, bias=None, stride=self.conv.stride, padding=0)
        return out_normal - self.theta * out_diff

