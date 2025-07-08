
import torch
import torch.nn as nn
import torch.nn.functional as F

# Frequency Branch
class FrequencyBranch(nn.Module):
    def __init__(self, out_features=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(32, out_features)

    def forward(self, x):
        x_gray = x.mean(dim=1, keepdim=True)
        fft = torch.fft.fft2(x_gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.log(torch.abs(fft_shift) + 1e-8)
        freq_features = self.conv(magnitude)
        freq_features = freq_features.view(freq_features.size(0), -1)
        freq_features = self.fc(freq_features)
        return freq_features

