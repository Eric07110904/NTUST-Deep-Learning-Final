import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchsummary import summary 

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gate_channels, inter_channels=None, bias=True):
        super(AttentionBlock, self).__init__()
        if inter_channels is None:
            inter_channels = in_channels // 2
        self.g = None 
        self.x = None 
        self.bias = True 
        self.downsampleX = nn.Conv2d(in_channels, inter_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.downsampleG = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.W = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=self.bias),
            nn.BatchNorm2d(in_channels)
        )
    def forward(self, g, x):
        """
        W need downsample first 
        """
        down_g = self.downsampleG(g)
        down_x = self.downsampleX(x)
        down_g = F.interpolate(down_g, size=down_x.shape[2:]) #g.shape <= w.shape
        
        activated = torch.sigmoid(self.psi(torch.relu(down_g + down_x)))
        resampled = F.interpolate(activated, size=x.shape[2:])
        result = resampled.expand_as(x) * x 
        result = self.W(result)
        
        return result, resampled

        

    
        