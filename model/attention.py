import torch 
import torch.nn as nn 
from torchsummary import summary 

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels=None, bias=True):
        super(AttentionBlock, self).__init__()
        self.g = None 
        self.x = None 
        self.bias = True 
        self.downsampleX = nn.Conv2d(in_channels, inter_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.downsampleG = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        
        
    def forward(self, g, x):
        """
        W need downsample first 
        """
        down_g = self.downsampleG(g)
        down_x = self.downsampleX(x)
        w = nn.ReLU(down_g + down_x)
        
        return 0 

model = AttentionBlock(512, 256, 256)
summary(model, [(512, 8, 8), (512, 16,16)])
        

    
        