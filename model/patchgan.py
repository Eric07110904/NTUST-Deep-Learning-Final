import torch.nn as nn 
import torch 

class PatchGAN(nn.Module):
    def __init__(self, dim=64):
        super(PatchGAN, self).__init__()
        
        self.dim = dim 
        layers = nn.ModuleList()
        
        layers.append(self._building_block(6, self.dim, False))
        layers.append(self._building_block(self.dim, self.dim * 2))
        layers.append(self._building_block(self.dim * 2, self.dim * 4))
        layers.append(self._building_block(self.dim * 4, self.dim * 8, stride=1))
        layers.append(
            nn.Sequential(
                nn.Conv2d(self.dim * 8, 1, 4, 1, 1),
                nn.Sigmoid()
            )
        )  
        self.layers = nn.Sequential(*layers)
    
    def forward(self, image):
        image = self.layers(image)
        return image 
    
    def _building_block(self, in_channel, out_channel, norm=True, stride=2):
        layers = []
        layers.append(
            nn.Conv2d(in_channel, out_channel, 4, stride=stride, padding=1)
        )
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        layers.append(nn.LeakyReLU(0.2, True))
        return nn.Sequential(*layers)
    
