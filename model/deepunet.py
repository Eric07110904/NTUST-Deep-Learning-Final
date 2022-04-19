import torch 
import torch.nn as nn 
from torchsummary import summary 
from attention import AttentionBlock

class DeepUNet(nn.Module):
    def __init__(self, bias=True):
        super(DeepUNet, self).__init__()
        self.bias = bias 
        self.dim = 64
        self.first_layer = nn.Sequential(
            nn.Conv2d(15, self.dim, 3, 1, 1, bias=bias),
            nn.BatchNorm2d(64),
        )
        self.last_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.dim, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.gate_block = nn.Sequential(
            nn.Conv2d(self.dim * 8, self.dim * 8, kernel_size=1, stride=1, bias=self.bias),
            nn.BatchNorm2d(self.dim * 8)
        )
        self.encoder_blocks = self._down_sample()
        self.decoder_blocks = self._up_sample()
        self.attention_blocks = self._attention_blocks()
    
    def forward(self, sketch, color):
        #print(sketch.shape, color.shape)
        cache = []
        image = torch.cat([sketch, color], 1) # batch_size, 3+12, 512, 512
        image = self.first_layer(image)
       
        """
        Encoding 
        """
        for i, layer in enumerate(self.encoder_blocks):
            image, connection, idx = layer(image)
            cache.append((connection, idx))
        cache = list(reversed(cache))
        gate = self.gate_block(image)
        attentions = []
        print("[INFO] after encoding ", image.shape)
        """
        Decoding 
        """
        for i, (layer, attention, (connection, idx)) in enumerate(zip(self.decoder_blocks, self.attention_blocks, cache)):
            #print("gate ", gate.shape," connection ", connection.shape)
            connection, attr = attention(gate, connection)
            #print(connection.shape, attr.shape)
            image = layer(image, connection, idx)
            #print(image.shape) # decode's output 
        image = self.last_layer(image)
        return image 
    
    def _attention_blocks(self):
        layers = nn.ModuleList()
        layers.append(AttentionBlock(self.dim * 8, self.dim * 8))
        layers.append(AttentionBlock(self.dim * 8, self.dim * 8))
        layers.append(AttentionBlock(self.dim * 8, self.dim * 8))
        layers.append(AttentionBlock(self.dim * 8, self.dim * 8))
        layers.append(AttentionBlock(self.dim * 4, self.dim * 8))
        layers.append(AttentionBlock(self.dim * 2, self.dim * 8))
        return layers 
    
    def _up_sample(self):
        layers = nn.ModuleList()
        # in_channel * 2 is because U-net concatentate feature map 
        layers.append(DeepUNetUpSample(self.dim * 8 * 2, self.dim * 8, self.bias, True))
        layers.append(DeepUNetUpSample(self.dim * 8 * 2, self.dim * 8, self.bias, True))
        layers.append(DeepUNetUpSample(self.dim * 8 * 2, self.dim * 8, self.bias, True))
        layers.append(DeepUNetUpSample(self.dim * 8 * 2, self.dim * 4, self.bias, True))
        layers.append(DeepUNetUpSample(self.dim * 4 * 2, self.dim * 2, self.bias, True))
        layers.append(DeepUNetUpSample(self.dim * 2 * 2, self.dim , self.bias, True))
        return layers 
    
    def _down_sample(self):
        layers = nn.ModuleList()
        """
            after: 128 x 256 x 256
        """
        layers.append(DeepUNetDownSample(self.dim, self.dim * 2, self.bias))
        """
            after: 128 x 256 x 256
        """
        layers.append(DeepUNetDownSample(self.dim * 2, self.dim * 4, self.bias))
        """
            after: 128 x 256 x 256
        """
        layers.append(DeepUNetDownSample(self.dim * 4, self.dim * 8, self.bias))
        """
            after: 128 x 256 x 256
        """
        layers.append(DeepUNetDownSample(self.dim * 8, self.dim * 8, self.bias))
        """
            after: 128 x 256 x 256
        """
        layers.append(DeepUNetDownSample(self.dim * 8, self.dim * 8, self.bias))
        """
            after: 128 x 256 x 256
        """
        layers.append(DeepUNetDownSample(self.dim * 8, self.dim * 8, self.bias))
        
        return layers 
        

        
class DeepUNetDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(DeepUNetDownSample, self).__init__()
        # conv layer just do some feature extraction
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True) # down sample

        if in_channels == out_channels:
            self.channel_map = nn.Sequential()
        else:
            self.channel_map = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        feature = torch.relu(x)
        feature = self.conv1(feature)
        feature = self.norm1(feature)
        feature = torch.relu(feature)
        feature = self.conv2(feature)
        feature = self.norm2(feature)
        connection = feature + self.channel_map(x)
        feature, idx = self.pool(connection)
        return feature, connection, idx
    
class DeepUNetUpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, dropout=False):
        super(DeepUNetUpSample, self).__init__()
        self.pool = nn.MaxUnpool2d(2, 2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(0.5, True) if dropout else None
        if (in_channels // 2) == out_channels:
            self.channel_map = nn.Sequential()
        else:
            self.channel_map = nn.Conv2d((in_channels // 2), out_channels,1, bias=False)

    def forward(self, x, connection, idx):
        x = self.pool(x, idx) # upsample 
        feature = torch.relu(x)
        feature = torch.cat([feature, connection], 1)
        feature = self.conv1(feature)
        feature = self.norm1(feature)
        feature = torch.relu(feature)
        feature = self.conv2(feature)
        feature = self.norm2(feature)
        feature = feature + self.channel_map(x)
        if self.dropout is not None:
            feature = self.dropout(feature)
        return feature 
model = DeepUNet(bias=True).to("cuda")


sketch = torch.ones((1, 3, 512, 512)).to("cuda")
colors = torch.ones((1, 12, 512, 512)).to("cuda")



res = model(sketch, colors)
print(res.shape)