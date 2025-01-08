import torch
import torch.nn as nn



class ResidualConvUnit(nn.Module):
    """
    See https://arxiv.org/pdf/1611.06612
    Copied from DPT repository'
    Residual convolution module (Retains same dimension and spatial size).
    Batch normalization helps for segmentation.
    """

    def __init__(self, token_dim, use_bn):
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(
            in_channels=token_dim,
            out_channels=token_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.use_bn,
        )
        self.conv2 = nn.Conv2d(
            in_channels=token_dim,
            out_channels=token_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.use_bn,
        )
        self.relu = nn.ReLU()
        if self.use_bn is True:
            self.bn1 = nn.BatchNorm2d(token_dim)
            self.bn2 = nn.BatchNorm2d(token_dim)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, X):
        out = self.relu(X)
        out = self.conv1(out)
        if self.use_bn:
            out = self.bn1(out)

        out = self.relu(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        return self.skip_add.add(out, X)



class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        

        self.ResConv = None


    def forward(self, X):
        X = self.ResConv(X)
        X = self.Resample(X)
        X = self.Project(X)
        
        pass 