import torch
import torch.nn as nn


class ResidualConvUnit(nn.Module):
    """
    Modified from the original DPT repo
    Also:
    See RefineNet: https://arxiv.org/pdf/1611.06612. Revise Sec. 3.3: Identity Mapping in RefineNet
    Residual convolution module (Retains same dimension and spatial size).
    Batch normalization helps for segmentation.

    To keep track of how dimensions change, remember: 
    - In a conv2d H -> {H_in + 2*padding - dilation*(kernel_size - 1) - 1}/stride + 1
    - if H_in = token_dimen => H_out = (H_in + 2 - 2 - 1)/1 + 1 = H_in 
    - then the resulting tensor will be of shape
    (B, channels = token_, H_out = H_in, W_out = W_in)
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
        """
        - Output: size=(same as input)
        """
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
    """
    input, after `reassemble_block`: (B, D, W/s, H/s)
    output: (B, D, W/(s/2), H/(s/2)) | meaning inpute is scaled by 2
    """
    def __init__(self, features, use_bn):
        super().__init__()

        out_features = features
        self.ResConv1 = ResidualConvUnit(features, use_bn)
        self.ResConv2 = ResidualConvUnit(features, use_bn)
        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

    def forward(self, X, prev_reassemble):
        X = self.ResConv1(X)
        X = torch.add(X,prev_reassemble)
        X = self.ResConv2(X)
        X = nn.functional.interpolate(X, scale_factor=2, mode="bilinear", align_corners=True)
        X = self.out_conv(X)
        return X
    
if __name__ == '__main__':
    dummy_array = torch.randn((10,256,20,20))
    # expected output after passing through Fusion: (10,256,40,40)
    model = Fusion(features=256,use_bn=False)
    out = model(dummy_array,0)
    print(out.shape)


    