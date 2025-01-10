import torch.nn as nn
import torch.nn.functional as F

class Interpolate(nn.Module):
    """
    This function is only used for the definition of `head`
    """
    def __init__(self, scale_factor, mode, align_corners):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
