import torch
import torch.nn as nn
import torch.nn.functional as F 
from ViT import ViT

class DPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ViT()
        Reassemble_blocks = nn.ModuleList([

        ])
        Fusion_blocks = nn.ModuleList([

        ])
    
    def forward(self, X):
        X = self.backbone(X) 
        return X


if __name__ == '__main__':
    model = DPT()
    dummy_data = torch.randn(size=(10,3,224,224))
    out = model(dummy_data)
    print(type(out))