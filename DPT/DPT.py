import torch
import torch.nn as nn
import torch.nn.functional as F
from Reassemble import Reassemble
from ViT import ViT

class DPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ViT()
        D_in = self.backbone.hidden_dim
        patch_size =  self.backbone.patch_size
        s = [4,8,16,32]
        Reassemble_blocks = nn.ModuleList([
            Reassemble(s=s[0], embedding_dimension=D_in, patch_size=patch_size),
            Reassemble(s=s[1], embedding_dimension=D_in, patch_size=patch_size),
            Reassemble(s=s[2], embedding_dimension=D_in, patch_size=patch_size),
            Reassemble(s=s[3], embedding_dimension=D_in, patch_size=patch_size),
        ])
        Fusion_blocks = nn.ModuleList([

        ])
    
    def forward(self, X):
        X = self.backbone(X) 
        return X


if __name__ == '__main__':
    model = DPT()
    dummy_data = torch.randn(size=(10,3,320,320))
    out = model(dummy_data)
    print(type(out))