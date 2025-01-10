import torch
import torch.nn as nn
import torch.nn.functional as F
from Reassemble import Reassemble
from Fusion import Fusion
from ViT import ViT

class DPT(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hooks = [4, 11, 17, 23]
        self.backbone = ViT(hooks=self.hooks)
        
        D_in = self.backbone.hidden_dim
        patch_size =  self.backbone.patch_size
        image_size = self.backbone.image_size
        
        s = [4,8,16,32]
        
        self.Reassemble_blocks = nn.ModuleList([
            Reassemble(s=s_i, embedding_dimension=D_in, patch_size=patch_size,image_size=image_size) for s_i in s
        ])
        self.Fusion_blocks = nn.ModuleList([
            Fusion(),
            Fusion(),
            Fusion(),
            Fusion(),
        ])

        self.head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
    
    def forward(self, X):
        # returns a dictionary with four elements, the outputs of the transformers
        # Name of the encoders as `f'encoder_layer_{hook}'`, see `self.hooks`
        out_ViT = self.backbone(X)
        out_reassemble = [0]*4
        for idx, hook in enumerate(self.hooks):
            out_reassemble[idx] = self.Reassemble_blocks[idx](out_ViT[f'encoder_layer_{hook}'])
        
        out_fusion = 0 
        for idx, out_reass in enumerate(out_reassemble.reverse()):
            index = 3 - idx 
            out_fusion[index] = self.backboneFusion_block[index](X=out_reassemble[index],prev_reassemble=out_fusion)

        out = self.head(out_fusion)

        return X


if __name__ == '__main__':
    model = DPT()
    dummy_data = torch.randn(size=(10,3,320,320))
    out = model(dummy_data)
    print(type(out))