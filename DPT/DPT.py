import torch
import torch.nn as nn
import torch.nn.functional as F
from Reassemble import Reassemble
from Fusion import Fusion
from ViT import ViT
from Interpolate import Interpolate

class DPT(nn.Module):
    def __init__(self, D_out=None, head = None):
        super().__init__()
        
        self.hooks = [4, 11, 17, 23]
        self.backbone = ViT(hooks=self.hooks)
        use_bn = False
        D_in = self.backbone.model.hidden_dim
        D_out = 512 if D_out is not None else 256
        patch_size =  self.backbone.model.patch_size
        image_size = self.backbone.model.image_size
        
        s = [4,8,16,32]
        
        self.Reassemble_blocks = nn.ModuleList([
            Reassemble(s_i, D_in, patch_size, image_size, D_out) for s_i in s
        ])
        self.Fusion_blocks = nn.ModuleList([
            Fusion(features=D_out, use_bn=use_bn) for _ in range(4)
        ])
        if head is None:
            # Task-specific Head: proposed in the paper for image segmentation tasks
            self.head = nn.Sequential(
                nn.Conv2d(D_out, D_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(D_out),
                nn.ReLU(True),
                nn.Dropout(0.1, False),
                nn.Conv2d(D_out, 3, kernel_size=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            )
        else: # mode == 'Lseg'
            self.head = nn.Identity()
    
    def forward(self, X):
        """
        input: receives an image of size (B, Channels, W, H)
        output: (B, number_classes, W, H)
        """
        self.backbone(X)
        # returns a dictionary with four elements, the outputs of the transformers
        # Name of the encoders as `f'encoder_layer_{hook}'`, see `self.hooks`
        out_ViT = self.backbone.features
        out_reassemble = [0]*4
        for idx, hook in enumerate(self.hooks):
            out_reassemble[idx] = self.Reassemble_blocks[idx](out_ViT[f'encoder_layer_{hook}'])
        # each out_reassemble[i] has shape (B, D_out, W/s_i, H/s_i)
        out_fusion = 0 
        for index in reversed(range(4)):
            out_fusion = self.Fusion_blocks[index](X=out_reassemble[index],prev_reassemble=out_fusion)
        # out_fusion shape is (B, D_out, W/2, H/2)
        out = self.head(out_fusion)
        # out = (B, num_classes=150, W, H) | as it passes through a nn.functional interoplate
        return out


if __name__ == '__main__':
    model = DPT(D_out=512)
    dummy_data = torch.randn(size=(10,3,320,320)) # simulates a batch of RGB images (batch_size, channels, W, H)
    print('INFO ORIGINAL TENSOR')
    print(type(dummy_data))
    # Final output should be of shape (B, num_classes=150, W, H)
    print(dummy_data.shape)
    out = model(dummy_data)
    print('OUTPUT')
    print('if head is NONE:', type(out), 'and with size ', out.shape)
    model = DPT(D_out=512, head='Lseg')
    out = model(dummy_data)
    print('if head is Lseg:', type(out), 'and with size ', out.shape)
    print('---------------------- Everything OK --------------------')
    # Expected output:
    # INFO ORIGINAL TENSOR
    # <class 'torch.Tensor'>
    # torch.Size([10, 3, 320, 320])
    # OUTPUT
    # if head is NONE: <class 'torch.Tensor'> and with size  torch.Size([10, 3, 320, 320])
    # if head is Lseg: <class 'torch.Tensor'> and with size  torch.Size([10, 512, 160, 160])
    # everything ok