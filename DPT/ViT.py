import torch
import torch.nn as nn
import torchvision
from torchvision.models import vit_l_16
from torchvision.models import ViT_L_16_Weights

class ViT(nn.Module):
    """
    
    """
    def __init__(self):
        super().__init__()
        
        self.hooks = [4, 11, 17, 23]
        self.model = torchvision.models.vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
        
        # Access the encoder
        self.encoder = self.model.encoder.layers
        
        # Ensure the encoder has the layers we expect
        assert hasattr(self.encoder, f'encoder_layer_0'), "Encoder layer naming is inconsistent!"
        self.features = {}

        # Register forward hooks
        for hook in self.hooks:
            layer_name = f'encoder_layer_{hook}'
            if hasattr(self.encoder, layer_name):
                getattr(self.encoder, layer_name).register_forward_hook(self.save_output_hook(layer_name))
            else:
                print(f"Warning: {layer_name} not found in encoder layers.")
            
    def save_output_hook(self, layer_key):
        def save_output(_, __, output):
            self.features[layer_key] = output
        return save_output

    def forward(self, x):
        """
        - Admits images of size (B, C, H, W)
        - Outputs a dictionary: {enconder_name : transformer_output (size=(B,197,D))}
        """
        self.model(x)
        return self.features
    

if __name__ == '__main__':
    silly_model = ViT()
    dummy_data = torch.randn(size=(10,3,224,224))
    out = silly_model(dummy_data)
    assert len(out) == len(silly_model.hooks), print(f'size of the output : {len(out)}')
    print(out['encoder_layer_11'].shape)