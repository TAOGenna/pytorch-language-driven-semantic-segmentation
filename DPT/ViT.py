import torch
import torch.nn as nn
import torchvision
from torchvision.models import vit_l_16
from torchvision.models import ViT_L_16_Weights
from torchvision.models.vision_transformer import interpolate_embeddings


class ViT(nn.Module):
    """
    Backbone of the architecture is ViT_l_16
    - patch_size = 16x16
    - embedding_dimension = 1024
    - Admits images of 224x224 but we change it to 320x320 or any image with W and H mod 16 = 0.
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

        #TODO: analyze what components should have require_grad = False

        # The Vit_l_16 was trained on 224x224 images with a patch_size of 16. To capture more details of the image we must increase the `image_size` to 320x320 and maintain the `patch_size`, thus obtaining more patches
        self.model.image_size = 320
        
        # Since we are changing the `image_size` we must also interpolate the position embedding to match this change otherwise it will stay with the positions for a 224x224 image

        # Arguments of interpolate_embeddings:
        # Args:
        # image_size (int): Image size of the new model.
        # patch_size (int): Patch size of the new model.
        # model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
        new_pos_embed = interpolate_embeddings(image_size=self.model.image_size, patch_size=self.model.patch_size,model_state=self.model.state_dict())
        
        # next we update the position embedding with the interpolation we just computed 
        self.model.encoder.pos_embedding.data = new_pos_embed['encoder.pos_embedding']
            
    def save_output_hook(self, layer_key):
        def save_output(_, __, output):
            self.features[layer_key] = output
        return save_output

    def forward(self, X):
        """
        - Admits images of size (B, C, H, W)
        - Outputs a dictionary: {enconder_name : transformer_output (size=(B,1ClassToken+#Patches,D))}
        """
        # check image dimensions before processing
        assert X.shape[-1] % self.model.patch_size == 0, 'Image shape is not divisible by patch_size'
        assert X.shape[-1] == X.shape[-2], 'H and W does not match'

        if X.shape[-1] % self.model.patch_size == 0:
            self.model.image_size = X.shape[-1]
            new_pos_embed = interpolate_embeddings(image_size=self.model.image_size, patch_size=self.model.patch_size,model_state=self.model.state_dict())
            self.model.encoder.pos_embedding.data = new_pos_embed['encoder.pos_embedding']

        # end of analizer
        self.model(X)
        return self.features
    

if __name__ == '__main__':
    silly_model = ViT()
    dummy_data = torch.randn(size=(10,3,320,320))
    out = silly_model(dummy_data) 
    assert len(out) == len(silly_model.hooks), print(f'size of the output : {len(out)}')
    print(out['encoder_layer_11'].shape) # output = (10, 197, 1024)