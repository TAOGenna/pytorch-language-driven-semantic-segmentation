import torch
import torch.nn as nn
import math

class Read(nn.Module):
    """
    1st step:
    Assume the number of patches is 401
    Input for `forward` is of size `transformer_output (size=(B,1+num_patches,D))`
    Then if follows the process:
    - (B,1+num_patches,D) -> separate and get (B,1,D) and (B,196,D)
    - Concatenate the readout tensor (B,1,D) to each of the `num_patches` embeddings
    - It will leaves us with (B, num_patches, 2D)
    - Get back the original dimension by applying a linear layer followed by GELU. That's 
    (B,num_patches,2D) to (B,num_patches,D)
    """
    def __init__(self, embedding_dimension, num_patches = 401):
        super().__init__()

        self.D = embedding_dimension
        self.num_patches = num_patches-1 # subtract the readout token 
        self.linear = nn.Linear(in_features = 2*self.D, out_features = self.D)
        self.activation = nn.GELU()

    def forward(self,X):
        """
        Input should be of size `transformer_output (size=(B,1+#Patches,D))`
        """
        readout = X[:,0,:].unsqueeze(1) # shape = (B,1,D)
        other_embed = X[:,1:,:] # shape = (B,num_patches,D)
        readout_expand = readout.expand(-1,self.num_patches,-1)
        concatenation = torch.cat([other_embed,readout_expand], dim=-1)
        assert concatenation.shape == torch.Size([X.shape[0],self.num_patches,2*self.D]), 'Reassemble-Read module | Concatenation not working | shape does not match'

        projection = self.linear(concatenation)
        projection = self.activation(projection)
        assert projection.shape == torch.Size([X.shape[0],self.num_patches,self.D]), 'Reassemble-Read module | projection not working | shape does not match'

        return projection


class Concatenate(nn.Module):
    """
    2nd step:
    - Input of size: (B,num_patches,embedding_dimension-usually-1024)
    - Reshape (N_p x D) to (H/p x W/p x D)
    - We are assuming W/p = H/p
    """
    def __init__(self,patch_size,image_size):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        
    def forward(self,X):
        batch_size, num_patches, embedding_dimension = X.shape
        return X.view(batch_size, self.image_size//self.patch_size, self.image_size//self.patch_size, embedding_dimension)


class Resample(nn.Module):
    """
    3rd step:
    - Pass the representation from (H/p, W/p, D) to (H/s, W/s, D')
    - From shallow to deep transformers, we use s \in [4, 8, 16, 32]
    """
    def __init__(self, s:int, patch_size:int, embedding_dimension:int):
        super().__init__()
        self.D_prime = 256
        # We implement this operation by first using 1 × 1 convolutions to project the hidden_dimension to D_prime
        self.projection_layer = nn.Conv2d(in_channels=embedding_dimension,out_channels=self.D_prime,kernel_size=1) 
        # Then depending on the value of `s` we use a Conv2D or a ConvTranspose2D to downsample or upsample
        arg = {
            4:{'in_channels':self.D_prime, 'out_channels':self.D_prime,'stride':4, 'dilation':1, 'kernel_size':4},
            8:{'in_channels':self.D_prime, 'out_channels':self.D_prime,'stride':2, 'dilation':1, 'kernel_size':2},
            16:{'in_channels':self.D_prime, 'out_channels':self.D_prime,'stride':1, 'padding':1, 'kernel_size':3},
            32:{'in_channels':self.D_prime, 'out_channels':self.D_prime,'stride':2, 'padding':1, 'kernel_size':3}
            }
        self.spatial_resample = nn.Conv2d(**arg[s]) if s>=patch_size else nn.ConvTranspose2d(**arg[s])
    
    def forward(self,X):
        """
        input: (B, image_size/patch_size, image_size/patch_size, D)
        Remember: 
        - Conv2D admits input of shape: (N, C_in, H_in, W_in). Use `permute`
        """
        X = torch.permute(X,(0,3,1,2))
        X = self.projection_layer(X)
        X = self.spatial_resample(X)
        return X

class Reassemble(nn.Module):
    """
    input: (size=(batch_size, readout_token + num_patches, embedding_dimension)
    output: spatially resampled concatenated patches according to the transformer level
     
    `s` admits only numbers [4,8,16,32]
    """    
    def __init__(self,s: int, embedding_dimension: int, patch_size:int, image_size:int):
        super().__init__()
        admit = [4,8,16,32]
        assert s in admit, 'value s is invalid'
        
        # the input to this sequence is going to be the output of a transformer
        # input size: (B, num_patches+readout, hidden_dimension)
        self.reassemble = nn.Sequential(
            Read(embedding_dimension), # out.shape = (B,num_patches,D)
            Concatenate(patch_size,image_size), # out.shape = (B, image_size/patch_size, image_size/patch_size, D) | assume H=W=image_size
            Resample(s,patch_size,embedding_dimension)
        )

    def forward(self, X):
        """
        Input should be of size `transformer_output (size=(B,1+#Patches,D))`
        Output: (B, D, W/s, H/s)
        """
        return self.reassemble(X)


if __name__ == '__main__':

    def test_read():
        embedding_dimension = 1024
        foo = Read(embedding_dimension=embedding_dimension)
        dummy_array = torch.Tensor(size=(10,401,embedding_dimension))
        out = foo(dummy_array)
        print(out.shape)
    
    def test_concatenate():
        embedding_dimension = 1024
        image_size = 320
        dummy_array = torch.Tensor(size=(10,401,embedding_dimension))
        process = nn.Sequential(Read(embedding_dimension=embedding_dimension),Concatenate(patch_size=16,image_size=image_size))
        out = process(dummy_array)
        print(out.shape) # should print (B, H/p, W/p, D)

    def test_Reassemble():
        embedding_dimension = 1024
        image_size = 320
        patch_size=16
        s = [4,8,16,32]
        dummy_array = torch.Tensor(size=(10,401,embedding_dimension))
        print('-----------------------------------------------------------')
        for s_i in s:
            model = Reassemble(s_i,embedding_dimension,patch_size,image_size)
            out = model(dummy_array)
            print(out.shape)
        print('-----------------------------------------------------------')

    test_read()
    test_concatenate()
    test_Reassemble()