import torch
import torch.nn as nn
import math

class Read(nn.Module):
    """
    1st step:
    Assums the number of patches is 197
    Input for `forward` is of size `transformer_output (size=(B,1+num_patches,D))`
    Then if follows the process:
    - (B,1+num_patches,D) -> separate and get (B,1,D) and (B,196,D)
    - Concatenate the readout tensor (B,1,D) to each of the `num_patches` embeddings
    - It will leaves us with (B, num_patches, 2D)
    - Get back the original dimension by applying a linear layer followed by GELU. That's 
    (B,num_patches,2D) to (B,num_patches,D)
    """
    def __init__(self, embedding_dimension):
        super().__init__()

        self.D = embedding_dimension
        self.linear = nn.Linear(2*self.D, self.D)
        self.activation = nn.GELU()

    def forward(self,X):
        """
        Input should be of size `transformer_output (size=(B,1+196,D))`
        """
        readout = X[:,0,:].unsqueeze(1) # shape = (B,1,D)
        other_embed = X[:,1:,:] # shape = (B,196,D)
        readout_expand = readout.expand(-1,196,-1)
        concatenation = torch.cat([other_embed,readout_expand], dim=-1)
        assert concatenation.shape == torch.Size([X.shape[0],196,2*self.D]), 'Reassemble-Read module | Concatenation not working | shape does not match'

        projection = self.linear(concatenation)
        projection = self.activation(projection)
        assert projection.shape == torch.Size([X.shape[0],196,self.D]), 'Reassemble-Read module | projection not working | shape does not match'

        return projection


class Concatenate(nn.Module):
    """
    2nd step:
    - Input of size: (B,num_patches,embedding_dimension-usually-1024)
    - Reshape (N_p x D) to (H/p x W/p x D)
    - We are assuming W/p = H/p
    """
    def __init__(self):
        super().__init__()
        
    def forward(self,X):
        batch_size, num_patches, embedding_dimension = X.shape
        return X.view(batch_size, int(math.sqrt(num_patches)), int(math.sqrt(num_patches)), embedding_dimension)

class Resample(nn.Module):
    """
    3th step:
    - Pass the representation from (H/p, W/p, D) to (H/s, W/s, D')
    - From shallow to deep transformers, we use s in [4, 8, 16, 32] correspondingly
    """
    def __init__(self,s:int):
        super().__init__()
    
    def forward(self):
        pass

class Reassemble(nn.Module):
    """
    input: (size=(B,1+196,D)
    Why doing a resample? 
    """

    # `s` admits only numbers [4,8,16,32]
    admit = [4,8,16,32]
    
    def __init__(self,s: int, embedding_dimension: int):
        super().__init__()
        assert s in admit, 'value for s is invalid'
        
        self.reassemble = nn.Sequential(
            Read(embedding_dimension),
            Concatenate(),
            Resample(s)
        )

    def forward(self, X):
        return self.reassemble(X)


if __name__ == '__main__':

    def test_read():
        embedding_dimension = 1024
        foo = Read(embedding_dimension=embedding_dimension)
        dummy_array = torch.Tensor(size=(10,197,embedding_dimension))
        out = foo(dummy_array)
        print(out.shape)
    
    def test_concatenate():
        embedding_dimension = 1024
        dummy_array = torch.Tensor(size=(10,197,embedding_dimension))
        process = nn.Sequential(Read(embedding_dimension=embedding_dimension),Concatenate())
        out = process(dummy_array)
        print(out.shape) # should print (B,H/p,W/p,D)


    test_read()
    test_concatenate()