import torch.nn as nn
import clip
from PIL import Image
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "DPT"))  
from DPT import DPT 
from Lseg.utils.util import get_dataset
import torch
from Lseg.utils.util import ToUniversalLabel, semantic_label_tsv_path
from DPT.Interpolate import Interpolate

class Lseg(nn.Module):
    def __init__(self):
        super().__init__()

        # text encoder
        self.clip_pretrained, _ = clip.load('ViT-B/32', jit=False)
        self.clip_pretrained.requires_grad_(False)
        self.clip_pretrained = self.clip_pretrained.to(device='cuda')

        # image encoder
        self.DPT = DPT(D_out = 512, head='Lseg')

        # predefined labels
        self.labels = ToUniversalLabel.read_MSeg_master(semantic_label_tsv_path)
        self.labels = list(self.labels)
        self.labels[-1] = 'other'

        # Lseg parameters
        self.temperature = nn.Parameter(torch.tensor(0.07))

        # final head
        self.head = Interpolate(scale_factor=2, mode="bilinear", align_corners=True),

    def forward(self, images, labels=None):
        """
        images:
            - During training we use a fixed vocabulary for all the images | list of strings to be embedded
            - During evaluation: we must be able to define the list of words the model can identity
        """

        # LABELS EMBEDDING PROCESSING 
        if labels is None:
            words_tok = clip.tokenize(self.labels)
        else:
            words_tok = clip.tokenize(labels)
        words_tok = words_tok.to(self.clip_pretrained.token_embedding.weight.device)

        # word_embd.shape = [number_of_words, encode_dimension]
        words_embedding = self.clip_pretrained.encode_text(words_tok)

        # output shape = (batch_size, encode_dimension, W', H') | in the paper H' = H/2
        img_embd = self.DPT(images)
        img_shape = img_embd.shape

        # normalized features (why? taken from original repo)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # WORD-PIXEL CORRELATION TENSOR 
        # compute similarity | this are the logits
        # computational trick, send everything to 2D and then reshape
        img_embd = img_embd.permute(0,2,3,1).view(-1,img_shape[1])
        correlation_tensor = torch.matmul(img_embd,words_embedding.transpose(0,1))
        correlation_tensor = correlation_tensor * torch.exp(self.temperature)
        correlation_tensor = correlation_tensor.view(img_shape[0],img_shape[2],img_shape[3],img_shape[1]).permute(0,3,1,2)

        out = self.head(correlation_tensor)

        return correlation_tensor

        
if __name__ == '__main__':
    model = Lseg()
    dummy_img = torch.randn(size=(10,3,320,320))
    text_batches = ['cat', 'dot']
    foo = model(dummy_img, text_batches)
    print(f'--------------- finished with Lseg -------------------------')
    print(f'size of dummy_img: {dummy_img.shape}')
    

