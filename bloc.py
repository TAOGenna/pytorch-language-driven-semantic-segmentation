import torch.nn as nn
import clip
from PIL import Image
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "DPT"))  
from DPT import DPT 
from Lseg.utils.util import get_dataset
import torch

class Lseg(nn.Module):
    def __init__(self):
        super().__init__()

        # text encoder
        self.clip_pretrained, _ = clip.load('ViT-B/32', jit=False)
        self.clip_pretrained.requires_grad_(False)
        self.clip_pretrained = self.clip_pretrained.to(device='cuda')

        # image encoder
        self.DPT = DPT(D_out = 512)


    def forward(self,annotations, images=None):
        """
        images: batch of images of size (batch_size, C, W, H)
        annotations: batch of annotations of size (batch_size, # annotations)
        """
        word_embd = self.clip_pretrained.encode_text(clip.tokenize(annotations).cuda())
        print(f'word embedding: {word_embd.shape}')
        img_embd = self.DPT(images)
        print(f'dpt output = {img_embd.shape}')
        return 'finished foward'

        
if __name__ == '__main__':
    model = Lseg()
    #print(clip.tokenize('asd'))
    #print(dir(model))
    #bar = [['asd','tmr','hola'],['asd','tmr','hola']]
    
    #foo = model.clip_pretrained.encode_text(clip.tokenize(bar).cuda()).float()
    dummy_img = torch.randn(size=(10,3,320,320)) 
    foo = model('hola',dummy_img)
    print(f'--------------- finished with Lseg -------------------------')
    print(f'size of dummy_img: {dummy_img.shape}')
    

