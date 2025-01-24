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
from Interpolate import Interpolate


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
        self.labels_tokens = clip.tokenize(self.labels).to(device='cuda')

        # final head
        self.head = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, images, labels=None):
        """
        images:
            - During training we use a fixed vocabulary for all the images | list of strings to be embedded
            - During evaluation: we must be able to define a new list of words
        """

        # LABELS EMBEDDING PROCESSING 
        if labels is None:
            words_tok = self.labels_tokens
        else:
            words_tok = clip.tokenize(labels)
        words_tok = words_tok.to(self.clip_pretrained.token_embedding.weight.device)

        # word_embd.shape = [number_of_words, encode_dimension]
        words_embd = self.clip_pretrained.encode_text(words_tok)

        # output shape = (batch_size, encode_dimension, W', H') | in the paper H' = H/2
        img_embd = self.DPT(images)
        img_shape = img_embd.shape

        # normalized features (why? taken from original repo)
        img_embd = img_embd / img_embd.norm(dim=-1, keepdim=True)
        words_embd = words_embd / words_embd.norm(dim=-1, keepdim=True)

        # WORD-PIXEL CORRELATION TENSOR 
        # compute similarity | this are the logits
        # computational trick, send everything to 2D and then reshape
        img_embd = img_embd.permute(0,2,3,1).view(-1,img_shape[1])
        correlation_tensor = torch.matmul(img_embd,words_embd.transpose(0,1))
        correlation_tensor = correlation_tensor * torch.exp(self.temperature)
        correlation_tensor = correlation_tensor.view(img_shape[0],img_shape[2],img_shape[3],img_shape[1]).permute(0,3,1,2)

        out = self.head(correlation_tensor) # shape (batch_size, encode_dimension, W, H)
        return correlation_tensor
