import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

other_text = clip.tokenize(['tmr']).to(device)

with torch.no_grad():
    text_features = model.encode_text(other_text)
    print(type(text_features))
    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
