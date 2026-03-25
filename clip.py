import torch
import numpy as np

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from img import LoadImageViaURL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# >> convert tensor to numpy array. also normalised. <<
def _normalize_embedding(vec: torch.Tensor) -> np.ndarray:
    vec = vec / vec.norm (dim=-1, keepdim=True)
    return vec.cpu().numpy()[0]

def EmbedPILImg(img: Image.Image) -> np.ndarray:
    input = processor(images=img, return_tensors="pt")
    input =  {k: v.to(DEVICE) for k, v in input.items()}

    with torch.no_grad():
        imgFeatures = model.get_image_features(**input)

    return _normalize_embedding(imgFeatures)