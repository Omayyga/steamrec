import torch
import numpy as np

from img import LoadImageViaURL, TryLoadUploadedImg
from db import all_fetch

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from fastapi import UploadFile

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# >> accept either a tensor or a HF model output containing the embedding tensor. <<
def _embedding_tensor(vec) -> torch.Tensor:
    if isinstance(vec, torch.Tensor):
        return vec

    poolerOutput = getattr(vec, "pooler_output", None)
    if isinstance(poolerOutput, torch.Tensor):
        return poolerOutput

    raise TypeError(f"Unsupported embedding type: {type(vec)!r}")

# >> convert tensor to numpy array. also normalised. <<
def _normalize_embedding(vec) -> np.ndarray:
    vec = _embedding_tensor(vec)
    vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy()[0]

# >> convert PIL -> clip embedding vector <<<
def EmbedPILImg(img: Image.Image) -> np.ndarray:
    input = processor(images=img, return_tensors="pt")
    input =  {k: v.to(DEVICE) for k, v in input.items()}

    with torch.no_grad():
        imgFeatures = model.get_image_features(**input)

    return _normalize_embedding(imgFeatures)

# >> pull image from url -> embed via clip. <<<
def EmbedImgURL(url:str) -> np.ndarray:
    img = LoadImageViaURL(url)
    return EmbedPILImg(img)

# >> upload embedding helper <<
def EmbedUploaded(file: UploadFile):
    """
    attempt to load and embed uploaded image.
    should return:
    (embedding, None) on success. // (None, error message) on failure.
    """
    img, err = TryLoadUploadedImg(file)
    if err:
        return None, err
    
    return EmbedPILImg(img), None

# >> similarity helper <<
def CosSimilarity(vecA, vecB) -> float:
    """
    return cosine similarities between two vectors..
    """
    return float(np.dot(vecA, vecB))

def embedSSRows(limit: int = 200):
    """
    load some screenshot rows from db to embed.
    exp return -> list of dicts:
    i.e..
    [
        {
            "appid": 123,
            "url": "https://cdn.akamai.steamstatic.com/steam/apps/123/ss_xxx.jpg",
            "embedding": np.array([...])
        },
        ...
    """
    rows = all_fetch(
        """
        SELECT appid, url
        FROM app_screenshots
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (limit,)
    )

    embeddedRows = []

    for r in rows:
        try:
            emb = EmbedImgURL(r["url"])
            embeddedRows.append({
                "appid": r["appid"],
                "url": r["url"],
                "embed": emb,
            })
        # >> exception should skip broken urls. Prevents crashing whole search (?) <<
        except Exception:
            continue

    return embeddedRows