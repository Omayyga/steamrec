import requests

from io import BytesIO
from PIL import Image
from fastapi import UploadFile

# >> image loaded from url. saved as PIL. <<
def LoadImageViaURL(url: str) -> Image.Image:
    response = requests.get(url, timeout=30)
    response.raise_for_status() 

    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img

def imgInfo(img: Image.Image) -> dict:
    return {
        "mode": img.mode,
        "size": img.size,
        "width": img.width,
        "height": img.height,
    }

# >> loads an uploaded img and returns as pil. <<
def LoadUploadedImg(file: UploadFile) -> Image.Image:
    img = Image.open(file.file).convert("RGB")
    return img

# >> failsafe ver of above var. should prevent crashes from alt uploads.. refer back if doesnt work (!!!) <<
def TryLoadUploadedImg(file: UploadFile) -> tuple[Image.Image | None, str | None]:
    try:
        img = Image.open(file.file).convert("RGB")
        return img, None
    except Exception as e:
        return None, "Invalid or unsupported image file.."