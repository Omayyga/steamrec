import requests
from io import BytesIO
from PIL import Image

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