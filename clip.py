import torch
import numpy as np

from img import LoadImageViaURL, TryLoadUploadedImg
from db import all_fetch, exec, single_fetch, timestamp

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

def findTopMatches(queryEmbed, embRows, top_k: int = 5):
    """
    Compares query to stored screenshot embeddings.
    Returns top_k matches based on similarity..
    """

    scored = []

    for r in embRows:
        score = CosSimilarity(queryEmbed, r["embed"])
        scored.append({
            "appid": r["appid"],
            "url": r["url"],
            "score": score,
        })

    scored.sort(key=lambda x: x["score"], reverse = True)
    return scored[:top_k]

# >> note: col = collapse. 
# should help find the singular best match. 
# i.e. if 3 ss match 1 appid, keeps the best result.. <<
def colMatchByAppid(match):
    """
    Collapse ss level matches to appid matches..
    best score per appid is kept.
    """

    bestByAppid = {}

    for m in match:
        appid = m["appid"]

        if appid not in bestByAppid:
            bestByAppid[appid] = m
            continue

        if m["score"] > bestByAppid[appid]["score"]:
            bestByAppid[appid] = m

    col = list(bestByAppid.values())
    col.sort(key=lambda x: x["score"], reverse = True)
    return col

# >> convert float32 enbedding vector to raw bytes <<
def f32toBytes(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()

# >> inverse of above, convert raw bytes back to float32 vector <<
def bytesToF32(blob : bytes, dim : int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count = dim)

def UpsertSSEmbedding(appid: int, url: str, embed: np.ndarray):
    """
    add one screenshot embedding into sqlite
    """

    ts = timestamp()
    exec("""
        INSERT INTO screenshot_embeddings (appid, url, embedding, dim, added_at)
        VALUES (?,?,?,?,?)
            ON CONFLICT(appid, url) DO UPDATE SET
            embedding = excluded.embedding,
            dim = excluded.dim,
            added_at = excluded.added_at
        """,
        (
            appid,
            url,
            f32toBytes(embed),
            int(len(embed)),
            ts,
        )
    )

def GetSSEmbeddingStored(limit: int | None = None) -> list[dict]:
    """load stored embeddings form sqlite"""
    sql = """
        SELECT appid, url, embedding, 
            COALESCE(dim, CAST(length(embedding) / 4 AS INTEGER)) AS dim
        FROM screenshot_embeddings
        """

    params = []

    if limit is not None:
        sql += "\nLIMIT ?"
        params.append(limit)

    rows = all_fetch(sql, tuple(params))
    results = []

    for r in rows:
        results.append({
            "appid": r["appid"],
            "url": r["url"],
            "embed": bytesToF32(r["embedding"], int(r["dim"])),            
        })

    return results

def findStoredTopMatches(queryEmbed, top_k: int = 20, limit: int | None = None):
    """search from stored embeddings"""

    rows  = GetSSEmbeddingStored(limit = limit)
    scored = []

    for r in rows:
        score = CosSimilarity(queryEmbed, r["embed"])
        scored.append({
            "appid": r["appid"],
            "url": r["url"],
            "score": score,
        })

    scored.sort(key = lambda x: x["score"], reverse = True)
    return scored[:top_k]

def findMissingEmb(limit: int | None = 200, appid: int | None = None) -> list[dict]:
    """
    should process rows that are missing
    can also filter by specific appid"""

    sql = """
        SELECT ss.appid, ss.url
        FROM app_screenshots ss
        LEFT JOIN screenshot_embeddings se
            ON ss.appid = se.appid AND ss.url = se.url
        WHERE se.appid IS NULL
    """

    params = []

    if appid is not None:
        sql += "\nAND ss.appid = ?"
        params.append(appid)

    sql += "\nORDER BY ss.appid ASC"

    if limit is not None:
        sql += "\nLIMIT ?"
        params.append(limit)

    rows = all_fetch(sql, tuple(params))

    return [{
        "appid": int(r["appid"]),
        "url": r["url"],
    }
    for r in rows]
    
def embedMissingSS(limit: int | None = 200, appid: int | None = None) -> dict:
    """
    Embeds ONLY screenshot rows that are missing an embedding"""

    rows = findMissingEmb(limit = limit, appid=appid)

    complete = 0
    failed = 0
    failedSample = []

    for r in rows:
        try:
            emb = EmbedImgURL(r["url"])
            UpsertSSEmbedding(r["appid"], r["url"], emb)
            complete += 1
        except Exception as e:
            failed += 1
            if len(failedSample) < 10:
                failedSample.append({
                    "appid": r["appid"],
                    "url": r["url"],
                    "error": str(e),
                })

    return {
        "processed": len(rows),
        "embedded": complete,
        "failed": failed,
        "failedSample": failedSample,
    }
