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

def _normalize_embedding_batch(vec) -> np.ndarray:
    """
    normalise a batch of embeddings. returns numpy array.
    """
    vec = _embedding_tensor(vec)
    vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy()

def embTxtPrompts(prompts: list[str]) -> np.ndarray:
    """
    convert list of txt prompts to normalised clip embeddings"""

    inputs = processor(
        text = prompts,
        return_tensors = "pt",
        padding = True,
        truncation = True
    )
    inputs = {k : v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        txtFeatures = model.get_text_features(**inputs)

    return _normalize_embedding_batch(txtFeatures)

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

def appNameGet(appid: int) -> str | None:
    """
    get appname from app_index"""

    row = single_fetch(
        """
        SELECT name
        FROM app_index
        WHERE appid = ?
        """,
        (appid,)
    )

    if not row:
        return None
    
    return row["name"]

def appTxtPrompts(name : str) -> list[str]:
    """
    build clip text prompts for an app name *wip*"""

    return [
        f"gameplay screenshot from {name}",
        f"screenshot from game: {name}",
        f"screenshot via steamstore for {name}",
    ]

def txtScoreAgg(score: list[float]) -> float:
    """
    Aggregate multiple scores for one app.
    use top 2, hopefully makes it so one doesnt dominate"""

    if not score:
        return 0.0
    ranked = sorted(score, reverse = True)#

    if len(ranked) == 1:
        return float(ranked[0])
    
    return float((0.70 * ranked[0]) + (0.30 * ranked[1]))

# >> combining the scores should help with balance out the incorrect matches to some extent <<
def appScoreMultiSS(scores: list [float]) -> float:
    """
    Combine multiple top scores for one app into one final score
    """
    
    ranked = sorted(scores, reverse = True)
    sc1 = ranked[0] if len(ranked) > 0 else 0.0
    sc2 = ranked[1] if len(ranked) > 1 else 0.0
    sc3 = ranked[2] if len(ranked) > 2 else 0.0

    # > these should only be counted if they are close to the best <<
    bn1 = sc2 if sc2 >= (sc1 - 0.03) else 0.0
    bn2 = sc3 if sc3 >= (sc1 - 0.05) else 0.0

    # >> should keep the best ss important; should reward apps with multiple strong matches (??) <<
    return float(sc1 + (0.15 * bn1) + (0.05 * bn2))

def rerankASMulti(matches: list[dict]) -> list[dict]:
    """
    Groups the matfches by appid and rerank using multiple screenshots.
    Best screenshot url is basically used as the representative
    """

    group = {}

    for m in matches:
        appid = int(m["appid"])

        if appid not in group:
            group[appid] = {
                "appid": appid,
                "scores": [],
                "best_url": m["url"],
                "best_score": m["score"],
            }

        group[appid]["scores"].append(float(m["score"]))

        if float(m["score"]) > group[appid]["best_score"]:
            group[appid]["best_score"] = float(m["score"])
            group[appid]["best_url"] = m["url"]

    rerank = []

    for appid, data in group.items():
        appScore = appScoreMultiSS(data["scores"])
        rerank.append({
            "appid": appid,
            "url": data["best_url"],
            "score": data["best_score"],
            "appScore": appScore,
            "match_count": len(data["scores"]),
        })

    rerank.sort(key=lambda x:x["appScore"], reverse = True)
    return rerank


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

def normVec(v: np.ndarray) -> np.ndarray:
    """
    l2 normalise a np vector"""

    dn = np.linalg.norm(v)
    if dn == 0:
        return v
    
    return v / dn

def appidStoredEmbGet(appids: list[int]) -> list[dict]:
    """
    load stores ss embeddings only for requested appid"""
    if not appids:
        return []
    
    ph = ",".join("?" for _ in appids)
    sql = f"""
        SELECT appid, url, embedding,
            COALESCE(dim, CAST(length(embedding) / 4 AS INTEGER)) AS dim
        FROM screenshot_embeddings
        WHERE appid IN ({ph})
    """

    rows = all_fetch(sql, tuple(appids))

    results = []

    for r in rows:
        results.append({
            "appid": int(r["appid"]),
            "url": r["url"],
            "embed": bytesToF32(r["embedding"], int(r["dim"])),
        })

    return results
    
def buildAppCentroids(appids: list[int]) -> dict[int, np.ndarray]:
    """
    to build a normalised centroid embedding per appid from ss embeddings
    """

    rows = appidStoredEmbGet(appids)
    group: dict[int, list[np.ndarray]] = {}

    for r in rows:
        group.setdefault(r["appid"], []).append(r["embed"])

    centroids: dict[int, np.ndarray] = {}

    for appid, embeds in group.items():
        if not embeds:
            continue

        m = np.stack(embeds, axis = 0)
        centroid = m.mean(axis = 0)
        centroids[appid] = normVec(centroid)

    return centroids

def centroidReranker(queryEmb, appMatches: list[dict], sl_k: int = 15) -> list[dict]:
    """
    2nd stage of rerank.
        - Take top app matchees from ss rerank
        - compare query to per app centroid
        - return reranked matches"""
    
    shortlist = appMatches[:sl_k]
    appids = [int(m["appid"]) for m in shortlist]
    centroids = buildAppCentroids(appids)

    rerank = []

    # >>> kw: rScore = RAWSCORE -> best single ss score
    # mssScore = MULTISCREENSHOTSCORE -> multi-screenshot app score
    # crScore = CENTROIDRERANKSCORE -> app centroid similarity <<<
    for m in shortlist:
        appid = int(m["appid"])
        cr = centroids.get(appid)

        rScore = float(m["score"])
        mssScore = float(m.get("appScore", m["score"]))

        if cr is None:
            crScore = float("-inf")
            fScore = rScore
            bm = "raw_fallback"
        else:
            crScore = float(CosSimilarity(queryEmb, cr))

            gap = rScore - crScore
            # >> raw ss = stronger than app centroid <<
            # >> should help with for single frame matches that look exact <<
            if gap >= 0.04:
                fScore = float(
                    (0.60 * rScore) +
                    (0.30 * crScore) +
                    (0.10 * mssScore)
                )
                bm = "raw_heavy"
            else:
                # >> raw + centroid broadly agree <<
                # >> should trust applevel / centroid signal more in this case <<
                fScore = float(
                    (0.40 * rScore) + 
                    (0.45 * crScore) + 
                    (0.15 * mssScore)
                )
                bm = "balanced_centroid"

        row = dict(m)
        row["ssAppScore"] = mssScore
        row["centroidScore"] = crScore
        row["finalScore"] = fScore
        row["appScore"] = fScore
        row["blendMode"] = bm
        rerank.append(row)

    rerank.sort(key = lambda x: x["finalScore"], reverse = True)
    return rerank

def txtPromptRerank(queryEmb, appmatches: list[dict], sl_k: int = 15, bMax: float = 0.04) -> list[dict]:
    """
    third stage of rerank -> uses clip image to text as a bonus
    if anything goes wrong make sure:
    - doesn't replace visual matching
    - should only nudge the closer candidates only"""

    sl = appmatches[:sl_k]

    prompts = []
    owners = []

    for i in sl:
        appid = int(i["appid"])
        name = appNameGet(appid)

        if not name:
            continue

        for pr in appTxtPrompts(name):
            prompts.append(pr)
            owners.append({
                "appid": appid,
                "name": name,
                "prompt": pr,
            })

    # >> return original matches if no prompts are able to be built <<
    if not prompts:
        return appmatches

    txtEmb = embTxtPrompts(prompts)
    scoreByAppid: dict[int, list[float]] = {}

    for owner, txtmb, in zip(owners, txtEmb):
        appid = int(owner["appid"])
        score = CosSimilarity(queryEmb, txtmb)
        scoreByAppid.setdefault(appid, []).append(score)

    txtScores = {appid: txtScoreAgg(scores) for appid, scores in scoreByAppid.items()}

    v = list(txtScores.values())
    minTxt = min(v) if v else 0.0
    maxTxt = max(v) if v else 1.0
    dn = maxTxt - minTxt

    reranked = []

    for m in appmatches:
        appid = int(m["appid"])
        vScore = float(m.get("fScore", m.get("appScore", m.get("score", 0.0))))
        txtScore = txtScores.get(appid)

        # >> txtScoreN makes it so text can be compared within current shortlist <<
        if txtScore is None or dn == 0:
            txtScoreN = 0.0
        else:
            txtScoreN = float((txtScore - minTxt) / dn)

        txtBn = txtScoreN * bMax
        fScore = vScore + txtBn

        row = dict(m)
        row["preTextScore"] = vScore
        row["textScore"] = txtScore
        row["NormalisedTextScore"] = txtScoreN
        row["finalScore"] = fScore
        row["appScore"] = fScore
        row["rerankStage"] = "text_prompt"
        reranked.append(row)

    reranked.sort(key = lambda x: x["finalScore"], reverse = True)
    return reranked

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
