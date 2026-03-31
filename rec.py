import math
import json
import random

from db import all_fetch, single_fetch

from collections import Counter
from steamdata import f_appdetails_cached

# >>> extracts game genres. <<<
def ext_genre(appdetails: dict) -> list[str]:
    genre = appdetails.get("genres", [])
    return [i.get("description") for i in genre if i.get("description")]

async def BuildUserProfile_genre(steamid64: str, TopGames_n: int = 50) -> Counter:
    """
    Genre counter for users top played games .
    TopGames_n -> limited to 50; Filter out noise 
    (should inc -> Unplayed games? Short playtime?)
    """

    rows = all_fetch(
        """
        SELECT appid, pt_forever_min
        FROM owned_games
        WHERE steamid64 = ?
        ORDER BY pt_forever_min DESC
        LIMIT ?
        """,
        (steamid64, TopGames_n)
    )

    profile = Counter()

    for row in rows:
        appid = row["appid"]
        playtime = row["pt_forever_min"]
        weight = math.log1p(playtime) # >> weighting system; should lessen burden of outliers? keep eye on -> may need tweaking.

        if playtime < 30: # >>> filter out games with less than 30 minutes playtime <<<
            continue

        appdetails = await f_appdetails_cached(appid)
        if not appdetails:
            continue

        for genre in ext_genre(appdetails):
            profile[genre] += weight

            return profile
        
async def BuildUserProfile_cat(steamid64: str, TopGames_n: int = 50) -> Counter:
    """
    Another preference profile based on categories.
    """
    rows = all_fetch(
        """
        SELECT appid, pt_forever_min
        FROM owned_games
        WHERE steamid64 = ?
        ORDER BY pt_forever_min DESC
        LIMIT ?
        """,
        (steamid64, TopGames_n)
    )

    profile = Counter()

    for r in rows:
        appid = int(r["appid"])
        mins = int(r["pt_forever_min"])
        weight = math.log1p(mins)

        details = await f_appdetails_cached(appid)
        if not details:
            continue

        cat = details.get("categories", [])
        catNames = [c.get("description") for c in cat if c.get("description")]

        for c in catNames:
            profile[c] += weight

        return profile
    
def topMatch(itemlist: list[str], profile: Counter, n: int = 3) -> list[str]:
    rank = sorted(itemlist, key = lambda x: profile.get(x, 0), reverse = True)
    return [x for x in rank[:n] if profile.get(x, 0) > 0]

async def GameScoring(appid: int, genreProfile: Counter, catProfile: Counter) -> tuple[float, list[str]]:
    """
    Scores based on genre and category overlap.
    Should be more than just genres (??)
    """

    appdetails = await f_appdetails_cached(appid)
    if not appdetails:
        return 0.0, []

    genres = [g.get ("description") for g in appdetails.get("genres") or [] if g.get("description")]
    genreScore = sum(genreProfile.get(g, 0) 
                for g in genres)
    
    cat = [c.get("description") for c in appdetails.get("categories") or [] if c.get("description")]
    catScore = sum(catProfile.get(c, 0) for c in cat)

    score = float(genreScore + 0.35 * catScore) # >> !!!! reminder to finetune starter weight.. <<<

    # >> Outcome reasons; top three contributors.
    outcomeReasons = []
    topGenre = topMatch (genres, genreProfile, 2)
    topCat = topMatch (cat, catProfile, 2)

    if topGenre:
        outcomeReasons.append(f"Genre match: {', '.join(topGenre)}")
    if topCat:
        outcomeReasons.append(f"Category match: {', '.join(topCat)}")

    return score, outcomeReasons

# >>> generates candidate appids; should be based on profiles top genres? <<<
def TopProfileGenres_get(profile, i = 3):
    return [gen for gen, _ in profile.most_common(i)]

def GenCandidates(profile, limit = 300, explore = 150): # >>> explore -> random sample; should add some diversity <<<
    TopGenres = TopProfileGenres_get(profile)

    if not TopGenres:
        return []

    rows = all_fetch("SELECT appid, genres FROM app_index")
    MatchedGenres, OtherGenres = [], []

    for r in rows:
        appid = int(r["appid"])
        genres = json.loads(r["genres"] or "[]")

        if TopGenres and any(gen in genres for gen in TopGenres):
            MatchedGenres.append((appid))
        else:
            OtherGenres.append((appid))

    if len(OtherGenres) > explore :
        OtherGenres = random.sample(OtherGenres, explore)

    candidates = MatchedGenres + OtherGenres
    random.shuffle(candidates)
    
    return candidates[:limit]

def indexinfoGet(appid: int) -> dict:
    """
    Get basic info from app_index for a given appid.
    """
    row = single_fetch(
        """
        SELECT appid, name, genres, categories
        FROM app_index
        WHERE appid = ?
        """,
        (appid,)
    )

    if not row:
        return None
    
    return {
        "appid": int(row["appid"]),
        "name": row["name"],
        "genres": json.loads(row["genres"] or "[]"),
        "categories": json.loads(row["categories"] or "[]")
    }

async def ScoreGame(appid: int, steamid64: str) -> dict:
    genreProfile = await BuildUserProfile_genre(steamid64)
    catProfile = await BuildUserProfile_cat(steamid64)

    score, reasons = await GameScoring(appid, genreProfile, catProfile)
    appinfo = indexinfoGet(appid)

    return {
        "appid": appid,
        "name": appinfo["name"],
        "score": score,
        "reasons": reasons,
        "genres": appinfo["genres"],
        "categories": appinfo["categories"]
    }

async def ScoreGameMulti(appids: list[int], steamid64: str) -> list[dict]:
    genreProfile = await BuildUserProfile_genre(steamid64)
    catProfile = await BuildUserProfile_cat(steamid64)

    results = []

    for appid in appids:
        score, reasons = await GameScoring(appid, genreProfile, catProfile)
        appinfo = indexinfoGet(appid)

        results.append({
            "appid": appid,
            "name": appinfo["name"] if appinfo else None,
            "score": score,
            "reasons": reasons,
            "genres": appinfo["genres"] if appinfo else [],
            "categories": appinfo["categories"] if appinfo else []
        })

    return results