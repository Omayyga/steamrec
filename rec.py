import math
import json
import random

from collections import Counter
from db import all_fetch
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

async def GameScoring_genre(appid: int, user_profile: Counter) -> tuple[float, list[str]]:
    """
    Sums up weights for each genre; 
    based of users profile.
    
    """

    appdetails = await f_appdetails_cached(appid)
    if not appdetails:
        return 0.0, []

    genres = ext_genre(appdetails)
    score = sum(user_profile.get(g, 0) 
                for g in genres)

    # >> Outcome reasons; top three contributors.
    OC_reasons = sorted(genres, key=lambda g: user_profile.get(g, 0.0), reverse = True)[:3]
    OC_reasons = [g for g in OC_reasons if user_profile.get(g, 0.0) > 0]

    return float(score), OC_reasons

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