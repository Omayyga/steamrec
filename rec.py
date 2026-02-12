import math

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
        SELECT appid, pt_forever_min FROM owned_games
        FROM owned_games
        WHERE steamid64 = ?
        ORDER BY pt_forever_min DESC
        LIMIT ?;
        """,
        [steamid64, TopGames_n]
    )

    # >> inc for statement for rows tmr

    # >> Small scale scoring system?