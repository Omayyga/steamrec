import json
import os
from dotenv import load_dotenv
import httpx

from db import single_fetch, exec, timestamp

load_dotenv()

steam_api_key = os.getenv("STEAM_API_KEY", "")

## >>> fetch info from steam api; returned as list of dicts. <<<
async def f_owned(steamid64: str) -> list[dict]:

    url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"

    params = {
    "key": steam_api_key,
    "steamid": steamid64,
    "include_appinfo": 1,
    "include_played_free_games": 1,
    "format": "json",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    return data.get("response", {}).get("games", []) or []

# >>> calls steam store "appdetails" endpoint;
# returns metadata JSON. <<<
async def f_appdetails_store(appid: int) -> dict | None:

    url = "https://store.steampowered.com/api/appdetails"
    params = {
        "appids": str(appid),
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        appdata = response.json()

    app_entry = appdata.get(str(appid), {})

    if not app_entry.get("success"):
        return None
    
    return app_entry.get("data")

# >>> basically ensures data is up to date. <<<
async def f_appdetails_cached(appid: int, ttl_seconds: int = 60 * 60 * 24 * 7) -> dict | None: # >>> ttl_seconds -> cached data lifespan <<<

    r = single_fetch("SELECT json, fetched_at FROM app_details WHERE appid = ?", [appid])
    current_time = timestamp()

    if r:
        data_age = current_time - r["fetched_at"]
        if data_age < ttl_seconds:
            return json.loads(r["json"])

    data = await f_appdetails_store(appid)
    if data is None:
        return None
    
    exec(
        """
        INSERT INTO app_details (appid, json, fetched_at)
        VALUES (?, ?, ?)
        ON CONFLICT(appid) DO UPDATE SET
            json = excluded.json,
            fetched_at = excluded.fetched_at
        """,
        [appid, json.dumps(data), current_time]
    )
    
    UpsertAppIndex(appid, data)
    return data

# >>> Extraction Helpers. <<<
def ExtGenres(appdetails: dict) -> list[str]:
    genres = appdetails.get("genres") or []
    return [gen.get ("description") for gen in genres if gen.get ("description")]

def ExtCat(appdetails: dict) -> list[str]:
    categories = appdetails.get("categories") or []
    return [cat.get("description") for cat in categories if cat.get("description")]

# >>> Upsert for App index.<<<
def UpsertAppIndex(appid: int, appdetails: dict) -> None:
    name = appdetails.get("Name")
    genres = ExtGenres(appdetails)
    categories = ExtCat(appdetails)
    timestamp = timestamp()

    exec(
        """
        INSERT INTO app_index(appid, name, genres, categories, updated_at)
        VALUES(?,?,?,?,?)
        ON CONFLICT(appid) DO UPDATE SET
            name = excluded.name
            genres = excluded.genres
            categories = excluded.genres
            updated_at = excluded.updated_at
        """,
        (
            appid,
            name,
            json.dumps(genres),
            json.dumps(categories),
            timestamp
        )
    )