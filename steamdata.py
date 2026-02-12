import json
import os
import httpx

from db import single_fetch, exec, timestamp

steam_api_key = os.getenv("STEAM_API_KEY", "")

## >>> fetch info from steam api; returned as list of dicts. <<<
async def f_owned(steamid64: str) -> list[dict]:

    url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"

    params = {
    "api-key": steam_api_key,
    "steamid": steamid64,
    "inc_appinfo": 1,
    "inc_played_free_games": 1,
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
        "application ids": str(appid),
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
    return data