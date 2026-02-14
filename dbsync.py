from db import exec, timestamp
from steamdata import f_owned

async def dbsync_owned(steamid64: str) -> dict:
    """
    fetch owned games via steam api;
    store in sqlite

    """
    ts = timestamp()

    exec(
        """
    INSERT INTO users(steamid64, created_at)
    VALUES (?, ?)
    ON CONFLICT(steamid64) DO NOTHING;
        """,
        [steamid64, ts]
    )

    games = await f_owned(steamid64)

    for i in games:
        exec(
            """
        INSERT INTO owned_games (steamid64, appid, name, pt_forever_min, last_synced)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(steamid64, appid) DO UPDATE SET
        name = excluded.name,
        pt_forever_min = excluded.pt_forever_min,
        last_synced = excluded.last_synced
            """,
        (
            steamid64, 
             int(i.get("appid")), 
             i.get("name"), 
             int(i.get("playtime_forever", 0)), 
             ts,
        ),)

    return {"steamid64": steamid64, "synced-games": len(games), "last-synced": ts}