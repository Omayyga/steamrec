import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

db_path = Path("db.sqlite3")

def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection

def init_db() -> None:
    connection = get_connection()
    cursor = connection.cursor()
    
    ## Create users table if nonexistant; ##
    ## stores steamid64 and account creation timestamp. ##
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            steamid64 TEXT PRIMARY KEY,
            created_at INTEGER NOT NULL
        );
        """
    )

    ## Owned games table. steamid64 stored with game data of each owned game. ##
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS owned_games (
            steamid64 TEXT NOT NULL,
            appid INTEGER NOT NULL,
            name TEXT,
            playtime_forever_min INTEGER NOT NULL,
            last_synced INTEGER NOT NULL,
            PRIMARY KEY (steamid64, appid)
        );
        """
    )

    ## cached store metadeta. ##
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS app_details (
            appid INTEGER PRIMARY KEY,
            json TEXT NOT NULL,
            fetched_at INTEGER NOT NULL
        );
        """
    )

    connection.commit()
    connection.close()

    ## gets current unix timestamp (s). ##
    def timestamp() -> int:
        return int(time.time())

def exec(sql: str, params: Iterable[Any] = []) -> None:
    """
    Exexcutes an SQL statement;
    does not return rows (i.e. INSERT, UPDATE, DELETE).
    
    Use case:
        Central helper so writes are consistent.
    """
    connection = get_connection()
    connection.execute(sql, tuple(params))
    connection.commit()
    connection.close()