import os
import re
import httpx

from urllib.parse import urlencode
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from itsdangerous import URLSafeSerializer

load_dotenv()

# >>> obtains configuration from environment variables. <<<
steam_api_key = os.getenv("STEAM_API_KEY", "")
base_url = os.getenv("BASE_URL", "http://127.0.0.1:8000")
session_secret = os.getenv("SESSION_SECRET", "dev-secret")

serializer = URLSafeSerializer(session_secret, salt="steamrec-session")

# >>> if no API key provided, user is warned, required to fetch owned games via steam api <<<
if not steam_api_key:
    print("Warning: No API key provided. limited functionality.")


safe_serializer = URLSafeSerializer(session_secret, salt="steamrec-session")

steam_openid_url = "https://steamcommunity.com/openid/login"

# >>> openid 2.0 parameters for steam login. <<<
def openid_login_url() -> str:
    
    params = {
        "openid.ns": "http://specs.openid.net/auth/2.0",
        "openid.mode": "checkid_setup",
        "openid.return_to": f"{base_url}/auth/steam/callback",
        "openid.realm": base_url,
        "openid.identity": "http://specs.openid.net/auth/2.0/identifier_select",
        "openid.claimed_id": "http://specs.openid.net/auth/2.0/identifier_select"
    }

    return f"{steam_openid_url}?{urlencode(params)}"

# >>> openid response verification; posts back to steam to verify authenticity. <<<
async def openid_verify(query_params: dict) -> str:
    
    data = dict(query_params)
    data["openid.mode"] = "check_authentication"

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(steam_openid_url, data=data)
        return "is_valid:true" in response.text

# >>> steamid64 extraction process. <<< 
def sid64_extract(steam_id: str) -> str | None:
   id = re.search(r"^https?://steamcommunity\.com/openid/id/(\d+)$", steam_id)
   return id.group(1) if id else None


# >>> SECURE = FALSE -> DUE TO LOCAL DEVELOPMENT. <<<
def set_session_cookie(response: RedirectResponse, steamid64: str) -> None:
    tk = serializer.dumps({"steamid64": steamid64})
    response.set_cookie("session", tk, httponly=True, secure=False, samesite="lax", max_age=60*60*24*7)

def get_session_steamid64(request: Request) -> str | None:
    tk = request.cookies.get("session")
    if not tk:
        return None
    try:
        data = serializer.loads(tk)
        return data.get("steamid64")
    except Exception:
        return None

app = FastAPI()

# >>> login route; if user logged in, shows steamid64 as well a owned games and logout links. 
#  if not logged in, shows login link. <<<
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    steamid64 = get_session_steamid64(request)
    if steamid64:
        return f"""<h1>Log in successful.</h1>
        <p>Your SteamID64: <code>{steamid64}</code></p>
        <ul>
            <li><a href="/me">/me</a></li>
            <li><a href="/me/owned-games">/me/owned-games</a></li>
            <li><a href="/logout">/logout</a></li>
        </ul>
        """
    
    return """<h1>Welcome to [NAME]!</h1>
    <p><a href="/login">Log in with Steam</a></p
    """

@app.get("/login")
def login():
    return RedirectResponse(openid_login_url(), status_code=302)

# >>> callback route for steam login. 
# runs and extracts query parameters so they can be validated and processed. <<<
@app.get("/auth/steam/callback")
async def steam_auth_callback(request: Request):
    query_params = dict(request.query_params)

# >>> validates steam's login response; 
# extracts steamid64. <<<
    claimed_id = query_params.get("openid.claimed_id")
    if not claimed_id:
        return JSONResponse({"error": "Missing openid.claimed_id"}, status_code=400)
    
    ok = await openid_verify(query_params)
    if not ok:
        return JSONResponse({"error": "OpenID authentication failed"}, status_code=400)
    
    steamid64 = sid64_extract(claimed_id)
    if not steamid64:
        return JSONResponse({"error": "Failed to extract steamid64"}, status_code=400)
    
    response = RedirectResponse("/", status_code=302)
    set_session_cookie(response, steamid64)
    return response

# >>> logout route. <<<
@app.get("/logout")
def logout():
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie("session")
    return response

# >>> returns steamid64 of logged in user. 
# failure responds with 401 unauthorized. <<<
@app.get("/me")
def me(request: Request):
    steamid64 = get_session_steamid64(request)
    if not steamid64:
        return JSONResponse({"error": "User not logged in."}, status_code=401)
    
    return {"steamid64": steamid64}

# >>> + owned games list via steam api. 
# 401 = unauthorised; 500 = missing API key. <<<
@app.get("/me/owned-games")
async def owned_games(request: Request):
    steamid64 = get_session_steamid64(request)
    if not steamid64:
        return JSONResponse({"error": "User is not logged in."}, status_code=401)
    
    if not steam_api_key:
        return JSONResponse({"error": "User did not provide Steam API key."}, status_code=500)
    url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001"
    params = {
        "key": steam_api_key,
        "steamid": steamid64,
        "format": "json",
        "include_appinfo": 1,
        "include_played_free_games": 1,
        "format": "json",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

# >>> returns trimmed view, of users top owned games. sorted by playtime. <<<
    games = data.get("response", {}).get("games", []) or []
    sorted_games = sorted(games, key = lambda g: g.get("playtime_forever", 0), reverse=True)
    top_owned = [
        {
            "appid": i.get("appid"),
            "name": i.get("name"),
            "pt_hours_min":i.get("playtime_forever", 0)
        }
        for i in sorted_games[:30] 
    ]
    return {"steamid64": steamid64, "top_games": top_owned, "total_games": len(games)}