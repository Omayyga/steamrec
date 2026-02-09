import os
import re
import httpx

from urllib.parse import urlencode
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from itsdangerous import URLSafeSerializer

load_dotenv()

steam_api_key = os.getenv("STEAM_API_KEY", "")
base_url = os.getenv("BASE_URL", "http://127.0.0.1:8000")
session_secret = os.getenv("SESSION_SECRET", "dev-secret")

if not steam_api_key:
    print("Warning: No API key provided. limited functionality.")


safe_serializer = URLSafeSerializer(session_secret, salt="steamrec-session")

steam_openid_url = "https://steamcommunity.com/openid/login"

## openid 2.0 parameters for steam login. ##
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

## openid response verification; posts back to steam to verify authenticity. ##
async def verify_openid_response(query_params: dict) -> str:
    
    data = dict(query_params)
    data["openid.mode"] = "check_authentication"

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(steam_openid_url, data=data)
        return "is_valid:true" in response.text

## steamid64 extraction process. ## 
def steamid64_extract(steam_id: str) -> str | None:
   id = re.search(r"^https?://steamcommunity\.com/openid/id/(\d+)$", steam_id)
   return id.group(1) if id else None

def set_session_cookie(response: RedirectResponse, steamid64: str) -> None:
    tk = serializer.dumps({"steamid64": steamid64})
    response.set_cookie("session", tk, httponly=True, secure=True, samesite="lax", max_age=60*60*24*7)

def get_session_steamid64(request: Request) -> str | None:
    tk = request.cookies.get("session")
    if not tk:
        return None
    try:
        data = serializer.loads(tk)
        return data.get("steamid64")
    except Exception:
        return None
    