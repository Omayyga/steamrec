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





