"""Settings, API URLs, team name mapping."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
try:
    DATA_DIR.mkdir(exist_ok=True)
    # Test writability (Streamlit Cloud repo dir is read-only)
    _test = DATA_DIR / ".write_test"
    _test.touch()
    _test.unlink()
except OSError:
    DATA_DIR = Path("/tmp/footy_data")
    DATA_DIR.mkdir(exist_ok=True)
DUCKDB_PATH = DATA_DIR / "footy.duckdb"

# --- API Keys ---
# Reads from .env locally, falls back to Streamlit secrets on Community Cloud
def _get_secret(key: str) -> str:
    val = os.getenv(key, "")
    if not val:
        try:
            import streamlit as st
            val = st.secrets.get(key, "")
        except Exception:
            pass
    return val

THE_ODDS_API_KEY = _get_secret("THE_ODDS_API_KEY")
ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")

# --- Champion Data ---
CHAMPION_DATA_BASE_URL = "https://mc.championdata.com/data"
COMPETITION_ID = 12999

# --- The Odds API ---
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "rugbyleague_nrl"
ODDS_REGIONS = "au"
ODDS_MARKETS = "h2h"

# --- Cache TTLs (seconds) ---
CD_CACHE_TTL = 300       # 5 min for Champion Data
ODDS_CACHE_TTL = 3600    # 1 hour for Odds API

# --- Elo Defaults ---
ELO_INITIAL = 1500
ELO_K = 32
ELO_HOME_ADVANTAGE = 50

# --- All 17 NRL Teams ---
NRL_TEAMS = [
    "Brisbane Broncos",
    "Canberra Raiders",
    "Canterbury-Bankstown Bulldogs",
    "Cronulla-Sutherland Sharks",
    "Dolphins",
    "Gold Coast Titans",
    "Manly-Warringah Sea Eagles",
    "Melbourne Storm",
    "Newcastle Knights",
    "North Queensland Cowboys",
    "Parramatta Eels",
    "Penrith Panthers",
    "South Sydney Rabbitohs",
    "St George-Illawarra Dragons",
    "Sydney Roosters",
    "Warriors",
    "Wests Tigers",
]

# --- Team Name Normalization ---
# Maps various external names → canonical Champion Data name
_TEAM_ALIASES: dict[str, str] = {
    # Odds API variants
    "Canterbury Bulldogs": "Canterbury-Bankstown Bulldogs",
    "Bulldogs": "Canterbury-Bankstown Bulldogs",
    "Canterbury-Bankstown Bulldogs": "Canterbury-Bankstown Bulldogs",
    "Cronulla Sharks": "Cronulla-Sutherland Sharks",
    "Sharks": "Cronulla-Sutherland Sharks",
    "Cronulla-Sutherland Sharks": "Cronulla-Sutherland Sharks",
    "Gold Coast Titans": "Gold Coast Titans",
    "Titans": "Gold Coast Titans",
    "Manly Sea Eagles": "Manly-Warringah Sea Eagles",
    "Manly Warringah Sea Eagles": "Manly-Warringah Sea Eagles",
    "Manly-Warringah Sea Eagles": "Manly-Warringah Sea Eagles",
    "Sea Eagles": "Manly-Warringah Sea Eagles",
    "Melbourne Storm": "Melbourne Storm",
    "Storm": "Melbourne Storm",
    "Newcastle Knights": "Newcastle Knights",
    "Knights": "Newcastle Knights",
    "North Queensland Cowboys": "North Queensland Cowboys",
    "Cowboys": "North Queensland Cowboys",
    "Parramatta Eels": "Parramatta Eels",
    "Eels": "Parramatta Eels",
    "Penrith Panthers": "Penrith Panthers",
    "Panthers": "Penrith Panthers",
    "South Sydney Rabbitohs": "South Sydney Rabbitohs",
    "Rabbitohs": "South Sydney Rabbitohs",
    "St George Illawarra Dragons": "St George-Illawarra Dragons",
    "St George-Illawarra Dragons": "St George-Illawarra Dragons",
    "Dragons": "St George-Illawarra Dragons",
    "Sydney Roosters": "Sydney Roosters",
    "Roosters": "Sydney Roosters",
    "Brisbane Broncos": "Brisbane Broncos",
    "Broncos": "Brisbane Broncos",
    "Canberra Raiders": "Canberra Raiders",
    "Raiders": "Canberra Raiders",
    "Dolphins": "Dolphins",
    "Redcliffe Dolphins": "Dolphins",
    "New Zealand Warriors": "Warriors",
    "NZ Warriors": "Warriors",
    "Warriors": "Warriors",
    "Wests Tigers": "Wests Tigers",
    "West Tigers": "Wests Tigers",
    "Tigers": "Wests Tigers",
}

# Build case-insensitive lookup
_ALIAS_LOWER = {k.lower(): v for k, v in _TEAM_ALIASES.items()}


def normalize_team(name: str) -> str:
    """Normalize a team name to the canonical Champion Data form."""
    cleaned = name.strip()
    result = _ALIAS_LOWER.get(cleaned.lower())
    if result:
        return result
    # Fuzzy fallback: check if any alias is contained in the name
    for alias, canonical in _ALIAS_LOWER.items():
        if alias in cleaned.lower() or cleaned.lower() in alias:
            return canonical
    return cleaned  # Return as-is if no match
