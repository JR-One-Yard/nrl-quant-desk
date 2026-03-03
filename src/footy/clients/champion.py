"""Champion Data API client for NRL fixtures and results."""

from __future__ import annotations

import time
from datetime import datetime

import httpx
import polars as pl

from footy.config import CHAMPION_DATA_BASE_URL, COMPETITION_ID, CD_CACHE_TTL


class ChampionDataClient:
    """Client for the Champion Data NRL fixture API."""

    def __init__(self) -> None:
        self._client = httpx.Client(timeout=15)
        self._cache: dict[str, tuple[float, object]] = {}

    def _get_cached(self, url: str) -> dict:
        now = time.time()
        if url in self._cache:
            ts, data = self._cache[url]
            if now - ts < CD_CACHE_TTL:
                return data
        resp = self._client.get(url)
        resp.raise_for_status()
        data = resp.json()
        self._cache[url] = (now, data)
        return data

    def get_fixture(self) -> pl.DataFrame:
        """Fetch the full season fixture as a polars DataFrame."""
        url = f"{CHAMPION_DATA_BASE_URL}/{COMPETITION_ID}/fixture.json"
        data = self._get_cached(url)
        matches = data.get("fixture", {}).get("match", [])
        if not matches:
            return pl.DataFrame()

        rows = []
        for m in matches:
            rows.append({
                "match_id": m.get("matchId"),
                "round": m.get("roundNumber"),
                "match_number": m.get("matchNumber"),
                "status": m.get("matchStatus", ""),
                "home_team": m.get("homeSquadName", ""),
                "away_team": m.get("awaySquadName", ""),
                "home_score": m.get("homeSquadScore"),
                "away_score": m.get("awaySquadScore"),
                "venue": m.get("venueName", ""),
                "kickoff_utc": m.get("utcStartTime", ""),
                "kickoff_local": m.get("localStartTime", ""),
            })

        return pl.DataFrame(rows).with_columns(
            pl.col("kickoff_utc").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC", strict=False),
            pl.col("kickoff_local").str.to_datetime("%Y-%m-%dT%H:%M:%S%z", strict=False),
        )

    def get_current_round(self) -> int:
        """Return the earliest round that still has a scheduled match."""
        df = self.get_fixture()
        if df.is_empty():
            return 1
        scheduled = df.filter(pl.col("status") == "scheduled")
        if scheduled.is_empty():
            return df["round"].max()
        return scheduled["round"].min()

    def get_round_matches(self, round_num: int) -> pl.DataFrame:
        """Return matches for a specific round."""
        df = self.get_fixture()
        return df.filter(pl.col("round") == round_num)

    def get_completed_matches(self) -> pl.DataFrame:
        """Return all completed matches, sorted chronologically."""
        df = self.get_fixture()
        return df.filter(pl.col("status") == "complete").sort("kickoff_utc")

    def invalidate_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
