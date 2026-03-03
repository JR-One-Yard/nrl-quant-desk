"""The Odds API client for NRL bookmaker odds."""

from __future__ import annotations

import time

import httpx
import polars as pl

from footy.config import (
    ODDS_API_BASE_URL,
    ODDS_SPORT,
    ODDS_REGIONS,
    ODDS_MARKETS,
    THE_ODDS_API_KEY,
    ODDS_CACHE_TTL,
    normalize_team,
)


class OddsAPIClient:
    """Client for The Odds API — NRL head-to-head odds."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or THE_ODDS_API_KEY
        self._client = httpx.Client(timeout=15)
        self._cache: dict[str, tuple[float, object]] = {}
        self._remaining_quota: int | None = None
        self._last_fetched: float | None = None

    def _get_cached(self, url: str, params: dict) -> tuple[dict | list, dict]:
        """Fetch with cache. Returns (data, headers)."""
        cache_key = url
        now = time.time()
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if now - ts < ODDS_CACHE_TTL:
                return data, {}
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        self._cache[cache_key] = (now, data)
        self._last_fetched = now
        # Track quota
        remaining = resp.headers.get("x-requests-remaining")
        if remaining is not None:
            self._remaining_quota = int(remaining)
        return data, dict(resp.headers)

    def get_h2h_odds(self) -> pl.DataFrame:
        """Fetch head-to-head odds for NRL, returning a polars DataFrame."""
        if not self._api_key:
            return pl.DataFrame()

        url = f"{ODDS_API_BASE_URL}/sports/{ODDS_SPORT}/odds/"
        params = {
            "apiKey": self._api_key,
            "regions": ODDS_REGIONS,
            "markets": ODDS_MARKETS,
            "oddsFormat": "decimal",
        }
        events, _ = self._get_cached(url, params)
        if not events:
            return pl.DataFrame()

        rows = []
        for event in events:
            event_id = event.get("id", "")
            home_team = normalize_team(event.get("home_team", ""))
            away_team = normalize_team(event.get("away_team", ""))
            commence = event.get("commence_time", "")

            for bookmaker in event.get("bookmakers", []):
                bookie_name = bookmaker.get("title", "")
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                    # Map outcomes to home/away using original event names
                    raw_home = event.get("home_team", "")
                    raw_away = event.get("away_team", "")
                    home_odds = outcomes.get(raw_home)
                    away_odds = outcomes.get(raw_away)
                    if home_odds and away_odds:
                        rows.append({
                            "event_id": event_id,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": bookie_name,
                            "home_odds": float(home_odds),
                            "away_odds": float(away_odds),
                            "commence_time": commence,
                        })

        if not rows:
            return pl.DataFrame()

        return pl.DataFrame(rows).with_columns(
            pl.col("commence_time").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC", strict=False),
        )

    def get_remaining_quota(self) -> int | None:
        """Return the remaining API request quota, or None if unknown."""
        return self._remaining_quota

    def get_cache_age_seconds(self) -> float | None:
        """Seconds since last live fetch, or None if never fetched."""
        if self._last_fetched is None:
            return None
        return time.time() - self._last_fetched

    def invalidate_cache(self) -> None:
        """Clear cache to force a fresh fetch on next call."""
        self._cache.clear()
