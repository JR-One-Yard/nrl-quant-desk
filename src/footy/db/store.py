"""DuckDB storage for odds snapshots, match results, and Elo state."""

from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import polars as pl

from footy.config import DUCKDB_PATH


class FootyStore:
    """Persistent storage backed by DuckDB."""

    def __init__(self, db_path: str | None = None) -> None:
        self._path = db_path or str(DUCKDB_PATH)
        self._con = duckdb.connect(self._path)
        self._init_tables()

    def _init_tables(self) -> None:
        self._con.execute("CREATE SEQUENCE IF NOT EXISTS odds_seq START 1")
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                snapshot_id INTEGER PRIMARY KEY DEFAULT nextval('odds_seq'),
                captured_at TIMESTAMPTZ NOT NULL,
                event_id VARCHAR,
                home_team VARCHAR NOT NULL,
                away_team VARCHAR NOT NULL,
                bookmaker VARCHAR NOT NULL,
                home_odds DOUBLE NOT NULL,
                away_odds DOUBLE NOT NULL
            )
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS match_results (
                match_id INTEGER PRIMARY KEY,
                round INTEGER NOT NULL,
                home_team VARCHAR NOT NULL,
                away_team VARCHAR NOT NULL,
                home_score INTEGER,
                away_score INTEGER,
                venue VARCHAR,
                kickoff_utc TIMESTAMPTZ,
                status VARCHAR
            )
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS elo_ratings (
                team VARCHAR PRIMARY KEY,
                rating DOUBLE NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
        """)

    def save_odds_snapshot(self, odds_df: pl.DataFrame) -> int:
        """Save odds snapshot. Only inserts if odds have changed from last snapshot.

        Returns number of rows inserted.
        """
        if odds_df.is_empty():
            return 0

        # Check last snapshot for each match/bookmaker
        last = self._con.execute("""
            SELECT home_team, away_team, bookmaker, home_odds, away_odds
            FROM odds_snapshots
            WHERE captured_at = (SELECT MAX(captured_at) FROM odds_snapshots)
        """).pl()

        now = datetime.now(timezone.utc)

        if not last.is_empty():
            # Join to find changed odds
            compare = odds_df.join(
                last,
                on=["home_team", "away_team", "bookmaker"],
                how="left",
                suffix="_prev",
            )
            changed = compare.filter(
                (pl.col("home_odds") != pl.col("home_odds_prev"))
                | (pl.col("away_odds") != pl.col("away_odds_prev"))
                | pl.col("home_odds_prev").is_null()
            )
            if changed.is_empty():
                return 0
            to_insert = changed.select(odds_df.columns)
        else:
            to_insert = odds_df

        insert_df = to_insert.with_columns(
            pl.lit(now).alias("captured_at"),
        ).select("captured_at", "event_id", "home_team", "away_team", "bookmaker", "home_odds", "away_odds")

        self._con.execute("INSERT INTO odds_snapshots (captured_at, event_id, home_team, away_team, bookmaker, home_odds, away_odds) SELECT * FROM insert_df")
        return len(insert_df)

    def save_match_results(self, fixture_df: pl.DataFrame) -> None:
        """Upsert match results from Champion Data fixture."""
        if fixture_df.is_empty():
            return
        completed = fixture_df.filter(pl.col("status") == "complete")
        if completed.is_empty():
            return

        for row in completed.iter_rows(named=True):
            self._con.execute("""
                INSERT OR REPLACE INTO match_results
                (match_id, round, home_team, away_team, home_score, away_score, venue, kickoff_utc, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                row["match_id"], row["round"], row["home_team"], row["away_team"],
                row.get("home_score"), row.get("away_score"), row.get("venue"),
                row.get("kickoff_utc"), row["status"],
            ])

    def get_completed_results(self) -> pl.DataFrame:
        """Retrieve all completed match results."""
        return self._con.execute(
            "SELECT * FROM match_results WHERE status = 'complete' ORDER BY kickoff_utc"
        ).pl()

    def save_elo_ratings(self, ratings: dict[str, float]) -> None:
        """Save current Elo ratings."""
        now = datetime.now(timezone.utc)
        for team, rating in ratings.items():
            self._con.execute("""
                INSERT OR REPLACE INTO elo_ratings (team, rating, updated_at)
                VALUES (?, ?, ?)
            """, [team, rating, now])

    def get_latest_elo_ratings(self) -> dict[str, float]:
        """Load the latest saved Elo ratings."""
        df = self._con.execute("SELECT team, rating FROM elo_ratings").pl()
        if df.is_empty():
            return {}
        return dict(zip(df["team"].to_list(), df["rating"].to_list()))

    def close(self) -> None:
        self._con.close()
