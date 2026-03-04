"""Elo rating system with margin-of-victory adjustment for NRL."""

from __future__ import annotations

import math

import polars as pl

from footy.config import NRL_TEAMS, ELO_INITIAL, ELO_K, ELO_HOME_ADVANTAGE


class EloModel:
    """Elo ratings for NRL teams with MOV (margin of victory) adjustment."""

    def __init__(
        self,
        k: float = ELO_K,
        home_advantage: float = ELO_HOME_ADVANTAGE,
        initial_rating: float = ELO_INITIAL,
        priors: dict[str, float] | None = None,
    ) -> None:
        self.k = k
        self.home_advantage = home_advantage
        # Seed each team: use prior if provided, else fall back to initial_rating
        self.ratings: dict[str, float] = {
            t: (priors[t] if priors and t in priors else initial_rating)
            for t in NRL_TEAMS
        }

    def _expected(self, rating_a: float, rating_b: float) -> float:
        """Expected score for team A against team B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _mov_multiplier(self, margin: int, rating_diff: float) -> float:
        """Margin-of-victory multiplier (FiveThirtyEight-style)."""
        return math.log(abs(margin) + 1) * 2.2 / (rating_diff * 0.001 + 2.2)

    def update(
        self, home_team: str, away_team: str, home_score: int, away_score: int
    ) -> None:
        """Update ratings after a completed match."""
        home_r = self.ratings.get(home_team, ELO_INITIAL)
        away_r = self.ratings.get(away_team, ELO_INITIAL)

        # Home advantage baked into expected calculation
        expected_home = self._expected(home_r + self.home_advantage, away_r)

        margin = home_score - away_score
        if margin > 0:
            actual_home = 1.0
        elif margin < 0:
            actual_home = 0.0
        else:
            actual_home = 0.5

        rating_diff = abs(home_r - away_r)
        mov = self._mov_multiplier(margin, rating_diff)

        delta = self.k * mov * (actual_home - expected_home)
        self.ratings[home_team] = home_r + delta
        self.ratings[away_team] = away_r - delta

    def predict(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Predict win probabilities. Returns (home_prob, away_prob)."""
        home_r = self.ratings.get(home_team, ELO_INITIAL)
        away_r = self.ratings.get(away_team, ELO_INITIAL)
        home_prob = self._expected(home_r + self.home_advantage, away_r)
        return home_prob, 1.0 - home_prob

    def bootstrap_from_results(self, results_df: pl.DataFrame) -> None:
        """Process completed matches chronologically to build ratings.

        Expects columns: home_team, away_team, home_score, away_score, kickoff_utc
        """
        if results_df.is_empty():
            return
        sorted_df = results_df.sort("kickoff_utc")
        for row in sorted_df.iter_rows(named=True):
            home_score = row.get("home_score")
            away_score = row.get("away_score")
            if home_score is None or away_score is None:
                continue
            self.update(
                row["home_team"],
                row["away_team"],
                int(home_score),
                int(away_score),
            )

    def mean_revert(self, reversion_factor: float = 0.3) -> dict[str, float]:
        """Return mean-reverted ratings suitable for seeding the next season.

        Applies the FiveThirtyEight carry-over approach:
            carried = (ELO_INITIAL * reversion_factor) + (current * (1 - reversion_factor))

        A reversion_factor of 0.3 pulls each team 30% of the way back toward
        the league average (1500) before the new season begins, preventing
        historical dominance from compounding indefinitely.

        Returns the reverted ratings dict without mutating self.ratings.
        """
        return {
            team: ELO_INITIAL * reversion_factor + rating * (1.0 - reversion_factor)
            for team, rating in self.ratings.items()
        }

    def get_ratings_df(self) -> pl.DataFrame:
        """Return ratings as a polars DataFrame sorted by rating descending."""
        rows = [{"team": t, "elo_rating": round(r, 1)} for t, r in self.ratings.items()]
        return pl.DataFrame(rows).sort("elo_rating", descending=True)
