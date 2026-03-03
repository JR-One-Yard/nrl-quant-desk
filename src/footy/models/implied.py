"""Market-implied probabilities, consensus, and overround calculations."""

from __future__ import annotations

import polars as pl


def decimal_odds_to_implied_prob(
    home_odds: float, away_odds: float
) -> tuple[float, float, float]:
    """Convert decimal odds to implied probabilities and overround.

    Returns (home_prob, away_prob, overround_pct).
    Probabilities are normalized to sum to 1.
    """
    raw_home = 1.0 / home_odds
    raw_away = 1.0 / away_odds
    overround = (raw_home + raw_away - 1.0) * 100.0
    total = raw_home + raw_away
    return raw_home / total, raw_away / total, overround


def calculate_consensus(odds_df: pl.DataFrame) -> pl.DataFrame:
    """Average implied probabilities across all bookmakers per match.

    Input: DataFrame with columns [home_team, away_team, bookmaker, home_odds, away_odds]
    Output: DataFrame with [home_team, away_team, consensus_home_prob, consensus_away_prob, avg_overround]
    """
    if odds_df.is_empty():
        return pl.DataFrame()

    return (
        odds_df.with_columns(
            (1.0 / pl.col("home_odds")).alias("raw_home_prob"),
            (1.0 / pl.col("away_odds")).alias("raw_away_prob"),
        )
        .with_columns(
            (pl.col("raw_home_prob") + pl.col("raw_away_prob")).alias("total_prob"),
        )
        .with_columns(
            (pl.col("raw_home_prob") / pl.col("total_prob")).alias("home_prob"),
            (pl.col("raw_away_prob") / pl.col("total_prob")).alias("away_prob"),
            ((pl.col("total_prob") - 1.0) * 100.0).alias("overround_pct"),
        )
        .group_by("home_team", "away_team")
        .agg(
            pl.col("home_prob").mean().alias("consensus_home_prob"),
            pl.col("away_prob").mean().alias("consensus_away_prob"),
            pl.col("overround_pct").mean().alias("avg_overround"),
            pl.col("bookmaker").n_unique().alias("num_bookmakers"),
        )
        .sort("home_team")
    )


def best_available_odds(odds_df: pl.DataFrame) -> pl.DataFrame:
    """Find the highest available odds per outcome per match, with bookmaker name.

    Returns DataFrame with [home_team, away_team, best_home_odds, best_home_bookie,
                            best_away_odds, best_away_bookie]
    """
    if odds_df.is_empty():
        return pl.DataFrame()

    best_home = (
        odds_df.sort("home_odds", descending=True)
        .group_by("home_team", "away_team")
        .first()
        .select(
            "home_team",
            "away_team",
            pl.col("home_odds").alias("best_home_odds"),
            pl.col("bookmaker").alias("best_home_bookie"),
        )
    )

    best_away = (
        odds_df.sort("away_odds", descending=True)
        .group_by("home_team", "away_team")
        .first()
        .select(
            "home_team",
            "away_team",
            pl.col("away_odds").alias("best_away_odds"),
            pl.col("bookmaker").alias("best_away_bookie"),
        )
    )

    return best_home.join(best_away, on=["home_team", "away_team"], how="inner")
