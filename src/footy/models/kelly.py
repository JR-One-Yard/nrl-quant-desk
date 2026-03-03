"""Kelly criterion, expected value, and edge calculation."""

from __future__ import annotations

import polars as pl


def kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """Full Kelly fraction: optimal bet size as fraction of bankroll.

    Returns 0 if no edge (negative Kelly).
    """
    if decimal_odds <= 1.0 or model_prob <= 0.0 or model_prob >= 1.0:
        return 0.0
    b = decimal_odds - 1.0  # net odds (profit per unit wagered)
    f = (b * model_prob - (1.0 - model_prob)) / b
    return max(0.0, f)


def half_kelly(model_prob: float, decimal_odds: float) -> float:
    """Half Kelly — conservative sizing."""
    return kelly_fraction(model_prob, decimal_odds) / 2.0


def expected_value(model_prob: float, decimal_odds: float) -> float:
    """Expected value as percentage of stake.

    EV = (prob * payout) - 1, expressed as %.
    """
    return (model_prob * decimal_odds - 1.0) * 100.0


def calculate_edges(
    consensus_df: pl.DataFrame,
    best_odds_df: pl.DataFrame,
    elo_predictions: dict[tuple[str, str], tuple[float, float]],
    bankroll: float = 1000.0,
) -> pl.DataFrame:
    """Join model probabilities with best market odds to find edges.

    Parameters
    ----------
    consensus_df : Market consensus probabilities per match
    best_odds_df : Best available odds per match with bookmaker
    elo_predictions : {(home, away): (home_prob, away_prob)} from Elo model
    bankroll : Bankroll for Kelly sizing

    Returns DataFrame with edge analysis per outcome.
    """
    if consensus_df.is_empty() or best_odds_df.is_empty():
        return pl.DataFrame()

    joined = consensus_df.join(best_odds_df, on=["home_team", "away_team"], how="inner")

    rows = []
    for r in joined.iter_rows(named=True):
        home = r["home_team"]
        away = r["away_team"]

        elo_home_prob, elo_away_prob = elo_predictions.get(
            (home, away), (r["consensus_home_prob"], r["consensus_away_prob"])
        )

        best_h_odds = r["best_home_odds"]
        best_a_odds = r["best_away_odds"]

        for side, model_p, market_p, odds, bookie in [
            ("home", elo_home_prob, r["consensus_home_prob"], best_h_odds, r["best_home_bookie"]),
            ("away", elo_away_prob, r["consensus_away_prob"], best_a_odds, r["best_away_bookie"]),
        ]:
            team = home if side == "home" else away
            opponent = away if side == "home" else home
            edge_pct = (model_p - market_p) * 100.0
            ev_pct = expected_value(model_p, odds)
            hk = half_kelly(model_p, odds)
            stake = round(hk * bankroll, 2)

            signal = "neutral"
            if edge_pct > 2.0 and ev_pct > 0:
                signal = "value"
            elif edge_pct < -2.0:
                signal = "fade"

            rows.append({
                "team": team,
                "opponent": opponent,
                "side": side,
                "model_prob": round(model_p * 100, 1),
                "market_prob": round(market_p * 100, 1),
                "edge_pct": round(edge_pct, 1),
                "best_odds": odds,
                "best_bookie": bookie,
                "ev_pct": round(ev_pct, 1),
                "half_kelly_pct": round(hk * 100, 2),
                "kelly_stake": stake,
                "signal": signal,
            })

    return pl.DataFrame(rows).sort("edge_pct", descending=True)
