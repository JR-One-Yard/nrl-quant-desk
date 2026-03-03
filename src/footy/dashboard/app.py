"""NRL Quant Trading Desk — The Wall."""

from __future__ import annotations

import streamlit as st
import polars as pl
import plotly.graph_objects as go

from footy.config import normalize_team
from footy.clients.champion import ChampionDataClient
from footy.clients.odds import OddsAPIClient
from footy.models.implied import calculate_consensus, best_available_odds
from footy.models.elo import EloModel
from footy.models.kelly import calculate_edges
from footy.db.store import FootyStore

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="NRL Quant Trading Desk",
    page_icon="🏉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark professional CSS ────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: #1a1d23;
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 16px;
        margin: 4px 0;
    }
    .value-signal { color: #00d97e; font-weight: 700; }
    .fade-signal { color: #e63946; font-weight: 700; }
    .neutral-signal { color: #6c757d; }
    table { font-size: 0.85rem; }
    div[data-testid="stDataFrame"] { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ── Cached resources ─────────────────────────────────────────────
@st.cache_resource
def get_cd_client() -> ChampionDataClient:
    return ChampionDataClient()


@st.cache_resource
def get_odds_client() -> OddsAPIClient:
    return OddsAPIClient()


@st.cache_resource
def get_store() -> FootyStore:
    return FootyStore()


@st.cache_data(ttl=300)
def load_fixture(_client: ChampionDataClient) -> pl.DataFrame:
    return _client.get_fixture()


def load_odds(client: OddsAPIClient) -> pl.DataFrame:
    """Load odds — not cached by st.cache_data, controlled manually."""
    return client.get_h2h_odds()


# ── Initialize ───────────────────────────────────────────────────
cd = get_cd_client()
odds_client = get_odds_client()
store = get_store()

fixture_df = load_fixture(cd)
current_round = cd.get_current_round()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("Controls")

    rounds_available = sorted(fixture_df["round"].unique().to_list()) if not fixture_df.is_empty() else [1]
    selected_round = st.selectbox(
        "Round", rounds_available, index=rounds_available.index(current_round) if current_round in rounds_available else 0
    )

    st.divider()

    if st.button("Refresh Odds (uses API quota)", type="primary"):
        odds_client.invalidate_cache()
        st.rerun()

    quota = odds_client.get_remaining_quota()
    cache_age = odds_client.get_cache_age_seconds()
    st.caption(f"Quota remaining: {quota if quota is not None else '—'}")
    st.caption(f"Odds cache age: {int(cache_age)}s" if cache_age else "Odds: not yet fetched")

    st.divider()
    bankroll = st.number_input("Bankroll ($)", min_value=100, value=1000, step=100)
    edge_threshold = st.slider("Edge threshold (%)", 0.0, 10.0, 2.0, 0.5)

# ── Load data ────────────────────────────────────────────────────
odds_df = load_odds(odds_client)
round_matches = fixture_df.filter(pl.col("round") == selected_round) if not fixture_df.is_empty() else pl.DataFrame()
completed_df = fixture_df.filter(pl.col("status") == "complete") if not fixture_df.is_empty() else pl.DataFrame()

# Bootstrap Elo from completed matches
elo = EloModel()
if not completed_df.is_empty():
    elo.bootstrap_from_results(completed_df)
    store.save_match_results(fixture_df)
    store.save_elo_ratings(elo.ratings)

# ── Header ───────────────────────────────────────────────────────
st.title("NRL Quant Trading Desk")
st.caption("The Wall — Live Market Analysis & Edge Detection")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Round", selected_round)
col2.metric("Champion Data", "Connected" if not fixture_df.is_empty() else "Error")
col3.metric("Odds API Quota", str(quota) if quota is not None else "—")
col4.metric("Matches This Round", len(round_matches) if not round_matches.is_empty() else 0)

st.divider()

# ── Section 1: Fixtures ─────────────────────────────────────────
st.subheader(f"Round {selected_round} Fixtures")
if not round_matches.is_empty():
    display_fixtures = round_matches.select(
        "match_id", "status", "home_team", "away_team",
        "home_score", "away_score", "venue", "kickoff_local",
    )
    st.dataframe(display_fixtures, use_container_width=True, hide_index=True)
else:
    st.info("No fixture data available.")

st.divider()

# ── Section 2: Bookmaker Odds Comparison ─────────────────────────
st.subheader("Bookmaker Odds Comparison")
if not odds_df.is_empty():
    # Filter to current round matches by matching teams
    round_teams = set()
    if not round_matches.is_empty():
        for r in round_matches.iter_rows(named=True):
            round_teams.add((r["home_team"], r["away_team"]))

    # Filter odds to round matches
    if round_teams:
        round_odds = odds_df.filter(
            pl.struct(["home_team", "away_team"]).map_elements(
                lambda x: (x["home_team"], x["away_team"]) in round_teams,
                return_dtype=pl.Boolean,
            )
        )
    else:
        round_odds = odds_df

    if not round_odds.is_empty():
        # Create pivot: match vs bookmaker
        for match_row in round_odds.select("home_team", "away_team").unique().iter_rows(named=True):
            home, away = match_row["home_team"], match_row["away_team"]
            match_odds = round_odds.filter(
                (pl.col("home_team") == home) & (pl.col("away_team") == away)
            )
            st.markdown(f"**{home} vs {away}**")

            pivot_data = {}
            for r in match_odds.iter_rows(named=True):
                pivot_data[r["bookmaker"]] = {
                    "Home": r["home_odds"],
                    "Away": r["away_odds"],
                }

            if pivot_data:
                pivot_df = pl.DataFrame([
                    {"Bookmaker": bk, f"{home}": v["Home"], f"{away}": v["Away"]}
                    for bk, v in pivot_data.items()
                ])
                st.dataframe(pivot_df, use_container_width=True, hide_index=True)
    else:
        st.info("No odds available for this round's matches.")
else:
    st.info("No odds data loaded. Click 'Refresh Odds' in the sidebar.")

st.divider()

# ── Section 3: Market Analysis ───────────────────────────────────
st.subheader("Market Analysis — Consensus Probabilities")

if not odds_df.is_empty():
    consensus = calculate_consensus(odds_df)
    best_odds = best_available_odds(odds_df)

    if not consensus.is_empty():
        display_consensus = consensus.with_columns(
            (pl.col("consensus_home_prob") * 100).round(1).alias("Home %"),
            (pl.col("consensus_away_prob") * 100).round(1).alias("Away %"),
            pl.col("avg_overround").round(2).alias("Overround %"),
        ).select("home_team", "away_team", "Home %", "Away %", "Overround %", "num_bookmakers")

        st.dataframe(display_consensus, use_container_width=True, hide_index=True)

    if not best_odds.is_empty():
        st.markdown("**Best Available Odds**")
        st.dataframe(best_odds, use_container_width=True, hide_index=True)
else:
    st.info("Load odds to see market analysis.")

st.divider()

# ── Section 4: Edge Detection ────────────────────────────────────
st.subheader("Edge Detection")

if not odds_df.is_empty() and not consensus.is_empty() and not best_odds.is_empty():
    # Build Elo predictions for all matches with odds
    elo_preds: dict[tuple[str, str], tuple[float, float]] = {}
    for r in consensus.iter_rows(named=True):
        h, a = r["home_team"], r["away_team"]
        elo_preds[(h, a)] = elo.predict(h, a)

    edges = calculate_edges(consensus, best_odds, elo_preds, bankroll=bankroll)

    if not edges.is_empty():
        # Filter by edge threshold
        value_bets = edges.filter(pl.col("edge_pct") >= edge_threshold)
        fade_bets = edges.filter(pl.col("edge_pct") <= -edge_threshold)

        # Color-code the signals
        st.markdown("#### Value Bets (Model Edge)")
        if not value_bets.is_empty():
            st.dataframe(
                value_bets.select(
                    "team", "opponent", "side", "model_prob", "market_prob",
                    "edge_pct", "best_odds", "best_bookie", "ev_pct",
                    "half_kelly_pct", "kelly_stake", "signal",
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "edge_pct": st.column_config.NumberColumn("Edge %", format="%.1f%%"),
                    "ev_pct": st.column_config.NumberColumn("EV %", format="%.1f%%"),
                    "model_prob": st.column_config.NumberColumn("Model %", format="%.1f%%"),
                    "market_prob": st.column_config.NumberColumn("Market %", format="%.1f%%"),
                    "half_kelly_pct": st.column_config.NumberColumn("½ Kelly %", format="%.2f%%"),
                    "kelly_stake": st.column_config.NumberColumn("Stake $", format="$%.2f"),
                },
            )
        else:
            st.info(f"No value bets above {edge_threshold}% edge threshold.")

        st.markdown("#### All Outcomes")
        st.dataframe(
            edges.select(
                "team", "opponent", "side", "model_prob", "market_prob",
                "edge_pct", "best_odds", "best_bookie", "ev_pct", "signal",
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "edge_pct": st.column_config.NumberColumn("Edge %", format="%.1f%%"),
                "ev_pct": st.column_config.NumberColumn("EV %", format="%.1f%%"),
                "model_prob": st.column_config.NumberColumn("Model %", format="%.1f%%"),
                "market_prob": st.column_config.NumberColumn("Market %", format="%.1f%%"),
            },
        )

        # Save odds snapshot
        store.save_odds_snapshot(odds_df)
else:
    st.info("Load odds data to see edge detection analysis.")

st.divider()

# ── Section 5: Elo Power Rankings & Results ──────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Elo Power Rankings")
    ratings_df = elo.get_ratings_df()
    if not ratings_df.is_empty():
        # Add rank column
        ratings_display = ratings_df.with_row_index("rank", offset=1)

        fig = go.Figure(go.Bar(
            x=ratings_display["elo_rating"].to_list(),
            y=ratings_display["team"].to_list(),
            orientation="h",
            marker_color=["#00d97e" if r > 1500 else "#e63946" if r < 1500 else "#6c757d"
                          for r in ratings_display["elo_rating"].to_list()],
        ))
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            xaxis_title="Elo Rating",
            height=500,
            margin=dict(l=200, r=20, t=20, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#fafafa",
        )
        fig.add_vline(x=1500, line_dash="dash", line_color="#6c757d", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(ratings_display, use_container_width=True, hide_index=True)
    else:
        st.info("No completed matches yet to build Elo ratings.")

with col_right:
    st.subheader("Completed Results")
    if not completed_df.is_empty():
        results_display = completed_df.select(
            "round", "home_team", "home_score", "away_score", "away_team", "venue",
        ).with_columns(
            (pl.col("home_score") - pl.col("away_score")).alias("margin"),
        )
        st.dataframe(results_display, use_container_width=True, hide_index=True)
    else:
        st.info("No completed matches yet.")
