"""NRL Quant Trading Desk — The Wall."""

from __future__ import annotations

from datetime import datetime, timezone

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

# ── Palette ──────────────────────────────────────────────────────
CREAM      = "#F6F5F0"
CARD       = "#FFFFFF"
BORDER     = "#E8E5DE"
BROWN      = "#33302E"
BROWN_MED  = "#5C5854"
BROWN_LT   = "#7D7B77"
TAUPE      = "#B8A68F"
GREEN      = "#4A7C59"
GREEN_BG   = "#EEF4EF"
ROSE       = "#C4655A"
ROSE_BG    = "#FAEEEC"
AMBER      = "#C49A3C"
AMBER_BG   = "#FBF5E9"

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="NRL Quant Trading Desk",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,400&display=swap');

    :root {{
        --cream: {CREAM};
        --card: {CARD};
        --border: {BORDER};
        --brown: {BROWN};
        --brown-med: {BROWN_MED};
        --brown-lt: {BROWN_LT};
        --taupe: {TAUPE};
        --green: {GREEN};
        --green-bg: {GREEN_BG};
        --rose: {ROSE};
        --rose-bg: {ROSE_BG};
    }}

    .stApp {{
        background-color: var(--cream);
    }}

    .stApp header {{
        background-color: var(--cream);
    }}

    .stMainBlockContainer {{
        max-width: 1200px;
        padding-top: 2rem;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {CARD};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown label,
    section[data-testid="stSidebar"] label {{
        color: {BROWN} !important;
        font-family: 'Inter', sans-serif;
    }}

    /* Typography */
    h1, h2, h3, h4 {{
        font-family: 'Source Serif 4', Georgia, serif !important;
        color: {BROWN} !important;
        font-weight: 400 !important;
        letter-spacing: -0.01em;
    }}
    h1 {{ font-size: 2.2rem !important; }}
    h2 {{ font-size: 1.5rem !important; border: none !important; padding: 0 !important; }}
    h3 {{ font-size: 1.2rem !important; }}

    p, span, label, .stMarkdown, li {{
        font-family: 'Inter', -apple-system, sans-serif !important;
        color: {BROWN};
    }}

    /* Hide default Streamlit chrome */
    #MainMenu, footer, .stDeployButton {{ display: none !important; }}
    .block-container {{ padding-top: 1rem; }}

    /* Dividers */
    hr {{
        border: none;
        border-top: 1px solid {BORDER};
        margin: 2.5rem 0;
    }}

    /* Selectbox / inputs */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    div[data-testid="stSliderTickBarMin"],
    div[data-testid="stSliderTickBarMax"] {{
        font-family: 'Inter', sans-serif !important;
        color: {BROWN} !important;
    }}

    /* DataFrame overrides */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 8px;
        overflow: hidden;
    }}

    /* Button */
    .stButton > button {{
        background-color: {BROWN} !important;
        color: {CREAM} !important;
        border: none !important;
        border-radius: 4px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        padding: 0.6rem 1.5rem !important;
        transition: opacity 0.2s ease !important;
    }}
    .stButton > button:hover {{
        opacity: 0.8 !important;
        background-color: {BROWN} !important;
        color: {CREAM} !important;
    }}

    /* Status pills */
    .status-pill {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 100px;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-family: 'Inter', sans-serif;
    }}
    .pill-live {{ background: {GREEN_BG}; color: {GREEN}; }}
    .pill-complete {{ background: {BORDER}; color: {BROWN_LT}; }}
    .pill-scheduled {{ background: {AMBER_BG}; color: {AMBER}; }}

    /* Card component */
    .aes-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }}
    .aes-card-flush {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 1rem;
    }}

    /* Match card */
    .match-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: box-shadow 0.2s ease;
    }}
    .match-card:hover {{
        box-shadow: 0 2px 12px rgba(51, 48, 46, 0.06);
    }}
    .match-teams {{
        display: flex;
        align-items: center;
        gap: 1rem;
        flex: 1;
    }}
    .team-name {{
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 1rem;
        color: {BROWN};
        min-width: 200px;
    }}
    .team-name.home {{ text-align: right; }}
    .team-name.away {{ text-align: left; }}
    .match-score {{
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: {BROWN};
        padding: 0 0.75rem;
        min-width: 70px;
        text-align: center;
    }}
    .match-vs {{
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: {TAUPE};
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0 0.5rem;
    }}
    .match-meta {{
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: {BROWN_LT};
        text-align: right;
        min-width: 160px;
    }}
    .match-venue {{
        margin-bottom: 2px;
    }}

    /* KPI strip */
    .kpi-strip {{
        display: flex;
        gap: 0;
        border: 1px solid {BORDER};
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 2rem;
        background: {CARD};
    }}
    .kpi-item {{
        flex: 1;
        padding: 1.1rem 1.25rem;
        border-right: 1px solid {BORDER};
    }}
    .kpi-item:last-child {{ border-right: none; }}
    .kpi-label {{
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {BROWN_LT};
        margin-bottom: 4px;
    }}
    .kpi-value {{
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 1.3rem;
        color: {BROWN};
    }}

    /* Odds table */
    .odds-table {{
        width: 100%;
        border-collapse: collapse;
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
    }}
    .odds-table th {{
        padding: 0.7rem 1rem;
        text-align: left;
        font-weight: 500;
        font-size: 0.7rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {BROWN_LT};
        border-bottom: 1px solid {BORDER};
        background: {CREAM};
    }}
    .odds-table td {{
        padding: 0.65rem 1rem;
        color: {BROWN};
        border-bottom: 1px solid {BORDER};
    }}
    .odds-table tr:last-child td {{ border-bottom: none; }}
    .odds-table .best {{
        font-weight: 600;
        color: {GREEN};
    }}

    /* Probability bar */
    .prob-bar-container {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0.6rem 0;
    }}
    .prob-bar-label {{
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        color: {BROWN};
        min-width: 60px;
        text-align: right;
    }}
    .prob-bar-track {{
        flex: 1;
        height: 6px;
        background: {BORDER};
        border-radius: 3px;
        overflow: hidden;
    }}
    .prob-bar-fill {{
        height: 100%;
        border-radius: 3px;
        transition: width 0.4s ease;
    }}
    .prob-bar-pct {{
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        font-weight: 500;
        color: {BROWN};
        min-width: 45px;
    }}

    /* Edge row */
    .edge-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        transition: box-shadow 0.2s ease;
    }}
    .edge-card:hover {{
        box-shadow: 0 2px 12px rgba(51, 48, 46, 0.06);
    }}
    .edge-card.value {{ border-left: 3px solid {GREEN}; }}
    .edge-card.fade {{ border-left: 3px solid {ROSE}; }}
    .edge-header {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.75rem;
    }}
    .edge-team {{
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 1.05rem;
        color: {BROWN};
    }}
    .edge-badge {{
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 100px;
    }}
    .edge-badge.value {{ background: {GREEN_BG}; color: {GREEN}; }}
    .edge-badge.fade {{ background: {ROSE_BG}; color: {ROSE}; }}
    .edge-badge.neutral {{ background: {BORDER}; color: {BROWN_LT}; }}
    .edge-metrics {{
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 0.5rem;
    }}
    .edge-metric {{
        text-align: center;
    }}
    .edge-metric-label {{
        font-family: 'Inter', sans-serif;
        font-size: 0.6rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {BROWN_LT};
        margin-bottom: 2px;
    }}
    .edge-metric-value {{
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 500;
        color: {BROWN};
    }}
    .edge-metric-value.positive {{ color: {GREEN}; }}
    .edge-metric-value.negative {{ color: {ROSE}; }}

    /* Section label */
    .section-label {{
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {TAUPE};
        margin-bottom: 0.5rem;
    }}

    /* Elo ladder */
    .elo-row {{
        display: flex;
        align-items: center;
        padding: 0.6rem 0;
        border-bottom: 1px solid {BORDER};
    }}
    .elo-row:last-child {{ border-bottom: none; }}
    .elo-rank {{
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: {TAUPE};
        width: 28px;
        text-align: right;
        margin-right: 1rem;
    }}
    .elo-team {{
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 0.92rem;
        color: {BROWN};
        flex: 1;
    }}
    .elo-rating {{
        font-family: 'Inter', sans-serif;
        font-size: 0.88rem;
        font-weight: 500;
        min-width: 50px;
        text-align: right;
        margin-right: 1rem;
    }}
    .elo-bar-track {{
        width: 120px;
        height: 4px;
        background: {BORDER};
        border-radius: 2px;
        overflow: hidden;
        position: relative;
    }}
    .elo-bar-fill {{
        height: 100%;
        border-radius: 2px;
    }}

    /* Sidebar refinements */
    section[data-testid="stSidebar"] .stDivider {{
        border-color: {BORDER};
    }}

    /* Metric overrides - hide default */
    div[data-testid="stMetric"] {{ display: none; }}

    /* Empty state */
    .empty-state {{
        text-align: center;
        padding: 3rem 2rem;
        color: {BROWN_LT};
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
    }}
    .empty-state .empty-icon {{
        font-size: 2rem;
        margin-bottom: 0.75rem;
        opacity: 0.4;
    }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────
def format_kickoff(kickoff_local) -> str:
    """Format a kickoff datetime for display."""
    if kickoff_local is None:
        return ""
    try:
        if hasattr(kickoff_local, "strftime"):
            return kickoff_local.strftime("%a %d %b, %I:%M %p")
    except Exception:
        pass
    return str(kickoff_local)[:16]


def status_pill(status: str) -> str:
    cls = {"complete": "pill-complete", "scheduled": "pill-scheduled"}.get(
        status, "pill-live"
    )
    label = {"complete": "Full Time", "scheduled": "Upcoming"}.get(
        status, status.title()
    )
    return f'<span class="status-pill {cls}">{label}</span>'


def match_card_html(row: dict) -> str:
    """Render a single match as an HTML card."""
    home = row["home_team"]
    away = row["away_team"]
    venue = row.get("venue", "")
    kickoff = format_kickoff(row.get("kickoff_local"))
    status = row.get("status", "")
    h_score = row.get("home_score")
    a_score = row.get("away_score")

    if status == "complete" and h_score is not None and a_score is not None:
        center = f'<div class="match-score">{int(h_score)} — {int(a_score)}</div>'
    else:
        center = '<div class="match-vs">vs</div>'

    return f"""
    <div class="match-card">
        <div class="match-teams">
            <div class="team-name home">{home}</div>
            {center}
            <div class="team-name away">{away}</div>
        </div>
        <div class="match-meta">
            <div class="match-venue">{venue}</div>
            <div>{kickoff}</div>
            <div style="margin-top:4px">{status_pill(status)}</div>
        </div>
    </div>
    """


def kpi_strip_html(items: list[tuple[str, str]]) -> str:
    """Render a horizontal KPI strip."""
    cells = ""
    for label, value in items:
        cells += f"""
        <div class="kpi-item">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>"""
    return f'<div class="kpi-strip">{cells}</div>'


def prob_bar_html(label: str, pct: float, color: str) -> str:
    """Render a horizontal probability bar."""
    return f"""
    <div class="prob-bar-container">
        <div class="prob-bar-label">{label}</div>
        <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width:{pct}%;background:{color}"></div>
        </div>
        <div class="prob-bar-pct">{pct:.1f}%</div>
    </div>
    """


def edge_card_html(row: dict) -> str:
    """Render a single edge outcome as a card."""
    signal = row.get("signal", "neutral")
    edge = row.get("edge_pct", 0)
    ev = row.get("ev_pct", 0)

    edge_cls = "positive" if edge > 0 else "negative" if edge < 0 else ""
    ev_cls = "positive" if ev > 0 else "negative" if ev < 0 else ""

    return f"""
    <div class="edge-card {signal}">
        <div class="edge-header">
            <div class="edge-team">{row['team']} <span style="color:{BROWN_LT};font-family:Inter,sans-serif;font-size:0.8rem">({row['side']})</span> vs {row['opponent']}</div>
            <span class="edge-badge {signal}">{signal}</span>
        </div>
        <div class="edge-metrics">
            <div class="edge-metric">
                <div class="edge-metric-label">Model</div>
                <div class="edge-metric-value">{row['model_prob']:.1f}%</div>
            </div>
            <div class="edge-metric">
                <div class="edge-metric-label">Market</div>
                <div class="edge-metric-value">{row['market_prob']:.1f}%</div>
            </div>
            <div class="edge-metric">
                <div class="edge-metric-label">Edge</div>
                <div class="edge-metric-value {edge_cls}">{edge:+.1f}%</div>
            </div>
            <div class="edge-metric">
                <div class="edge-metric-label">Best Odds</div>
                <div class="edge-metric-value">{row['best_odds']:.2f}</div>
            </div>
            <div class="edge-metric">
                <div class="edge-metric-label">EV</div>
                <div class="edge-metric-value {ev_cls}">{ev:+.1f}%</div>
            </div>
            <div class="edge-metric">
                <div class="edge-metric-label">&frac12; Kelly</div>
                <div class="edge-metric-value">${row['kelly_stake']:.0f}</div>
            </div>
        </div>
        <div style="margin-top:6px;font-family:Inter,sans-serif;font-size:0.7rem;color:{BROWN_LT}">
            Best price at {row['best_bookie']}
        </div>
    </div>
    """


def elo_ladder_html(ratings_df: pl.DataFrame) -> str:
    """Render the Elo ladder as custom HTML."""
    rows_html = ""
    data = ratings_df.with_row_index("rank", offset=1)
    min_r = data["elo_rating"].min()
    max_r = data["elo_rating"].max()
    spread = max(max_r - min_r, 1)

    for row in data.iter_rows(named=True):
        r = row["elo_rating"]
        pct = ((r - min_r) / spread) * 100
        color = GREEN if r > 1500 else ROSE if r < 1500 else TAUPE
        rows_html += f"""
        <div class="elo-row">
            <div class="elo-rank">{row['rank']}</div>
            <div class="elo-team">{row['team']}</div>
            <div class="elo-rating" style="color:{color}">{r:.0f}</div>
            <div class="elo-bar-track">
                <div class="elo-bar-fill" style="width:{pct}%;background:{color}"></div>
            </div>
        </div>
        """
    return f'<div class="aes-card">{rows_html}</div>'


def odds_table_html(match_odds_rows: list[dict], home: str, away: str, best_home: float, best_away: float) -> str:
    """Render a bookmaker odds comparison table."""
    rows_html = ""
    for r in match_odds_rows:
        h_cls = ' class="best"' if r["home_odds"] == best_home else ""
        a_cls = ' class="best"' if r["away_odds"] == best_away else ""
        rows_html += f"""
        <tr>
            <td>{r['bookmaker']}</td>
            <td{h_cls}>{r['home_odds']:.2f}</td>
            <td{a_cls}>{r['away_odds']:.2f}</td>
        </tr>"""

    return f"""
    <div class="aes-card-flush">
        <table class="odds-table">
            <thead>
                <tr>
                    <th>Bookmaker</th>
                    <th>{home}</th>
                    <th>{away}</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """


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
    return client.get_h2h_odds()


# ── Initialize ───────────────────────────────────────────────────
cd = get_cd_client()
odds_client = get_odds_client()
store = get_store()

fixture_df = load_fixture(cd)
current_round = cd.get_current_round()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:0.5rem 0 1rem 0">
        <div style="font-family:'Source Serif 4',Georgia,serif;font-size:1.4rem;color:{BROWN};margin-bottom:2px">
            The Wall
        </div>
        <div style="font-family:Inter,sans-serif;font-size:0.7rem;color:{BROWN_LT};letter-spacing:0.08em;text-transform:uppercase">
            NRL Quant Trading Desk
        </div>
    </div>
    """, unsafe_allow_html=True)

    rounds_available = sorted(fixture_df["round"].unique().to_list()) if not fixture_df.is_empty() else [1]
    selected_round = st.selectbox(
        "Round", rounds_available,
        index=rounds_available.index(current_round) if current_round in rounds_available else 0,
    )

    st.divider()
    st.markdown(f'<div class="section-label">Market Data</div>', unsafe_allow_html=True)

    if st.button("Refresh Odds"):
        odds_client.invalidate_cache()
        st.rerun()

    quota = odds_client.get_remaining_quota()
    cache_age = odds_client.get_cache_age_seconds()

    quota_str = str(quota) if quota is not None else "—"
    age_str = f"{int(cache_age)}s ago" if cache_age else "Not fetched"
    st.markdown(f"""
    <div style="font-family:Inter,sans-serif;font-size:0.78rem;color:{BROWN_LT};margin-top:0.5rem;line-height:1.7">
        API quota remaining: <strong style="color:{BROWN}">{quota_str}</strong><br>
        Last refresh: <strong style="color:{BROWN}">{age_str}</strong>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f'<div class="section-label">Position Sizing</div>', unsafe_allow_html=True)
    bankroll = st.number_input("Bankroll ($)", min_value=100, value=1000, step=100)
    edge_threshold = st.slider("Min. edge threshold (%)", 0.0, 10.0, 2.0, 0.5)

# ── Load data ────────────────────────────────────────────────────
odds_df = load_odds(odds_client)
round_matches = fixture_df.filter(pl.col("round") == selected_round) if not fixture_df.is_empty() else pl.DataFrame()
completed_df = fixture_df.filter(pl.col("status") == "complete") if not fixture_df.is_empty() else pl.DataFrame()

# Bootstrap Elo
elo = EloModel()
if not completed_df.is_empty():
    elo.bootstrap_from_results(completed_df)
    store.save_match_results(fixture_df)
    store.save_elo_ratings(elo.ratings)

# Pre-compute market data
consensus = pl.DataFrame()
best_odds = pl.DataFrame()
if not odds_df.is_empty():
    consensus = calculate_consensus(odds_df)
    best_odds = best_available_odds(odds_df)

# ═══════════════════════════════════════════════════════════════════
# ── MAIN LAYOUT ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

# ── Header ───────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:0.25rem">
    <span style="font-family:'Source Serif 4',Georgia,serif;font-size:2.2rem;color:{BROWN}">
        NRL Quant Trading Desk
    </span>
</div>
<div style="font-family:Inter,sans-serif;font-size:0.82rem;color:{BROWN_LT};margin-bottom:1.5rem">
    Live market analysis, Elo power ratings &amp; edge detection
</div>
""", unsafe_allow_html=True)

# KPI strip
n_matches = len(round_matches) if not round_matches.is_empty() else 0
n_complete = len(round_matches.filter(pl.col("status") == "complete")) if not round_matches.is_empty() else 0
n_scheduled = n_matches - n_complete
cd_status = "Connected" if not fixture_df.is_empty() else "Unavailable"
n_bookmakers = odds_df["bookmaker"].n_unique() if not odds_df.is_empty() else 0

st.markdown(kpi_strip_html([
    ("Round", str(selected_round)),
    ("Matches", f"{n_complete} played, {n_scheduled} upcoming"),
    ("Champion Data", cd_status),
    ("Bookmakers", str(n_bookmakers) if n_bookmakers else "—"),
    ("Odds Quota", quota_str),
]), unsafe_allow_html=True)


# ── Fixtures ─────────────────────────────────────────────────────
st.markdown(f"""
<div class="section-label">Fixtures</div>
<h2 style="margin-top:0">Round {selected_round}</h2>
""", unsafe_allow_html=True)

if not round_matches.is_empty():
    for row in round_matches.sort("kickoff_utc").iter_rows(named=True):
        st.markdown(match_card_html(row), unsafe_allow_html=True)
else:
    st.markdown(f'<div class="empty-state"><div class="empty-icon">&#9866;</div>No fixture data available</div>', unsafe_allow_html=True)

st.divider()

# ── Bookmaker Odds ───────────────────────────────────────────────
st.markdown(f"""
<div class="section-label">Market</div>
<h2 style="margin-top:0">Bookmaker Odds Comparison</h2>
""", unsafe_allow_html=True)

if not odds_df.is_empty():
    round_teams = set()
    if not round_matches.is_empty():
        for r in round_matches.iter_rows(named=True):
            round_teams.add((r["home_team"], r["away_team"]))

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
        # Two-column layout for odds tables
        unique_matches = round_odds.select("home_team", "away_team").unique().sort("home_team")
        match_list = unique_matches.iter_rows(named=True)
        cols = st.columns(2)

        for i, match_row in enumerate(match_list):
            home, away = match_row["home_team"], match_row["away_team"]
            match_odds = round_odds.filter(
                (pl.col("home_team") == home) & (pl.col("away_team") == away)
            )
            rows_data = match_odds.iter_rows(named=True)
            rows_list = list(rows_data)

            best_h = max(r["home_odds"] for r in rows_list) if rows_list else 0
            best_a = max(r["away_odds"] for r in rows_list) if rows_list else 0

            with cols[i % 2]:
                st.markdown(odds_table_html(rows_list, home, away, best_h, best_a), unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="empty-state"><div class="empty-icon">&#9866;</div>No odds available for this round</div>', unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-icon">&#9863;</div>
        Press <strong>Refresh Odds</strong> in the sidebar to load bookmaker prices
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Market Analysis ──────────────────────────────────────────────
st.markdown(f"""
<div class="section-label">Analysis</div>
<h2 style="margin-top:0">Market Consensus</h2>
""", unsafe_allow_html=True)

if not consensus.is_empty():
    for row in consensus.iter_rows(named=True):
        home = row["home_team"]
        away = row["away_team"]
        h_pct = row["consensus_home_prob"] * 100
        a_pct = row["consensus_away_prob"] * 100
        overround = row["avg_overround"]
        n_books = row["num_bookmakers"]

        st.markdown(f"""
        <div class="aes-card">
            <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:0.75rem">
                <div style="font-family:'Source Serif 4',Georgia,serif;font-size:1rem;color:{BROWN}">{home} vs {away}</div>
                <div style="font-family:Inter,sans-serif;font-size:0.7rem;color:{BROWN_LT}">{n_books} bookmakers &middot; {overround:.1f}% overround</div>
            </div>
            {prob_bar_html(home, h_pct, GREEN if h_pct > 50 else TAUPE)}
            {prob_bar_html(away, a_pct, GREEN if a_pct > 50 else TAUPE)}
        </div>
        """, unsafe_allow_html=True)
else:
    if odds_df.is_empty():
        st.markdown(f'<div class="empty-state"><div class="empty-icon">&#9863;</div>Load odds to view market consensus</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="empty-state"><div class="empty-icon">&#9866;</div>No consensus data available</div>', unsafe_allow_html=True)

st.divider()

# ── Edge Detection ───────────────────────────────────────────────
st.markdown(f"""
<div class="section-label">Edge Detection</div>
<h2 style="margin-top:0">Model vs Market</h2>
<p style="font-family:Inter,sans-serif;font-size:0.82rem;color:{BROWN_LT};margin-bottom:1.25rem">
    Where our Elo model disagrees with the bookmaker consensus. Positive edge signals potential value.
</p>
""", unsafe_allow_html=True)

if not odds_df.is_empty() and not consensus.is_empty() and not best_odds.is_empty():
    elo_preds: dict[tuple[str, str], tuple[float, float]] = {}
    for r in consensus.iter_rows(named=True):
        h, a = r["home_team"], r["away_team"]
        elo_preds[(h, a)] = elo.predict(h, a)

    edges = calculate_edges(consensus, best_odds, elo_preds, bankroll=bankroll)

    if not edges.is_empty():
        value_bets = edges.filter(pl.col("edge_pct") >= edge_threshold)
        fade_bets = edges.filter(pl.col("edge_pct") <= -edge_threshold)
        neutral_bets = edges.filter(
            (pl.col("edge_pct") > -edge_threshold) & (pl.col("edge_pct") < edge_threshold)
        )

        if not value_bets.is_empty():
            st.markdown(f'<div style="font-family:Inter,sans-serif;font-size:0.7rem;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:{GREEN};margin-bottom:0.75rem">Opportunities ({len(value_bets)})</div>', unsafe_allow_html=True)
            for row in value_bets.iter_rows(named=True):
                st.markdown(edge_card_html(row), unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="aes-card" style="text-align:center;color:{BROWN_LT};font-family:Inter,sans-serif;font-size:0.85rem;padding:2rem">
                No value signals above {edge_threshold}% edge threshold
            </div>
            """, unsafe_allow_html=True)

        if not neutral_bets.is_empty():
            st.markdown(f'<div style="font-family:Inter,sans-serif;font-size:0.7rem;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:{BROWN_LT};margin:1.25rem 0 0.75rem 0">Within Threshold ({len(neutral_bets)})</div>', unsafe_allow_html=True)
            for row in neutral_bets.iter_rows(named=True):
                st.markdown(edge_card_html(row), unsafe_allow_html=True)

        if not fade_bets.is_empty():
            st.markdown(f'<div style="font-family:Inter,sans-serif;font-size:0.7rem;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:{ROSE};margin:1.25rem 0 0.75rem 0">Fade ({len(fade_bets)})</div>', unsafe_allow_html=True)
            for row in fade_bets.iter_rows(named=True):
                st.markdown(edge_card_html(row), unsafe_allow_html=True)

        store.save_odds_snapshot(odds_df)
else:
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-icon">&#9863;</div>
        Load odds to run edge detection analysis
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Elo Power Rankings & Results ─────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown(f"""
    <div class="section-label">Ratings</div>
    <h2 style="margin-top:0">Elo Power Rankings</h2>
    """, unsafe_allow_html=True)

    ratings_df = elo.get_ratings_df()
    if not ratings_df.is_empty():
        st.markdown(elo_ladder_html(ratings_df), unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="empty-state"><div class="empty-icon">&#9866;</div>No completed matches yet to build ratings</div>', unsafe_allow_html=True)

with col_right:
    st.markdown(f"""
    <div class="section-label">Results</div>
    <h2 style="margin-top:0">Completed</h2>
    """, unsafe_allow_html=True)

    if not completed_df.is_empty():
        for row in completed_df.sort("kickoff_utc", descending=True).iter_rows(named=True):
            h_score = int(row["home_score"]) if row["home_score"] is not None else 0
            a_score = int(row["away_score"]) if row["away_score"] is not None else 0
            margin = h_score - a_score
            winner = row["home_team"] if margin > 0 else row["away_team"] if margin < 0 else "Draw"
            margin_str = f"by {abs(margin)}"

            st.markdown(f"""
            <div class="aes-card" style="padding:1rem 1.25rem">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                        <div style="font-family:'Source Serif 4',Georgia,serif;font-size:0.95rem;color:{BROWN}">
                            {row['home_team']} {h_score} — {a_score} {row['away_team']}
                        </div>
                        <div style="font-family:Inter,sans-serif;font-size:0.72rem;color:{BROWN_LT};margin-top:3px">
                            Rd {row['round']} &middot; {row.get('venue', '')}
                        </div>
                    </div>
                    <div style="text-align:right">
                        <div style="font-family:Inter,sans-serif;font-size:0.78rem;font-weight:500;color:{BROWN}">{winner}</div>
                        <div style="font-family:Inter,sans-serif;font-size:0.72rem;color:{BROWN_LT}">{margin_str}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="empty-state"><div class="empty-icon">&#9866;</div>No completed matches yet</div>', unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:2rem 0 1rem 0;font-family:Inter,sans-serif;font-size:0.7rem;color:{TAUPE};letter-spacing:0.04em">
    NRL Quant Trading Desk &middot; Educational purposes only &middot; Not financial advice
</div>
""", unsafe_allow_html=True)
