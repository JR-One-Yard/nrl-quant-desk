"""NRL Quant Trading Desk — The Wall."""

from __future__ import annotations

import streamlit as st
import polars as pl
import plotly.graph_objects as go

from footy.clients.champion import ChampionDataClient
from footy.clients.odds import OddsAPIClient
from footy.models.implied import calculate_consensus, best_available_odds
from footy.models.elo import EloModel
from footy.models.kelly import calculate_edges
from footy.db.store import FootyStore

# ── Palette ──────────────────────────────────────────────────────
CREAM     = "#F6F5F0"
CARD      = "#FFFFFF"
BORDER    = "#E8E5DE"
BROWN     = "#33302E"
BROWN_LT  = "#7D7B77"
TAUPE     = "#B8A68F"
GREEN     = "#4A7C59"
GREEN_BG  = "#EEF4EF"
ROSE      = "#C4655A"
ROSE_BG   = "#FAEEEC"
AMBER     = "#C49A3C"
AMBER_BG  = "#FBF5E9"

# ── Short nicknames for tight spaces ────────────────────────────
NICKNAMES = {
    "Brisbane Broncos": "Broncos",
    "Canberra Raiders": "Raiders",
    "Canterbury-Bankstown Bulldogs": "Bulldogs",
    "Cronulla-Sutherland Sharks": "Sharks",
    "Dolphins": "Dolphins",
    "Gold Coast Titans": "Titans",
    "Manly-Warringah Sea Eagles": "Sea Eagles",
    "Melbourne Storm": "Storm",
    "Newcastle Knights": "Knights",
    "North Queensland Cowboys": "Cowboys",
    "Parramatta Eels": "Eels",
    "Penrith Panthers": "Panthers",
    "South Sydney Rabbitohs": "Rabbitohs",
    "St George-Illawarra Dragons": "Dragons",
    "Sydney Roosters": "Roosters",
    "Warriors": "Warriors",
    "Wests Tigers": "Tigers",
}


def nick(name: str) -> str:
    return NICKNAMES.get(name, name)


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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Source+Serif+4:wght@300;400;600&display=swap');

/* ── Base ────────────────────────────────── */
.stApp {{ background-color: {CREAM}; }}
.stApp header {{ background-color: {CREAM}; }}
.stMainBlockContainer {{ max-width: 1200px; padding-top: 2rem; }}
#MainMenu, footer, .stDeployButton {{ display: none !important; }}
.block-container {{ padding-top: 1rem; }}
hr {{ border: none; border-top: 1px solid {BORDER}; margin: 2.5rem 0; }}

/* ── Typography ──────────────────────────── */
h1, h2, h3 {{
    font-family: 'Source Serif 4', Georgia, serif !important;
    color: {BROWN} !important;
    font-weight: 400 !important;
    letter-spacing: -0.01em;
}}
h2 {{ font-size: 1.5rem !important; border: none !important; padding: 0 !important; }}
p, span, label, .stMarkdown, li {{
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: {BROWN};
}}

/* ── Sidebar ─────────────────────────────── */
section[data-testid="stSidebar"] {{
    background-color: {CARD};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {{
    color: {BROWN} !important;
    font-family: 'Inter', sans-serif;
}}

/* ── Buttons ─────────────────────────────── */
.stButton > button {{
    background-color: {BROWN} !important;
    color: {CREAM} !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.4rem !important;
}}
.stButton > button:hover {{
    opacity: 0.8 !important;
    background-color: {BROWN} !important;
    color: {CREAM} !important;
}}

/* ── Hide default metric widget ──────────── */
div[data-testid="stMetric"] {{ display: none; }}

/* ── Reusable classes ────────────────────── */
.serif {{ font-family: 'Source Serif 4', Georgia, serif; }}
.sans {{ font-family: 'Inter', -apple-system, sans-serif; }}
.c-brown {{ color: {BROWN}; }}
.c-lt {{ color: {BROWN_LT}; }}
.c-taupe {{ color: {TAUPE}; }}
.c-green {{ color: {GREEN}; }}
.c-rose {{ color: {ROSE}; }}

.section-label {{
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem; font-weight: 500;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: {TAUPE}; margin-bottom: 0.5rem;
}}
.pill {{
    display: inline-block; padding: 3px 10px; border-radius: 100px;
    font-size: 0.68rem; font-weight: 500; letter-spacing: 0.04em;
    text-transform: uppercase; font-family: 'Inter', sans-serif;
}}
.pill-ft {{ background: {BORDER}; color: {BROWN_LT}; }}
.pill-up {{ background: {AMBER_BG}; color: {AMBER}; }}
.pill-val {{ background: {GREEN_BG}; color: {GREEN}; }}
.pill-fade {{ background: {ROSE_BG}; color: {ROSE}; }}
.pill-neut {{ background: {BORDER}; color: {BROWN_LT}; }}

/* ── Card ────────────────────────────────── */
.card {{
    background: {CARD}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
}}
.card:hover {{ box-shadow: 0 2px 12px rgba(51,48,46,0.05); }}

/* ── Generic table ───────────────────────── */
.t {{
    width: 100%; border-collapse: collapse;
    font-family: 'Inter', sans-serif; font-size: 0.82rem;
}}
.t th {{
    padding: 0.6rem 0.75rem; text-align: left; font-weight: 500;
    font-size: 0.68rem; letter-spacing: 0.06em; text-transform: uppercase;
    color: {BROWN_LT}; border-bottom: 1px solid {BORDER};
}}
.t td {{
    padding: 0.55rem 0.75rem; color: {BROWN};
    border-bottom: 1px solid {BORDER};
}}
.t tr:last-child td {{ border-bottom: none; }}
.t .hi {{ font-weight: 600; color: {GREEN}; }}
.t .r {{ text-align: right; }}
.t .c {{ text-align: center; }}

/* ── KPI strip ───────────────────────────── */
.kpi {{
    display: flex; flex-wrap: wrap;
    border: 1px solid {BORDER}; border-radius: 8px;
    overflow: hidden; margin-bottom: 2rem; background: {CARD};
}}
.kpi > div {{
    flex: 1 1 0; min-width: 120px;
    padding: 1rem 1.1rem; border-right: 1px solid {BORDER};
}}
.kpi > div:last-child {{ border-right: none; }}
.kpi-l {{
    font-size: 0.63rem; font-weight: 500;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: {BROWN_LT}; margin-bottom: 3px;
    font-family: 'Inter', sans-serif;
}}
.kpi-v {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.25rem; color: {BROWN};
}}

/* ── Prob bar ────────────────────────────── */
.pb {{ display: flex; align-items: center; gap: 8px; margin: 5px 0; }}
.pb-lbl {{ font-size: 0.78rem; color: {BROWN}; width: 90px; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: 'Inter', sans-serif; }}
.pb-track {{ flex: 1; height: 6px; background: {BORDER}; border-radius: 3px; overflow: hidden; }}
.pb-fill {{ height: 100%; border-radius: 3px; }}
.pb-pct {{ font-size: 0.78rem; font-weight: 500; color: {BROWN}; width: 48px; font-family: 'Inter', sans-serif; }}

/* ── Edge card ───────────────────────────── */
.edge {{ border-left: 3px solid transparent; }}
.edge-v {{ border-left-color: {GREEN}; }}
.edge-f {{ border-left-color: {ROSE}; }}

/* ── Empty state ─────────────────────────── */
.empty {{
    text-align: center; padding: 2.5rem 2rem;
    color: {BROWN_LT}; font-family: 'Inter', sans-serif; font-size: 0.88rem;
}}
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
    return client.get_h2h_odds()


# ── Helpers ──────────────────────────────────────────────────────
def fmt_kick(kickoff_local) -> str:
    if kickoff_local is None:
        return ""
    try:
        if hasattr(kickoff_local, "strftime"):
            return kickoff_local.strftime("%a %d %b, %I:%M%p").replace(" 0", " ")
    except Exception:
        pass
    return str(kickoff_local)[:16]


def pill(label: str, cls: str) -> str:
    return f'<span class="pill {cls}">{label}</span>'


# ── Initialize ───────────────────────────────────────────────────
cd = get_cd_client()
odds_client = get_odds_client()
store = get_store()
fixture_df = load_fixture(cd)
current_round = cd.get_current_round()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:0.25rem 0 1rem">
        <div class="serif" style="font-size:1.35rem;color:{BROWN}">The Wall</div>
        <div class="sans" style="font-size:0.68rem;color:{BROWN_LT};letter-spacing:0.08em;text-transform:uppercase">NRL Quant Trading Desk</div>
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
    <div class="sans" style="font-size:0.76rem;color:{BROWN_LT};margin-top:0.5rem;line-height:1.8">
        Quota remaining: <strong style="color:{BROWN}">{quota_str}</strong><br>
        Last refresh: <strong style="color:{BROWN}">{age_str}</strong>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f'<div class="section-label">Position Sizing</div>', unsafe_allow_html=True)
    bankroll = st.number_input("Bankroll ($)", min_value=100, value=1000, step=100)
    edge_threshold = st.slider("Min. edge (%)", 0.0, 10.0, 2.0, 0.5)

# ── Load data ────────────────────────────────────────────────────
odds_df = load_odds(odds_client)
round_matches = fixture_df.filter(pl.col("round") == selected_round) if not fixture_df.is_empty() else pl.DataFrame()
completed_df = fixture_df.filter(pl.col("status") == "complete") if not fixture_df.is_empty() else pl.DataFrame()

elo = EloModel()
if not completed_df.is_empty():
    elo.bootstrap_from_results(completed_df)
    store.save_match_results(fixture_df)
    store.save_elo_ratings(elo.ratings)

consensus = pl.DataFrame()
best_odds_df = pl.DataFrame()
if not odds_df.is_empty():
    consensus = calculate_consensus(odds_df)
    best_odds_df = best_available_odds(odds_df)


# ═══════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════

# ── Header ───────────────────────────────────────────────────────
st.markdown(f"""
<div class="serif" style="font-size:2.1rem;color:{BROWN};margin-bottom:0.15rem">NRL Quant Trading Desk</div>
<div class="sans" style="font-size:0.82rem;color:{BROWN_LT};margin-bottom:1.5rem">Live market analysis, Elo power ratings &amp; edge detection</div>
""", unsafe_allow_html=True)

# KPI strip
n_matches = len(round_matches) if not round_matches.is_empty() else 0
n_complete_rd = len(round_matches.filter(pl.col("status") == "complete")) if not round_matches.is_empty() else 0
n_sched = n_matches - n_complete_rd
n_books = odds_df["bookmaker"].n_unique() if not odds_df.is_empty() else 0

st.markdown(f"""
<div class="kpi">
    <div><div class="kpi-l">Round</div><div class="kpi-v">{selected_round}</div></div>
    <div><div class="kpi-l">Matches</div><div class="kpi-v">{n_complete_rd} played, {n_sched} upcoming</div></div>
    <div><div class="kpi-l">Data Feed</div><div class="kpi-v">{"Connected" if not fixture_df.is_empty() else "—"}</div></div>
    <div><div class="kpi-l">Bookmakers</div><div class="kpi-v">{n_books if n_books else "—"}</div></div>
    <div><div class="kpi-l">API Quota</div><div class="kpi-v">{quota_str}</div></div>
</div>
""", unsafe_allow_html=True)


# ── Fixtures ─────────────────────────────────────────────────────
st.markdown(f'<div class="section-label">Fixtures</div>', unsafe_allow_html=True)
st.markdown(f"## Round {selected_round}")

if not round_matches.is_empty():
    rows_html = ""
    for row in round_matches.sort("kickoff_utc").iter_rows(named=True):
        status = row.get("status", "")
        h_score = row.get("home_score")
        a_score = row.get("away_score")

        if status == "complete" and h_score is not None:
            score_str = f"{int(h_score)} — {int(a_score)}"
            p = pill("Full Time", "pill-ft")
        else:
            score_str = "vs"
            p = pill("Upcoming", "pill-up")

        rows_html += f"""<tr>
            <td class="r" style="font-weight:500">{nick(row['home_team'])}</td>
            <td class="c" style="font-weight:600;font-size:1rem;min-width:70px">{score_str}</td>
            <td style="font-weight:500">{nick(row['away_team'])}</td>
            <td class="c">{p}</td>
            <td class="r c-lt" style="font-size:0.76rem">{row.get('venue','')}</td>
            <td class="r c-lt" style="font-size:0.76rem;white-space:nowrap">{fmt_kick(row.get('kickoff_local'))}</td>
        </tr>"""

    st.markdown(f"""
    <div class="card" style="padding:0;overflow:hidden">
        <table class="t">
            <thead><tr>
                <th class="r">Home</th><th class="c"></th><th>Away</th>
                <th class="c">Status</th><th class="r">Venue</th><th class="r">Kickoff</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="empty">No fixture data available.</div>', unsafe_allow_html=True)

st.divider()

# ── Bookmaker Odds ───────────────────────────────────────────────
st.markdown(f'<div class="section-label">Market</div>', unsafe_allow_html=True)
st.markdown("## Bookmaker Odds")

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
        unique_matches = round_odds.select("home_team", "away_team").unique().sort("home_team")
        cols = st.columns(2)

        for i, match_row in enumerate(unique_matches.iter_rows(named=True)):
            home, away = match_row["home_team"], match_row["away_team"]
            match_odds = round_odds.filter(
                (pl.col("home_team") == home) & (pl.col("away_team") == away)
            )
            rows_list = list(match_odds.iter_rows(named=True))
            best_h = max(r["home_odds"] for r in rows_list) if rows_list else 0
            best_a = max(r["away_odds"] for r in rows_list) if rows_list else 0

            rows_html = ""
            for r in rows_list:
                h_cls = ' class="hi r"' if r["home_odds"] == best_h else ' class="r"'
                a_cls = ' class="hi r"' if r["away_odds"] == best_a else ' class="r"'
                rows_html += f"""<tr>
                    <td>{r['bookmaker']}</td>
                    <td{h_cls}>{r['home_odds']:.2f}</td>
                    <td{a_cls}>{r['away_odds']:.2f}</td>
                </tr>"""

            html = f"""
            <div class="card" style="padding:0;overflow:hidden">
                <div class="sans" style="padding:0.75rem 1rem 0.5rem;font-size:0.82rem;font-weight:500;color:{BROWN}">
                    {nick(home)} vs {nick(away)}
                </div>
                <table class="t">
                    <thead><tr><th>Bookmaker</th><th class="r">{nick(home)}</th><th class="r">{nick(away)}</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>"""

            with cols[i % 2]:
                st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty">No odds available for this round.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="empty">Press <strong>Refresh Odds</strong> in the sidebar to load bookmaker prices.</div>', unsafe_allow_html=True)

st.divider()

# ── Market Consensus ─────────────────────────────────────────────
st.markdown(f'<div class="section-label">Analysis</div>', unsafe_allow_html=True)
st.markdown("## Market Consensus")

if not consensus.is_empty():
    for row in consensus.iter_rows(named=True):
        home = row["home_team"]
        away = row["away_team"]
        h_pct = row["consensus_home_prob"] * 100
        a_pct = row["consensus_away_prob"] * 100
        overround = row["avg_overround"]
        n_bk = row["num_bookmakers"]
        h_color = GREEN if h_pct > a_pct else TAUPE
        a_color = GREEN if a_pct > h_pct else TAUPE

        st.markdown(f"""
        <div class="card">
            <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:0.6rem;flex-wrap:wrap;gap:0.25rem">
                <div class="serif" style="font-size:1rem;color:{BROWN}">{nick(home)} vs {nick(away)}</div>
                <div class="sans" style="font-size:0.7rem;color:{BROWN_LT}">{n_bk} books &middot; {overround:.1f}% overround</div>
            </div>
            <div class="pb">
                <div class="pb-lbl">{nick(home)}</div>
                <div class="pb-track"><div class="pb-fill" style="width:{h_pct}%;background:{h_color}"></div></div>
                <div class="pb-pct">{h_pct:.1f}%</div>
            </div>
            <div class="pb">
                <div class="pb-lbl">{nick(away)}</div>
                <div class="pb-track"><div class="pb-fill" style="width:{a_pct}%;background:{a_color}"></div></div>
                <div class="pb-pct">{a_pct:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
elif odds_df.is_empty():
    st.markdown('<div class="empty">Load odds to view market consensus.</div>', unsafe_allow_html=True)

st.divider()

# ── Edge Detection ───────────────────────────────────────────────
st.markdown(f'<div class="section-label">Edge Detection</div>', unsafe_allow_html=True)
st.markdown("## Model vs Market")
st.markdown(f'<div class="sans" style="font-size:0.82rem;color:{BROWN_LT};margin-bottom:1.25rem">Where our Elo model disagrees with the bookmaker consensus. Positive edge = potential value.</div>', unsafe_allow_html=True)

if not odds_df.is_empty() and not consensus.is_empty() and not best_odds_df.is_empty():
    elo_preds: dict[tuple[str, str], tuple[float, float]] = {}
    for r in consensus.iter_rows(named=True):
        h, a = r["home_team"], r["away_team"]
        elo_preds[(h, a)] = elo.predict(h, a)

    edges = calculate_edges(consensus, best_odds_df, elo_preds, bankroll=bankroll)

    if not edges.is_empty():
        value_bets = edges.filter(pl.col("edge_pct") >= edge_threshold)
        neutral_bets = edges.filter(
            (pl.col("edge_pct") > -edge_threshold) & (pl.col("edge_pct") < edge_threshold)
        )
        fade_bets = edges.filter(pl.col("edge_pct") <= -edge_threshold)

        def render_edge_section(label: str, df: pl.DataFrame, label_cls: str) -> None:
            if df.is_empty():
                return
            st.markdown(f'<div class="sans" style="font-size:0.7rem;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:{label_cls};margin:1.25rem 0 0.75rem">{label} ({len(df)})</div>', unsafe_allow_html=True)

            # Build as a single table for clean alignment
            rows_html = ""
            for row in df.iter_rows(named=True):
                edge = row["edge_pct"]
                ev = row["ev_pct"]
                sig = row["signal"]
                e_cls = "c-green" if edge > 0 else "c-rose" if edge < 0 else ""
                ev_cls = "c-green" if ev > 0 else "c-rose" if ev < 0 else ""
                p_cls = "pill-val" if sig == "value" else "pill-fade" if sig == "fade" else "pill-neut"
                border_cls = "edge-v" if sig == "value" else "edge-f" if sig == "fade" else ""

                rows_html += f"""<tr>
                    <td style="font-weight:500">{nick(row['team'])}</td>
                    <td class="c-lt" style="font-size:0.76rem">{row['side']}</td>
                    <td class="r">{row['model_prob']:.1f}%</td>
                    <td class="r">{row['market_prob']:.1f}%</td>
                    <td class="r {e_cls}" style="font-weight:600">{edge:+.1f}%</td>
                    <td class="r">{row['best_odds']:.2f}</td>
                    <td class="r {ev_cls}">{ev:+.1f}%</td>
                    <td class="r">${row['kelly_stake']:.0f}</td>
                    <td class="c">{pill(sig, p_cls)}</td>
                    <td class="r c-lt" style="font-size:0.72rem">{row['best_bookie']}</td>
                </tr>"""

            st.markdown(f"""
            <div class="card edge {border_cls}" style="padding:0;overflow-x:auto">
                <table class="t">
                    <thead><tr>
                        <th>Team</th><th>Side</th><th class="r">Model</th><th class="r">Market</th>
                        <th class="r">Edge</th><th class="r">Odds</th><th class="r">EV</th>
                        <th class="r">Stake</th><th class="c">Signal</th><th class="r">Book</th>
                    </tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)

        if value_bets.is_empty():
            st.markdown(f'<div class="card" style="text-align:center;color:{BROWN_LT};padding:2rem">No value signals above {edge_threshold}% edge threshold.</div>', unsafe_allow_html=True)
        else:
            render_edge_section("Opportunities", value_bets, GREEN)

        render_edge_section("Within Threshold", neutral_bets, BROWN_LT)
        render_edge_section("Fade", fade_bets, ROSE)

        store.save_odds_snapshot(odds_df)
else:
    st.markdown('<div class="empty">Load odds to run edge detection analysis.</div>', unsafe_allow_html=True)

st.divider()

# ── Elo Power Rankings & Results ─────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown(f'<div class="section-label">Ratings</div>', unsafe_allow_html=True)
    st.markdown("## Elo Power Rankings")

    ratings_df = elo.get_ratings_df()
    if not ratings_df.is_empty():
        data = ratings_df.with_row_index("rank", offset=1)
        min_r = data["elo_rating"].min()
        max_r = data["elo_rating"].max()
        spread = max(max_r - min_r, 1)

        # Plotly horizontal bar — clean Aesop styling
        teams = data["team"].to_list()
        ratings = data["elo_rating"].to_list()
        colors = [GREEN if r > 1500 else ROSE if r < 1500 else TAUPE for r in ratings]
        short_names = [nick(t) for t in teams]

        fig = go.Figure(go.Bar(
            x=ratings,
            y=short_names,
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            text=[f"{r:.0f}" for r in ratings],
            textposition="outside",
            textfont=dict(family="Inter", size=11, color=BROWN),
        ))
        fig.update_layout(
            yaxis=dict(autorange="reversed", tickfont=dict(family="Inter", size=12, color=BROWN)),
            xaxis=dict(visible=False, range=[min(1440, min_r - 10), max_r + 40]),
            height=520,
            margin=dict(l=110, r=50, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            bargap=0.35,
        )
        fig.add_vline(x=1500, line_dash="dot", line_color=TAUPE, opacity=0.5)

        st.markdown('<div class="card" style="padding:0.75rem 0.5rem">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty">No completed matches yet to build ratings.</div>', unsafe_allow_html=True)

with col_right:
    st.markdown(f'<div class="section-label">Results</div>', unsafe_allow_html=True)
    st.markdown("## Completed")

    if not completed_df.is_empty():
        for row in completed_df.sort("kickoff_utc", descending=True).iter_rows(named=True):
            h_score = int(row["home_score"]) if row["home_score"] is not None else 0
            a_score = int(row["away_score"]) if row["away_score"] is not None else 0
            margin = h_score - a_score
            winner = nick(row["home_team"]) if margin > 0 else nick(row["away_team"]) if margin < 0 else "Draw"

            st.markdown(f"""
            <div class="card" style="padding:1rem 1.25rem">
                <div style="display:flex;justify-content:space-between;align-items:center;gap:0.5rem">
                    <div style="min-width:0">
                        <div class="serif" style="font-size:0.95rem;color:{BROWN}">
                            {nick(row['home_team'])} {h_score} — {a_score} {nick(row['away_team'])}
                        </div>
                        <div class="sans" style="font-size:0.72rem;color:{BROWN_LT};margin-top:2px">
                            Rd {row['round']} &middot; {row.get('venue','')}
                        </div>
                    </div>
                    <div style="text-align:right;flex-shrink:0">
                        <div class="sans" style="font-size:0.78rem;font-weight:500;color:{BROWN}">{winner}</div>
                        <div class="sans" style="font-size:0.72rem;color:{BROWN_LT}">by {abs(margin)}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty">No completed matches yet.</div>', unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:2.5rem 0 1rem;font-size:0.7rem;color:{TAUPE};letter-spacing:0.04em;font-family:'Inter',sans-serif">
    NRL Quant Trading Desk &middot; Educational purposes only &middot; Not financial advice
</div>
""", unsafe_allow_html=True)
