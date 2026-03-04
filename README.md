# NRL Prediction Market Insights

A Python app that ingests live NRL fixture data and bookmaker odds, runs Elo + market-implied probability models, and displays everything on a professional Streamlit dashboard ("The Wall"). Educational tool for understanding betting market mechanics.

**[Live Dashboard](https://jr-one-yard-nrl-quant-desk-srcfootydashboardapp-4kksgi.streamlit.app/)**

## Methodology

### Data Sources

**Champion Data** (`mc.championdata.com`) — The official NRL statistics provider. We pull the full 2026 season fixture (204 matches) in real-time: match status, scores, teams, venues, and kickoff times. This is the source of truth for results and is what the Elo model trains on.

**The Odds API** (`the-odds-api.com`) — Aggregates live pre-match odds from Australian bookmakers (Sportsbet, TAB, Ladbrokes, Neds, Unibet, etc.) in decimal format. We pull head-to-head (win) markets only. Quota-controlled — manual refresh to conserve the 500 requests/month free tier.

---

### Quant Models

#### 1. Market-Implied Probabilities

Every bookmaker's decimal odds encode an implied probability: `prob = 1 / odds`. But bookmakers build in a margin (the "overround") so the implied probabilities for both sides sum to more than 100%. We strip the overround by normalizing:

```
fair_prob = raw_prob / (raw_home_prob + raw_away_prob)
```

We then calculate the **consensus** — the average implied probability across all bookmakers for each match. This is the market's best estimate of true probability, since it aggregates information from all books. We also show the overround % (typically 3-6%) which represents the bookmaker's theoretical edge.

#### 2. Elo Ratings

A rating system that updates after every match. Every team starts at 1500. After each game:

- The winner gains rating points, the loser drops by the same amount
- **Home advantage**: +50 Elo points baked into the home team's expected performance (roughly a 57% baseline home win rate in NRL)
- **Margin of victory**: Blowouts move ratings more than close wins, using the FiveThirtyEight-style multiplier: `ln(|margin| + 1) * 2.2 / (rating_diff * 0.001 + 2.2)`. This auto-corrects — beating a weak team by 30 moves the needle less than beating a strong team by 30
- **K-factor of 32**: How reactive the system is. High enough to respond to form within a few rounds

The model processes all completed 2026 matches chronologically to build current ratings, then converts the rating gap between any two teams into a win probability using the logistic function.

#### 3. Kelly Criterion & Edge Detection

This is where the two models meet. For each match outcome we calculate:

- **Edge %** = Elo model probability minus market consensus probability. Positive edge means our model thinks the team is more likely to win than the market does
- **Expected Value (EV)** = `(model_prob × best_odds) - 1`. Positive EV means the bet is theoretically profitable long-term
- **Half-Kelly stake** = The Kelly criterion tells you the mathematically optimal fraction of your bankroll to wager given your edge and the odds. We use half-Kelly (betting half the optimal amount) because full Kelly is aggressive and assumes your model is perfectly calibrated — it isn't
- **Signal**: "value" (green) when edge > threshold and EV positive, "fade" (red) when the market has the team significantly shorter than our model

---

### What the Dashboard Shows

| Section | What It Tells You |
|---|---|
| **Fixtures** | Round-by-round match schedule with live status and scores |
| **Bookmaker Odds Comparison** | Pivot table showing each bookmaker's price per match — spot which book is offering the best price |
| **Market Analysis** | Consensus probabilities (what the market collectively thinks), overround %, and best available odds per outcome with the bookmaker offering them |
| **Edge Detection** | The money table — where Elo disagrees with the market, by how much, and what the optimal stake would be |
| **Elo Power Rankings** | Current team ratings ladder with bar chart — who's above/below the 1500 baseline |
| **Completed Results** | Historical results feeding the model, with margins |

---

### Key Caveat

The Elo model is deliberately simple — it only knows win/loss/margin and home advantage. It doesn't account for injuries, player changes, weather, travel, or the thousand other factors bookmakers price in. Early in the season with only a few results, the ratings are noisy. The value here is educational — seeing *how* edges are identified and sized, not blindly following the signals.

## Tech Stack

- Python 3.11+, `uv` package manager
- `httpx` (API clients), `polars` (DataFrames), `duckdb` (storage)
- `streamlit` + `plotly` (dashboard)

## Run Locally

```bash
# Install dependencies
uv sync

# Add your Odds API key
echo 'THE_ODDS_API_KEY=your_key' > .env

# Launch
uv run streamlit run src/footy/dashboard/app.py
```
