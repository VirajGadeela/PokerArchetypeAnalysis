## Overview

This project groups real online poker players into distinct **behavioral archetypes** based on how they play, then measures how profitable each archetype is over the long run and *where* on the street each one wins or loses money. It turns millions of raw hand records into a small set of interpretable player types (loose vs. tight, passive vs. aggressive) and a defensible, variance-controlled estimate of each type's win rate.

The analysis was built to support a research paper examining whether the behavioral "leaks" of losing archetypes are consistent with well-documented cognitive biases (loss aversion, the sunk-cost fallacy, and the illusion of control).

## Purpose

The model answers two questions:

1. **What distinct playing styles exist** in a large population of real players, derived purely from their betting behavior rather than from preset labels?
2. **How does each style perform financially** over thousands of hands, and at which stage of the hand (before vs. at showdown) does the money actually move?

## Problem it solves

Playing style and profitability are usually discussed anecdotally in poker, and cognitive biases are usually studied in small lab experiments with artificial stakes. This project connects the two: it uses a large naturalistic dataset to (a) discover player archetypes objectively via clustering, (b) attach a rigorous long-term win rate to each, and (c) locate the specific behavioral leak (for example, pre-showdown "bleed") that separates winners from losers. This provides a reproducible way to estimate the financial cost of a playing style, and by extension the cost of the biases that style reflects.

## Methodology summary

The core pipeline (`irc_archetypes.py`) runs in a single pass over the raw data:

1. **Parse** raw per-player hand files directly from the IRC database.
2. **Aggregate** six behavioral statistics per player:
   - **VPIP** — % of hands played voluntarily before the flop
   - **PFR** — % of hands raised before the flop
   - **AF** (aggression factor) — ratio of bets and raises to calls after the flop
   - **WTSD** — % of flops seen that reached showdown
   - **W$SD** — % of those showdowns won
3. **Filter** the sample: exclude self-identified bots (any account name containing `bot`) and keep only players with at least a minimum number of hands (default 1,000) so statistics are stable.
4. **Cluster** players with k-means on the six standardized features (default `k = 4`, chosen for interpretability against the standard looseness/aggression taxonomy). Candidate `k` values are evaluated with silhouette scores.
5. **Measure win rate** in big blinds per 100 hands (bb/100). Because poker win rate is heavily skewed by a few high-variance players, the **median** is used as the headline, with **95% bootstrap confidence intervals** and a **5% winsorized mean** as a robustness check.
6. **Decompose** each player's net result into **showdown** and **non-showdown** components to locate where each archetype gains or loses money.

An optional, experimental script (`irc_bias.py`) computes within-player behavioral signatures of specific biases (post-loss "tilt," house-money effect, and sunk-cost "stickiness") by re-reading the hands in time order and comparing winning vs. losing clusters. <!-- PLACEHOLDER: confirm whether irc_bias.py is included in your repo. It was not part of the final paper's analysis; keep or remove this section accordingly. -->

## Data / inputs

The model uses the **IRC Poker Database**, compiled by the Computer Poker Research Group at the University of Alberta: roughly ten million hands logged on the Internet Relay Chat poker server (1995–2001), with player identities preserved across hands.

- The analysis is restricted to the **fixed-limit Texas Hold'em ring game** (the `holdem` archives), which holds stakes constant across players.
- The raw archive is **large (~1 GB extracted)** and is **not included in this repository**. Download it separately:

```bash
# PLACEHOLDER: confirm this URL is still live before relying on it.
curl -O http://poker.cs.ualberta.ca/IRC/IRCdata.tgz
tar -xzf IRCdata.tgz
# then extract the limit hold'em months:
cd IRCdata
for f in holdem.*.tgz; do tar -xzf "$f" 2>/dev/null; done
```

After extraction, the model reads the per-player files at `.../holdem/<YYYYMM>/pdb/pdb.<playername>`.

## Outputs

Running the core pipeline produces:

| Output | Description |
|---|---|
| `player_archetypes.csv` | One row per qualifying player: the six behavioral stats, win rate (bb/100), showdown and non-showdown components, and assigned cluster. |
| `archetypes.png` | Scatter plot of players in VPIP vs. PFR space, colored by cluster. |
| Console tables | Cluster profiles (centroid statistics and player counts) and the win-rate + leak-decomposition table with confidence intervals. |

If `irc_bias.py` is run, it additionally produces `player_bias.csv` and a winners-vs-losers comparison table. <!-- PLACEHOLDER: remove if not included. -->

Publication-quality figures (silhouette scores, behavioral profiles, win-rate intervals, showdown decomposition) are produced by `make_figs.py`. <!-- PLACEHOLDER: confirm make_figs.py is in the repo; if not, either add it or delete this line. -->

## Installation and setup

```bash
# 1. Clone the repository
git clone <PLACEHOLDER: your-repo-url>
cd <PLACEHOLDER: repo-folder>

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Python version:** PLACEHOLDER — 3.9 or newer (developed and tested on Python 3.11).

## Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `scipy` (only required for the optional `irc_bias.py`)

<!-- These are confirmed from the scripts' imports. Pin exact versions in requirements.txt for full reproducibility. -->

## Running the model

1. Download and extract the data (see **Data / inputs**).
2. Open `irc_archetypes.py` and set the configuration block at the top:

```python
DATA_ROOT   = "PLACEHOLDER/path/to/IRCdata/holdem"  # the extracted limit-holdem tree
GAME_FILTER = ""      # leave empty; DATA_ROOT already narrows to holdem
MIN_HANDS   = 1000    # minimum hands per player
BIG_BLIND   = 10      # chips per big blind (10/20 limit game)
N_CLUSTERS  = 4       # number of archetypes
BOOTSTRAP   = 2000    # bootstrap resamples for confidence intervals
RANDOM_STATE = 42     # reproducibility seed
EXCLUDE_BOTS = True   # drop accounts with "bot" in the name
```

3. Run it:

```bash
python irc_archetypes.py
```

To save the console output as well as the files:

```bash
python irc_archetypes.py | tee results.txt
```

## Example usage

With `DATA_ROOT` pointed at the extracted `holdem` folder, `python irc_archetypes.py` prints a sanity check on the highest-volume player, the silhouette sweep, and then the two result tables. Example (abbreviated) output:

```
3122 players clear 1000 hands.
  k=3  silhouette=0.288
  k=4  silhouette=0.241

=== cluster profiles ===
         VPIP   PFR   AF  WTSD   WSD   hands  n_players
cluster
0        55.4   8.2  0.8  41.3  43.4  4032.4       1012
1        34.8   6.1  1.1  33.1  52.5  4336.1       1141
2        67.2  35.6  2.1  45.8  41.0  3309.0        155
3        40.5  15.0  1.7  36.3  48.9  3894.0        814

=== win rate & leak decomposition (bb/100, median) ===
 cluster  winrate_med          95% CI   sd_med  nonsd_med
       0        -6.52  [-7.80, -5.40]   +41.99     -49.19
       1        +9.22  [+8.58, +9.77]   +44.30     -34.67
       2       -12.89 [-18.67, -4.94]   +29.15     -40.81
       3       +12.22 [+11.32, +13.16]  +37.09     -25.10
```

<!-- These values are the actual results produced during development and are reproducible with RANDOM_STATE = 42. -->

## Understanding the generated files and results

- **Clusters** are labelled from their centroid statistics *after* the analysis, independently of profitability. The four archetypes correspond to the standard taxonomy: Loose-Passive, Tight-Passive, Loose-Aggressive, and Tight-Aggressive.
- **Win rate (bb/100):** positive means a long-term winner, negative a loser. Medians are reported rather than means because the distribution is heavy-tailed.
- **Showdown vs. non-showdown split:** every archetype tends to profit at showdown and lose beforehand; the size of the pre-showdown loss is what separates winners from losers.
- `player_archetypes.csv` can be reloaded for further analysis or to reproduce the figures without re-parsing the raw data.
