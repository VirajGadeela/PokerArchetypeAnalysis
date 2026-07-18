import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


DATA_ROOT = "./IRCdata/holdem"
GAME_FILTER = ""
MIN_HANDS = 1000
BIG_BLIND = 10
EXCLUDE_BOTS = True
N_CLUSTERS = 4
BOOTSTRAP = 2000
RANDOM_STATE = 42



def parse_pdb_line(line):
    """Parse one pdb line into a dict. Returns None on malformed lines.

    Expected whitespace-separated layout (CONFIRM money-column order):
      0 player  1 timestamp  2 n_players  3 position
      4 preflop 5 flop  6 turn  7 river
      8 bankroll  9 total_bet  10 winnings   11+ hole cards (only if shown)
    """
    p = line.split()
    if len(p) < 11:
        return None
    try:
        return {
            "player":   p[0],
            "pre":      p[4],
            "flop":     p[5],
            "turn":     p[6],
            "river":    p[7],
            "total_bet": int(p[9]),
            "winnings":  int(p[10]),
            "cards":     p[11:],
        }
    except (ValueError, IndexError):
        return None


def new_acc():
    """A fresh per-player accumulator."""
    return dict(hands=0, vpip=0, pfr=0, pf_b=0, pf_r=0, pf_c=0,
                saw_flop=0, showdowns=0, sd_won=0,
                net=0, sd_net=0, nonsd_net=0, bad=0)


def fold_line(acc, rec):
    """Update a player's accumulator with one parsed hand."""
    acc["hands"] += 1
    pre = rec["pre"]

    if any(a in pre for a in "cbr"):
        acc["vpip"] += 1

    if "r" in pre:
        acc["pfr"] += 1


    post = rec["flop"] + rec["turn"] + rec["river"]
    acc["pf_b"] += post.count("b")
    acc["pf_r"] += post.count("r")
    acc["pf_c"] += post.count("c")

    saw_flop = rec["flop"] not in ("-", "")
    if saw_flop:
        acc["saw_flop"] += 1

    reached_sd = len(rec["cards"]) >= 2
    if reached_sd:
        acc["showdowns"] += 1
        if rec["winnings"] > 0:
            acc["sd_won"] += 1

    net = rec["winnings"] - rec["total_bet"]
    acc["net"] += net
    if reached_sd:
        acc["sd_net"] += net
    else:
        acc["nonsd_net"] += net


def build_players(root, game_filter):
    """One memory-bounded pass over every pdb.* file under root."""
    players = {}
    pattern = os.path.join(root, "**", "pdb.*")
    files = [f for f in glob.iglob(pattern, recursive=True)
             if game_filter in f]
    print(f"Found {len(files)} pdb files matching '{game_filter}'")

    for i, path in enumerate(files):
        if i % 5000 == 0:
            print(f"  ...{i} files")
        with open(path, "r", errors="ignore") as fh:
            for line in fh:
                rec = parse_pdb_line(line)
                if rec is None:
                    continue
                if EXCLUDE_BOTS and "bot" in rec["player"].lower():
                    continue
                acc = players.setdefault(rec["player"], new_acc())
                fold_line(acc, rec)
    return players


def to_frame(players, min_hands):
    rows = []
    for name, a in players.items():
        if a["hands"] < min_hands:
            continue
        calls = a["pf_c"] if a["pf_c"] else 1
        rows.append({
            "player": name,
            "hands": a["hands"],
            "VPIP": 100 * a["vpip"] / a["hands"],
            "PFR": 100 * a["pfr"] / a["hands"],
            "AF": (a["pf_b"] + a["pf_r"]) / calls,
            "WTSD": 100 * a["showdowns"] / (a["saw_flop"] or 1),
            "WSD":  100 * a["sd_won"] / (a["showdowns"] or 1),
            "winrate_bb100": (a["net"] / BIG_BLIND) / (a["hands"] / 100),
            "sd_bb100":      (a["sd_net"] / BIG_BLIND) / (a["hands"] / 100),
            "nonsd_bb100":   (a["nonsd_net"] / BIG_BLIND) / (a["hands"] / 100),
        })
    df = pd.DataFrame(rows)
    print(f"\n{len(df)} players clear {min_hands} hands.")
    if len(df) < 40:
        print("WARNING: small sample for clustering. Consider pooling tiers "
              "or a split threshold (lower bar to cluster, 1000+ for win rate).")
    return df


def cluster(df, k):
    feats = ["VPIP", "PFR", "AF", "WTSD", "WSD"]
    X = StandardScaler().fit_transform(df[feats])
 
    for kk in range(3, 7):
        km = KMeans(n_clusters=kk, n_init=10, random_state=RANDOM_STATE).fit(X)
        print(f"  k={kk}  silhouette={silhouette_score(X, km.labels_):.3f}")
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    df["cluster"] = km.labels_
    return df, feats


def bootstrap_ci(vals, n=BOOTSTRAP):
    vals = np.asarray(vals)
    stats = [np.median(np.random.choice(vals, size=len(vals), replace=True))
             for _ in range(n)]
    return np.percentile(stats, [2.5, 97.5])


def winsorized_mean(vals, pct=5):
    """Mean after clipping the top/bottom pct% -- robust to a few outliers."""
    vals = np.asarray(vals, dtype=float)
    lo, hi = np.percentile(vals, [pct, 100 - pct])
    return np.clip(vals, lo, hi).mean()


def summarize(df, feats):
    print("\n=== cluster profiles (label these from the centroids) ===")
    prof = df.groupby("cluster")[feats + ["hands"]].mean().round(1)
    prof["n_players"] = df.groupby("cluster").size()
    print(prof)


    print("\n=== win rate & leak decomposition (bb/100, median) ===")
    print(f"{'cl':>2} {'winrate_med':>11} {'95% CI':>20} "
          f"{'winz_mean':>10} {'raw_mean':>10} "
          f"{'sd_med':>8} {'nonsd_med':>10}")
    for c, g in df.groupby("cluster"):
        wr = g["winrate_bb100"].values
        lo, hi = bootstrap_ci(wr)
        print(f"{c:>2} {np.median(wr):>11.2f} "
              f"[{lo:>8.2f},{hi:>8.2f}] "
              f"{winsorized_mean(wr):>10.2f} {wr.mean():>10.2f} "
              f"{g['sd_bb100'].median():>8.2f} {g['nonsd_bb100'].median():>10.2f}")


def scatter(df, path="archetypes.png"):
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(df["VPIP"], df["PFR"], c=df["cluster"],
                     cmap="tab10", alpha=0.6, s=12)
    plt.plot([0, 100], [0, 100], "k--", lw=0.5)
    plt.xlabel("VPIP (%)"); plt.ylabel("PFR (%)")
    plt.title("Player archetypes (color = cluster)")
    plt.legend(*sc.legend_elements(), title="cluster")
    plt.tight_layout(); plt.savefig(path, dpi=150)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    players = build_players(DATA_ROOT, GAME_FILTER)

    top = max(players, key=lambda n: players[n]["hands"])
    print(f"\nSanity check on '{top}': {players[top]}")

    df = to_frame(players, MIN_HANDS)
    if len(df):
        df, feats = cluster(df, N_CLUSTERS)
        summarize(df, feats)
        scatter(df)
        df.to_csv("player_archetypes.csv", index=False)
        print("Saved player_archetypes.csv")
