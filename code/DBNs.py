"""
Dynamic Bayesian Network (DBN) demo – English friendly output
Stocks: TSLA · NVDA · AMD · TSM   (last 800 trading days)

What you'll get
---------------
1.  "Momentum" question – If TSLA & NVDA rally today, how likely is AMD to rally tomorrow?
2.  "Shock table"       – Which single-day rally helps TSM the most for tomorrow?
3.  "Who-leads-whom"     – Arrow diagram of intra-day causal flow.
4.  Basket probability   – Odds that **all four** finish in their top-tercile tomorrow.

Libraries: pandas · numpy · yfinance · pgmpy ≥ 0.1.7 · networkx · matplotlib
"""

import warnings, numpy as np, pandas as pd, yfinance as yf
import matplotlib.pyplot as plt, networkx as nx
warnings.filterwarnings("ignore")

# pgmpy – handle API changes gracefully
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.inference  import DBNInference

# BicScore changed case in old vs new releases
try:
    from pgmpy.estimators import BicScore as _BIC
except ImportError:        # very old pgmpy
    from pgmpy.estimators import BicScore as _BIC

# -------------------------------------------------- #
# Download adjusted closes robustly        #
# -------------------------------------------------- #
def fetch_adj_close(tickers, lookback_days=850):
    df = yf.download(tickers, period=f"{lookback_days}d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.swaplevel(axis=1)  # ticker first
        out = pd.concat([(df[t]["Adj Close"]
                          if "Adj Close" in df[t]
                          else df[t]["Close"])
                         for t in tickers], axis=1)
    else:
        out = pd.concat([(df[f"{t} Adj Close"]
                          if f"{t} Adj Close" in df.columns
                          else df[f"{t} Close"])
                         for t in tickers], axis=1)
    out.columns = tickers
    return out

# -------------------------------------------------- #
# Data                                               #
# -------------------------------------------------- #
TIKS  = ["TSLA", "NVDA", "AMD", "TSM"]
prices = fetch_adj_close(TIKS).dropna().tail(800)
rets   = prices.pct_change().dropna()

# discretise each column into terciles 0/1/2
def tercile(series):
    q = series.rank(pct=True)
    return pd.cut(q, bins=[-0.01, 1/3, 2/3, 1.01],
                  labels=[0, 1, 2]).astype(int)

disc_cols = {c: tercile(rets[c]) for c in rets.columns}
disc      = pd.DataFrame(disc_cols)

# add time level
disc.columns = pd.MultiIndex.from_product([disc.columns, [0]],
                                          names=["sym", "time"])
disc.reset_index(drop=True, inplace=True)

# tomorrow slice (time=1)
lags = disc.copy()
lags.columns = lags.columns.set_levels([1], level="time")
lags.index   = lags.index - 1
disc_t12     = pd.concat([disc, lags], axis=1).dropna().astype(int)

# -------------------------------------------------- #
# Structure learning (handles old/new API styles)    #
# -------------------------------------------------- #
slice0 = disc_t12.xs(0, level="time", axis=1)
score  = _BIC(slice0)
try:                              # old API
    hc      = HillClimbSearch(slice0, scoring_method=score)
    best_bn = hc.estimate()
except TypeError:                 # new API
    hc      = HillClimbSearch(slice0)
    best_bn = hc.estimate(scoring_method=score)

# -------------------------------------------------- #
# Build Dynamic BN                                #
# -------------------------------------------------- #
dbn = DBN()
for u, v in best_bn.edges():
    dbn.add_edge((u, 0), (v, 0))      # intra-slice
for s in TIKS:
    dbn.add_edge((s, 0), (s, 1))      # self-loops
for u, v in best_bn.edges():          # copy edges forward
    dbn.add_edge((u, 0), (v, 1))

# -------------------------------------------------- #
#  Fit CPTs & make inference object                  #
# -------------------------------------------------- #
dbn.fit(disc_t12)
inf = DBNInference(dbn)

def prob_up(sym, evidence_today):
    """Return [P(down), P(flat), P(up)] for sym at t+1."""
    return inf.query([(sym, 1)], evidence=evidence_today)[(sym, 1)].values

# -------------------------------------------------- #
# Q-1  AMD ↑ given TSLA ↑ & NVDA ↑ today             #
# -------------------------------------------------- #
today = {("TSLA", 0): 2, ("NVDA", 0): 2}
print("\nQ1  P(AMD up) = {:.1f}%".format(prob_up("AMD", today)[2]*100))

# -------------------------------------------------- #
# Q-2  One-day "shock" ranking for TSM               #
# -------------------------------------------------- #
baseline = {(s, 0): 1 for s in TIKS}
base_p   = prob_up("TSM", baseline)[2]
deltas = []
for s in TIKS:
    shocked = baseline.copy(); shocked[(s, 0)] = 2
    deltas.append((s, prob_up("TSM", shocked)[2] - base_p))

print("\nQ2  ΔP(TSM up) from single-stock up-shocks:")
for s, d in sorted(deltas, key=lambda x: x[1], reverse=True):
    print(f"   {s}: {d:+.3f}")

# -------------------------------------------------- #
# Q-3  Draw causal map ("who leads whom", t = 0)     #
# -------------------------------------------------- #
# build an nx.DiGraph from the DBN's edge list
G_full = nx.DiGraph()
G_full.add_edges_from(dbn.edges())

# keep only intra-slice edges (time = 0 → time = 0)
edges_today = [(u[0], v[0]) for u, v in G_full.edges() if u[1] == v[1] == 0]

plt.figure(figsize=(5, 4))
nx.draw_networkx(nx.DiGraph(edges_today), arrows=True,
                 node_color="white", edgecolors="black")
plt.title("Who-Leads-Whom (today slice)")
plt.axis("off")
plt.tight_layout()
plt.show()

# -------------------------------------------------- #
# Q-4  Portfolio ↑ probability tomorrow             #
# -------------------------------------------------- #
today_ex = {("TSLA",0):0, ("NVDA",0):1, ("AMD",0):1, ("TSM",0):2}
p_port   = np.prod([prob_up(sym, today_ex)[2] for sym in TIKS])
print("\nQ4  P(portfolio up) = {:.1f}%".format(p_port*100))
