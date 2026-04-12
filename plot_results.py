"""
plot_results.py
===============
Merge four simulation CSV files and plot:
  1. Dispatched wind energy (mean ± 1 SD) across all runs.
  2. Wind-power revenue (mean ± 1 SD) per penalty rate.

Column names are inferred automatically from each file independently,
then all files are merged into a single dataframe before plotting.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CSV_PATHS = [
    "./market_1/market1_results.csv",
    "./market_2/market2_results.csv",
    "./market_3/market3_results.csv",
    "./market_4/market4_results.csv",
]

DISPATCH_OUT = "wind_dispatch.png"
REVENUE_OUT = "wind_revenue.png"
COLOURS = ["steelblue", "darkorange", "seagreen"]


# ---------------------------------------------------------------------------
# Column inference
# ---------------------------------------------------------------------------

def is_integer_like(series: pd.Series) -> bool:
    """True if all non-null values are whole numbers."""
    return pd.api.types.is_numeric_dtype(series) and (series.dropna() % 1 == 0).all()


def infer_columns(df: pd.DataFrame, source: str) -> dict:
    """
    Infer the role of each column from data characteristics.
    Returns a dict with keys: interval, kappa, run, dispatch, revenue.
    """
    int_cols = [c for c in df.columns if is_integer_like(df[c])]
    float_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                  and c not in int_cols]

    col_interval = next(
        (c for c in int_cols
         if set(df[c].dropna().astype(int).unique()) == set(range(1, 25))),
        None,
    )
    if col_interval is None:
        print(f"Error ({source}): could not identify interval column with values {{1..24}}.")
        sys.exit(1)

    remaining = [c for c in int_cols if c != col_interval]
    col_kappa = min(remaining, key=lambda c: df[c].nunique())
    remaining2 = [c for c in remaining if c != col_kappa]
    if not remaining2:
        print(f"Error ({source}): could not identify a run column.")
        sys.exit(1)
    col_run = remaining2[0]

    non_neg = [c for c in float_cols if (df[c].dropna() >= 0).all()]
    neg = [c for c in float_cols if c not in non_neg]

    if not non_neg or not neg:
        col_dispatch, col_revenue = float_cols[0], float_cols[1]
    else:
        col_dispatch = non_neg[0]
        col_revenue = neg[0]

    return {
        "interval": col_interval,
        "kappa": col_kappa,
        "run": col_run,
        "dispatch": col_dispatch,
        "revenue": col_revenue,
    }


# ---------------------------------------------------------------------------
# Load and merge all four CSV files
# ---------------------------------------------------------------------------

STANDARD_COLS = ["kappa", "run", "interval", "dispatch", "revenue"]

frames = []
for path in CSV_PATHS:
    raw = pd.read_csv(path)
    mapping = infer_columns(raw, path)
    print(f"{path}")
    for role, col in mapping.items():
        print(f"  {role:8s} : {col}")

    # Rename to standard column names and add a source tag
    renamed = raw[[mapping[r] for r in STANDARD_COLS]].copy()
    renamed.columns = STANDARD_COLS
    renamed["source"] = path
    frames.append(renamed)

df = pd.concat(frames, ignore_index=True)

# Make run indices unique across files so runs from different files
# are treated as independent replicates
df["run_uid"] = df["source"] + "_" + df["run"].astype(str)

kappa_values = sorted(df["kappa"].unique())
intervals = np.arange(1, 25)
odd_intervals = np.arange(1, 25, 2)

print(f"\nMerged dataset: {len(df)} rows, "
      f"{df['run_uid'].nunique()} unique runs, "
      f"{len(kappa_values)} penalty rates: {kappa_values}")

# ---------------------------------------------------------------------------
# Plot 1 — Dispatched wind energy (pooled across all kappa values and files)
# ---------------------------------------------------------------------------

dispatch_matrix = (
    df.groupby(["run_uid", "interval"])["dispatch"]
      .mean()
      .unstack("interval")
      .values
)

mean_dispatch = dispatch_matrix.mean(axis=0)
std_dispatch = dispatch_matrix.std(axis=0)

fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.fill_between(intervals,
                 mean_dispatch - std_dispatch,
                 mean_dispatch + std_dispatch,
                 color="steelblue", alpha=0.25, label=r"Mean $\pm$ 1 SD")
ax1.plot(intervals, mean_dispatch,
         color="steelblue", linewidth=2, marker="o", markersize=4,
         label="Mean dispatched wind")
ax1.set_xlabel("Interval $t$", fontsize=12)
ax1.set_ylabel("Dispatched wind energy (MWh)", fontsize=12)
ax1.set_xticks(odd_intervals)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(DISPATCH_OUT, dpi=150)
plt.close(fig1)
print(f"Dispatch plot saved to '{DISPATCH_OUT}'.")

# ---------------------------------------------------------------------------
# Plot 2 — Wind-power revenue per penalty rate
# ---------------------------------------------------------------------------

fig2, ax2 = plt.subplots(figsize=(6, 4))

for kappa_val, colour in zip(kappa_values, COLOURS):
    subset = df[df["kappa"] == kappa_val]

    revenue_matrix = (
        subset.groupby(["run_uid", "interval"])["revenue"]
        .mean()
        .unstack("interval")
        .values
    )

    mean_rev = revenue_matrix.mean(axis=0)
    std_rev = revenue_matrix.std(axis=0)

    ax2.fill_between(intervals,
                     mean_rev - std_rev,
                     mean_rev + std_rev,
                     color=colour, alpha=0.2,
                     label=f"$q_u$ = {int(kappa_val)}, " + r"$\pm$ 1 SD")
    ax2.plot(intervals, mean_rev,
             color=colour, linewidth=2, marker="o", markersize=4,
             label=f"$q_u$ = {int(kappa_val)}, mean")

ax2.set_xlabel("Interval $t$", fontsize=12)
ax2.set_ylabel("Wind-power revenue (\\$)", fontsize=12)
ax2.set_xticks(odd_intervals)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(REVENUE_OUT, dpi=150)
plt.close(fig2)
print(f"Revenue plot saved to '{REVENUE_OUT}'.")
