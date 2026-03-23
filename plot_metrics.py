"""
Plot batch_size vs S-MFU and S-MBU for multiple MoE models.

Reads all cap_metrics_*.json files from RESULTS_DIR and produces:
  - plots/smfu_prefill.png
  - plots/smfu_decoding.png
  - plots/smbu_prefill.png
  - plots/smbu_decoding.png

Result JSON fields used (from moe_cap.utils.continuous_batching_utils):
  server_batch_size, model_name, dataset,
  prefill_smfu, decoding_smfu, prefill_smbu, decoding_smbu
"""

import json
import os
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
PLOTS_DIR = os.environ.get("PLOTS_DIR", "plots")

# ── Load all result JSONs ─────────────────────────────────────────────────────

records = []
pattern = os.path.join(RESULTS_DIR, "**", "cap_metrics_*.json")
for path in glob.glob(pattern, recursive=True):
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"  skip (bad JSON): {path}")
            continue
    data["_file"] = path
    records.append(data)

if not records:
    print(f"No cap_metrics_*.json found under '{RESULTS_DIR}/'")
    print("Run sweep.sh first to generate results.")
    raise SystemExit(1)

df = pd.DataFrame(records)

# Normalise model_name → short label
def short_name(full: str) -> str:
    return full.split("/")[-1] if "/" in full else full

df["model"] = df["model_name"].apply(short_name)

# Keep only rows with a valid batch size
df = df.dropna(subset=["server_batch_size"])
df["server_batch_size"] = df["server_batch_size"].astype(int)

# Convert fractions → percentages (values are already 0-1 from the metric utils)
for col in ["prefill_smfu", "decoding_smfu", "prefill_smbu", "decoding_smbu"]:
    if col in df.columns:
        df[col] = df[col] * 100

print(f"Loaded {len(df)} result records across {df['model'].nunique()} model(s).")
print(df[["model", "server_batch_size", "dataset",
          "prefill_smfu", "decoding_smfu",
          "prefill_smbu", "decoding_smbu"]].sort_values(["model", "server_batch_size"]).to_string(index=False))

# ── Aggregate: multiple runs for same (model, bs, dataset) → mean ─────────────
group_keys = ["model", "server_batch_size", "dataset"]
agg_cols = [c for c in ["prefill_smfu", "decoding_smfu", "prefill_smbu", "decoding_smbu"] if c in df.columns]
df_agg = df.groupby(group_keys)[agg_cols].mean().reset_index()

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Plotting helper ───────────────────────────────────────────────────────────

MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]
LINE_STYLES = ["-", "--", "-.", ":"]

def plot_metric(df_agg: pd.DataFrame, y_col: str, title: str, ylabel: str, out_path: str):
    if y_col not in df_agg.columns:
        print(f"  skip {out_path}: column '{y_col}' not in data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    datasets = df_agg["dataset"].unique()
    models = sorted(df_agg["model"].unique())
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            subset = df_agg[(df_agg["model"] == model) & (df_agg["dataset"] == dataset)]
            subset = subset.sort_values("server_batch_size")
            if subset.empty:
                continue
            label = f"{model} ({dataset})" if len(datasets) > 1 else model
            ax.plot(
                subset["server_batch_size"],
                subset[y_col],
                marker=MARKERS[i % len(MARKERS)],
                linestyle=LINE_STYLES[j % len(LINE_STYLES)],
                color=color_cycle[i % len(color_cycle)],
                label=label,
                linewidth=1.8,
                markersize=6,
            )

    ax.set_xlabel("Batch size (max-running-requests)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Generate all four plots ───────────────────────────────────────────────────

plot_metric(df_agg, "prefill_smfu",
            "Prefill S-MFU vs Batch Size",
            "Prefill S-MFU (%)",
            os.path.join(PLOTS_DIR, "smfu_prefill.png"))

plot_metric(df_agg, "decoding_smfu",
            "Decoding S-MFU vs Batch Size",
            "Decoding S-MFU (%)",
            os.path.join(PLOTS_DIR, "smfu_decoding.png"))

plot_metric(df_agg, "prefill_smbu",
            "Prefill S-MBU vs Batch Size",
            "Prefill S-MBU (%)",
            os.path.join(PLOTS_DIR, "smbu_prefill.png"))

plot_metric(df_agg, "decoding_smbu",
            "Decoding S-MBU vs Batch Size",
            "Decoding S-MBU (%)",
            os.path.join(PLOTS_DIR, "smbu_decoding.png"))

print("All plots written to", PLOTS_DIR)
