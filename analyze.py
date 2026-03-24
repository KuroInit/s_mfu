#!/usr/bin/env python3
"""Post-processing script: recomputes S_MFU/S_MBU from sweep results and plots them."""

import glob
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moe_cap.configs import CAPConfig
from moe_cap.model_loader import HFModelInfoRetriever
from moe_cap.utils.continuous_batching_utils import _calculate_continuous_metrics


def find_latest_file(directory: Path, pattern: str) -> Optional[Path]:
    """Return the Path matching pattern with the latest name (lexicographic), or None."""
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        warnings.warn(f"Multiple files match {pattern} in {directory}; using latest: {matches[-1].name}")
    return matches[-1]


def load_triple(leaf_dir: Path, bs_dir_name: str):
    """Load metadata and records from a leaf results directory.

    Returns (metadata_dict, records_list) or (None, None) on failure.
    The metadata dict is enriched with a resolved 'batch_size' key at the top level.
    """
    meta_file = find_latest_file(leaf_dir, "metadata_*.json")
    records_file = find_latest_file(leaf_dir, "detailed_results_*.jsonl")

    if meta_file is None:
        warnings.warn(f"No metadata file in {leaf_dir} — skipping")
        return None, None
    if records_file is None:
        warnings.warn(f"No detailed_results file in {leaf_dir} — skipping")
        return None, None

    with open(meta_file) as f:
        meta = json.load(f)

    bs = meta.get("system_environment", {}).get("batch_size")
    if bs is None:
        try:
            bs = int(bs_dir_name.lstrip("bs"))
            meta.setdefault("system_environment", {})["batch_size"] = bs
        except (ValueError, AttributeError):
            warnings.warn(f"Cannot determine batch_size for {leaf_dir} — skipping")
            return None, None

    records = []
    with open(records_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return meta, records


def normalize_records(records: list) -> list:
    """Return copies of records with a 'latency' key added.

    _calculate_continuous_metrics expects 'latency'; the stored files use
    'ttft' (prefill) and 'tpot' (decoding). Uses explicit None-check so that
    a ttft of 0.0 is not silently replaced by tpot.
    """
    normalized = []
    for r in records:
        r2 = dict(r)
        r2["latency"] = r["ttft"] if r.get("ttft") is not None else r.get("tpot", 0)
        normalized.append(r2)
    return normalized


def compute_smfu_smbu(records: list, metadata: dict) -> Optional[dict]:
    """Recompute S_MFU/S_MBU from per-step records using MoE-CAP internals.

    Returns dict with prefill_smfu, decoding_smfu, prefill_smbu, decoding_smbu
    (all in percent, 0-100), or None on failure.
    """
    if not records:
        return None

    model_name = metadata["model_config"]["model_name"]
    precision_str = metadata["model_config"].get("precision", "bfloat16")
    num_gpus = metadata["hardware"].get("num_gpus", 1)
    gpu_raw_type = records[0].get("gpu_raw_type",
                                  metadata["hardware"].get("gpu_type"))

    cap_config = CAPConfig(
        dataset_names=[],
        metrics=[],
        model_id=model_name,
        precision=precision_str,
    )
    retriever = HFModelInfoRetriever(cap_config)

    arch = retriever.get_architecture_info()
    moe_info = retriever.get_moe_info()
    attn_info = retriever.get_attention_info()

    n_layers = arch.get("num_hidden_layers")
    d_model = arch.get("hidden_size")
    d_ff = moe_info.get("ffn_dim")
    n_attn_heads = attn_info.get("num_attention_heads")
    n_kv_heads = attn_info.get("num_key_value_heads")
    d_head = attn_info.get("head_dim")
    precision_bytes = retriever.get_model_precision_bytes()
    used_dtype = precision_str

    try:
        result = _calculate_continuous_metrics(
            n_layers=n_layers,
            d_model=d_model,
            gpu_raw_type=gpu_raw_type,
            n_attn_heads=n_attn_heads,
            d_head=d_head,
            n_kv_heads=n_kv_heads,
            d_ff=d_ff,
            hf_config=retriever.hf_config,
            num_gpus=num_gpus,
            model_name=model_name,
            used_dtype=used_dtype,
            precision=precision_bytes,
            output_data=records,
        )
    except KeyError as e:
        warnings.warn(f"Unknown GPU type {e} — skipping")
        return None
    except Exception as e:
        warnings.warn(f"compute_smfu_smbu failed: {e} — skipping")
        return None

    if not result:
        return None

    return {
        "prefill_smfu":   result.get("prefill_smfu", 0) * 100,
        "decoding_smfu":  result.get("decoding_smfu", 0) * 100,
        "prefill_smbu":   result.get("prefill_smbu", 0) * 100,
        "decoding_smbu":  result.get("decoding_smbu", 0) * 100,
    }


def aggregate_results(raw: list) -> dict:
    """Average metrics across datasets for each (slug, batch_size) pair.

    Args:
        raw: list of (slug, batch_size, metrics_dict) tuples

    Returns:
        {slug: {batch_size: {prefill_smfu, decoding_smfu, prefill_smbu, decoding_smbu}}}
    """
    keys = ["prefill_smfu", "decoding_smfu", "prefill_smbu", "decoding_smbu"]
    accum = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for slug, bs, metrics in raw:
        for k in keys:
            accum[slug][bs][k].append(metrics[k])

    result = {}
    for slug, bs_data in accum.items():
        result[slug] = {}
        for bs, kdata in bs_data.items():
            result[slug][bs] = {k: sum(v) / len(v) for k, v in kdata.items()}
    return result


def plot_metric(data: dict, metric_label: str, prefill_key: str,
               decoding_key: str, out_path: Path) -> None:
    """Plot prefill and decoding metric vs batch size for each model.

    Args:
        data: {slug: {batch_size: {prefill_key: float, decoding_key: float}}}
        metric_label: shown in figure title and y-axis label
        prefill_key: key in the inner dict for prefill values
        decoding_key: key in the inner dict for decoding values
        out_path: where to save the PNG
    """
    slugs = sorted(data.keys())
    n = len(slugs)
    if n == 0:
        print(f"No model data to plot for {metric_label}. Exiting.", file=sys.stderr)
        sys.exit(1)

    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    fig.suptitle(metric_label, fontsize=14)

    for idx, slug in enumerate(slugs):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        bs_data = data[slug]
        batch_sizes = sorted(bs_data.keys())
        prefill_vals = [bs_data[bs][prefill_key] for bs in batch_sizes]
        decoding_vals = [bs_data[bs][decoding_key] for bs in batch_sizes]

        ax.plot(batch_sizes, prefill_vals, linestyle="--", label="Prefill")
        ax.plot(batch_sizes, decoding_vals, linestyle="-",  label="Decoding")
        ax.set_xscale("log")
        ax.set_xticks(batch_sizes)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylim(0, 100)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel(f"{metric_label} (%)")
        ax.set_title(slug)

    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    if matplotlib.get_backend() != "Agg":
        plt.show()

    print(f"Saved {out_path}")


def walk_results(results_dir: Path):
    """Yield (slug, bs_dir_name, dataset, leaf_dir) for every leaf directory."""
    for slug_dir in sorted(results_dir.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name
        for bs_dir in sorted(slug_dir.iterdir()):
            if not bs_dir.is_dir():
                continue
            for dataset_dir in sorted(bs_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                yield slug, bs_dir.name, dataset_dir.name, dataset_dir


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python analyze.py <RESULTS_DIR>", file=sys.stderr)
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        print(f"Not a directory: {results_dir}", file=sys.stderr)
        sys.exit(1)

    raw: list = []

    for slug, bs_dir_name, dataset, leaf_dir in walk_results(results_dir):
        meta, records = load_triple(leaf_dir, bs_dir_name)
        if meta is None:
            continue

        normalized = normalize_records(records)
        metrics = compute_smfu_smbu(normalized, meta)
        if metrics is None:
            warnings.warn(f"Skipping {slug} {bs_dir_name} {dataset} — no valid metrics")
            continue

        bs = meta["system_environment"]["batch_size"]
        raw.append((slug, bs, metrics))
        print(f"  {slug} bs={bs} {dataset}: "
              f"decode S-MFU={metrics['decoding_smfu']:.1f}% "
              f"S-MBU={metrics['decoding_smbu']:.1f}%")

    if not raw:
        print("No valid results found.", file=sys.stderr)
        sys.exit(1)

    data = aggregate_results(raw)

    plot_metric(data, "S_MFU", prefill_key="prefill_smfu",
                decoding_key="decoding_smfu",
                out_path=results_dir / "smfu.png")
    plot_metric(data, "S_MBU", prefill_key="prefill_smbu",
                decoding_key="decoding_smbu",
                out_path=results_dir / "smbu.png")


if __name__ == "__main__":
    main()
