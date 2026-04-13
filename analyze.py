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
from moe_cap.utils.hardware_utils import get_peak_flops


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
    'ttft' (prefill) and 'tpot' (decoding). Selection is based on
    forward_mode so that decoding records always use tpot even if ttft
    is present.
    """
    normalized = []
    for r in records:
        r2 = dict(r)
        if r.get("forward_mode") == "prefill":
            r2["latency"] = r.get("ttft") if r.get("ttft") is not None else r.get("tpot", 0)
        else:
            r2["latency"] = r.get("tpot") if r.get("tpot") is not None else r.get("ttft", 0)
        normalized.append(r2)
    return normalized


def _run_metrics(records, model_name, precision_str, num_gpus, gpu_raw_type, cap_config):
    """Run _calculate_continuous_metrics for a given model_name and return the result dict."""
    retriever = HFModelInfoRetriever(cap_config)
    arch = retriever.get_architecture_info()
    moe_info = retriever.get_moe_info()
    attn_info = retriever.get_attention_info()

    try:
        return _calculate_continuous_metrics(
            n_layers=arch.get("num_hidden_layers"),
            d_model=arch.get("hidden_size"),
            gpu_raw_type=gpu_raw_type,
            n_attn_heads=attn_info.get("num_attention_heads"),
            d_head=attn_info.get("head_dim"),
            n_kv_heads=attn_info.get("num_key_value_heads"),
            d_ff=moe_info.get("ffn_dim"),
            hf_config=retriever.hf_config,
            num_gpus=num_gpus,
            model_name=model_name,
            used_dtype=precision_str,
            precision=retriever.get_model_precision_bytes(),
            output_data=records,
        )
    except KeyError as e:
        warnings.warn(f"Unknown GPU type {e} — skipping")
        return None
    except Exception as e:
        warnings.warn(f"compute metrics failed: {e} — skipping")
        return None


def compute_smfu_smbu(records: list, metadata: dict) -> Optional[dict]:
    """Recompute S_MFU/S_MBU from per-step records using MoE-CAP internals.

    Returns dict with prefill_smfu, decoding_smfu, prefill_smbu, decoding_smbu,
    prefill_raw_tflops, decoding_raw_tflops (all S-M* in percent 0-100).
    For Qwen3-Next models, also includes *_legacy keys for comparison.
    Returns None on failure.
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

    # Primary computation (uses correct dispatch for all models including Qwen3-Next)
    result = _run_metrics(records, model_name, precision_str, num_gpus, gpu_raw_type, cap_config)
    if not result:
        return None

    # Compute raw TFLOPS
    try:
        peak_flops_raw = get_peak_flops(gpu_raw_type, precision_str)
        peak_tflops = peak_flops_raw / 1e12
    except KeyError:
        peak_tflops = 0

    prefill_smfu_frac = result.get("prefill_smfu", 0)

    metrics = {
        "prefill_smfu":        prefill_smfu_frac * 100,
        "prefill_smbu":        result.get("prefill_smbu", 0) * 100,
        "prefill_raw_tflops":  prefill_smfu_frac * num_gpus * peak_tflops / 2,
    }

    # For Qwen3-Next: also compute with legacy Qwen3 path for comparison
    if "Qwen3-Next" in model_name:
        legacy_name = model_name.replace("Qwen3-Next", "Qwen3")
        legacy_result = _run_metrics(records, legacy_name, precision_str, num_gpus, gpu_raw_type, cap_config)
        if legacy_result:
            leg_prefill = legacy_result.get("prefill_smfu", 0)
            metrics["prefill_smfu_legacy"] = leg_prefill * 100
            metrics["prefill_smbu_legacy"] = legacy_result.get("prefill_smbu", 0) * 100
            metrics["prefill_raw_tflops_legacy"] = leg_prefill * num_gpus * peak_tflops / 2

    return metrics


def aggregate_results(raw: list) -> dict:
    """Average metrics across datasets for each (slug, batch_size) pair.

    Args:
        raw: list of (slug, batch_size, metrics_dict) tuples

    Returns:
        {slug: {batch_size: {metric_key: averaged_value, ...}}}
    """
    accum = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for slug, bs, metrics in raw:
        for k, v in metrics.items():
            accum[slug][bs][k].append(v)

    result = {}
    for slug, bs_data in accum.items():
        result[slug] = {}
        for bs, kdata in bs_data.items():
            result[slug][bs] = {k: sum(v) / len(v) for k, v in kdata.items()}
    return result


def plot_single_metric(slug: str, bs_data: dict, metric_label: str,
                       prefill_key: str, out_path: Path,
                       prefill_legacy_key: str = None) -> None:
    """Plot prefill metric vs batch size for a single model."""
    batch_sizes = sorted(bs_data.keys())
    if not batch_sizes:
        return

    prefill_vals = [bs_data[bs].get(prefill_key, 0) for bs in batch_sizes]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(batch_sizes, prefill_vals, "o--", label="Prefill")

    # Legacy comparison line (Qwen3-Next only)
    has_legacy = (prefill_legacy_key
                  and any(prefill_legacy_key in bs_data[bs] for bs in batch_sizes))
    if has_legacy:
        leg_prefill = [bs_data[bs].get(prefill_legacy_key, 0) for bs in batch_sizes]
        ax.plot(batch_sizes, leg_prefill, "o:", color="tab:blue", alpha=0.4,
                label="Prefill (Qwen3 legacy)")

    ax.set_xscale("log")
    ax.set_xticks(batch_sizes)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{slug} — {metric_label}")
    ax.legend()

    if "%" in metric_label:
        ax.set_ylim(0, 100)
    else:
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    if matplotlib.get_backend() != "Agg":
        plt.show()

    print(f"Saved {out_path}")


def walk_results(results_dir: Path):
    """Yield (slug, bs_dir_name, dataset, leaf_dir) for every leaf directory.

    The runner writes files two levels below the dataset dir:
      <slug>/bs<N>/<dataset>/<org>/<model_name>/metadata_*.json
    """
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
                for org_dir in sorted(dataset_dir.iterdir()):
                    if not org_dir.is_dir():
                        continue
                    for model_dir in sorted(org_dir.iterdir()):
                        if not model_dir.is_dir():
                            continue
                        yield slug, bs_dir.name, dataset_dir.name, model_dir


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

        # Only keep prefill records — decode is meaningless with target_output_tokens=1
        prefill_records = [r for r in records if r.get("forward_mode") == "prefill"]
        if not prefill_records:
            warnings.warn(f"No prefill records in {slug} {bs_dir_name} {dataset} — skipping")
            continue
        normalized = normalize_records(prefill_records)
        metrics = compute_smfu_smbu(normalized, meta)
        if metrics is None:
            warnings.warn(f"Skipping {slug} {bs_dir_name} {dataset} — no valid metrics")
            continue

        bs = meta["system_environment"]["batch_size"]
        raw.append((slug, bs, metrics))
        print(f"  {slug} bs={bs} {dataset}: "
              f"prefill S-MFU={metrics['prefill_smfu']:.1f}% "
              f"S-MBU={metrics['prefill_smbu']:.1f}% "
              f"TFLOPS={metrics['prefill_raw_tflops']:.1f}")
        if "prefill_smfu_legacy" in metrics:
            print(f"    legacy: "
                  f"S-MFU={metrics['prefill_smfu_legacy']:.1f}% "
                  f"S-MBU={metrics['prefill_smbu_legacy']:.1f}% "
                  f"TFLOPS={metrics['prefill_raw_tflops_legacy']:.1f}")

    if not raw:
        print("No valid results found.", file=sys.stderr)
        sys.exit(1)

    data = aggregate_results(raw)

    for slug in sorted(data.keys()):
        bs_data = data[slug]

        plot_single_metric(
            slug, bs_data, "Prefill S-MFU (%)",
            "prefill_smfu",
            results_dir / f"smfu_{slug}.png",
            "prefill_smfu_legacy",
        )
        plot_single_metric(
            slug, bs_data, "Prefill S-MBU (%)",
            "prefill_smbu",
            results_dir / f"smbu_{slug}.png",
            "prefill_smbu_legacy",
        )
        plot_single_metric(
            slug, bs_data, "Prefill TFLOPS",
            "prefill_raw_tflops",
            results_dir / f"raw_flops_{slug}.png",
            "prefill_raw_tflops_legacy",
        )


if __name__ == "__main__":
    main()
