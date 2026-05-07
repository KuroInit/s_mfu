#!/usr/bin/env python3
"""Post-processing script: recomputes S_MFU/S_MBU from sweep results and plots them."""

import glob
import csv
import json
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
import yaml

from moe_cap.configs import CAPConfig
from moe_cap.model_loader import HFModelInfoRetriever
from moe_cap.utils.continuous_batching_utils import _calculate_continuous_metrics
from moe_cap.utils.hardware_utils import get_peak_flops

UNKNOWN_GPU_VALUES = {"", "unknown", "none", "null"}


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
    records_file = find_latest_file(leaf_dir, "server_records_*.jsonl")
    if records_file is None:
        records_file = find_latest_file(leaf_dir, "detailed_results_*.jsonl")

    if meta_file is None:
        warnings.warn(f"No metadata file in {leaf_dir} — skipping")
        return None, None
    if records_file is None:
        warnings.warn(f"No server_records or detailed_results file in {leaf_dir} — skipping")
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
            if r.get("ttft") is not None:
                r2["latency"] = r.get("ttft")
            else:
                r2["latency"] = r.get("latency", r.get("tpot", 0))
        else:
            if r.get("tpot") is not None:
                r2["latency"] = r.get("tpot")
            else:
                r2["latency"] = r.get("latency", r.get("ttft", 0))
        normalized.append(r2)
    return normalized


def _is_known_gpu_value(value) -> bool:
    return value is not None and str(value).strip().lower() not in UNKNOWN_GPU_VALUES


def resolve_gpu_raw_type(records: list, metadata: dict) -> Optional[str]:
    """Pick the GPU type used for MoE-CAP hardware lookup.

    Older server results sometimes store "Unknown" in each detailed record even
    when metadata has the real GPU. ANALYZE_GPU_TYPE is an explicit escape hatch
    for analyzing such historical runs without editing result files.
    """
    candidates = [
        os.environ.get("ANALYZE_GPU_TYPE"),
        records[0].get("gpu_raw_type") if records else None,
        metadata.get("hardware", {}).get("gpu_type"),
    ]
    for candidate in candidates:
        if _is_known_gpu_value(candidate):
            return str(candidate)
    for candidate in candidates:
        if candidate is not None:
            return str(candidate)
    return None


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
    """Compute prefill metrics using MoE-CAP wherever possible.

    MoE-CAP is the ground truth for S-MFU, S-MBU, prefill throughput
    (`prefill_tp`), and decoding throughput (`decoding_throughput`).
    raw_tflops is reconstructed from MoE-CAP S-MFU × MoE-CAP peak dense FLOPS.

    Returns None on failure; all S-M* are in percent 0-100.
    """
    if not records:
        return None

    model_name = metadata["model_config"]["model_name"]
    precision_str = metadata["model_config"].get("precision", "bfloat16")
    num_gpus = metadata["hardware"].get("num_gpus", 1)
    gpu_raw_type = resolve_gpu_raw_type(records, metadata)

    cap_config = CAPConfig(
        dataset_names=[],
        metrics=[],
        model_id=model_name,
        precision=precision_str,
    )

    result = _run_metrics(records, model_name, precision_str, num_gpus, gpu_raw_type, cap_config)
    if not result:
        return None

    try:
        peak_flops_raw = get_peak_flops(gpu_raw_type, precision_str)
        peak_tflops_sparse = peak_flops_raw / 1e12  # 2:4 sparse (e.g. H100 HBM3 = 1979)
    except KeyError:
        peak_tflops_sparse = 0
    peak_tflops_dense = peak_tflops_sparse / 2
    prefill_smfu_frac = result.get("prefill_smfu", 0)
    decoding_smfu_frac = result.get("decoding_smfu", 0)
    moe_cap_prefill_tps = result.get("prefill_tp")
    if moe_cap_prefill_tps is None:
        moe_cap_prefill_tps = 0
    moe_cap_decoding_tps = result.get("decoding_throughput", 0)

    metrics = {
        "prefill_tokens_per_sec": moe_cap_prefill_tps,
        "decoding_tokens_per_sec": moe_cap_decoding_tps,
        "prefill_raw_tflops":     prefill_smfu_frac * num_gpus * peak_tflops_dense,
        "decoding_raw_tflops":    decoding_smfu_frac * num_gpus * peak_tflops_dense,
        "prefill_smfu":           prefill_smfu_frac * 100,
        "prefill_smbu":           result.get("prefill_smbu", 0) * 100,
        "decoding_smfu":          decoding_smfu_frac * 100,
        "decoding_smbu":          result.get("decoding_smbu", 0) * 100,
        "ttft":                   result.get("ttft", 0),
        "tpot":                   result.get("tpot", 0),
        "num_gpus": num_gpus,
    }

    # For Qwen3-Next: also compute with legacy Qwen3 path for comparison
    if "Qwen3-Next" in model_name:
        legacy_name = model_name.replace("Qwen3-Next", "Qwen3")
        legacy_result = _run_metrics(records, legacy_name, precision_str, num_gpus, gpu_raw_type, cap_config)
        if legacy_result:
            leg_prefill = legacy_result.get("prefill_smfu", 0)
            metrics["prefill_raw_tflops_legacy"] = leg_prefill * num_gpus * peak_tflops_dense
            metrics["prefill_smfu_legacy"]       = leg_prefill * 100
            metrics["prefill_smbu_legacy"]       = legacy_result.get("prefill_smbu", 0) * 100
            leg_decode = legacy_result.get("decoding_smfu", 0)
            metrics["decoding_raw_tflops_legacy"] = leg_decode * num_gpus * peak_tflops_dense
            metrics["decoding_smfu_legacy"]       = leg_decode * 100
            metrics["decoding_smbu_legacy"]       = legacy_result.get("decoding_smbu", 0) * 100

    return metrics


def _copy_run_metadata(metrics: dict, metadata: dict, dataset_cfg: dict) -> None:
    """Attach run-shaping metadata that affects result interpretation."""
    model_cfg = metadata.get("model_config", {})
    system = metadata.get("system_environment", {})

    metrics["runner_mode"] = system.get("runner", "moe_cap.openai_api_profile")
    metrics["inference_engine"] = system.get("inference_engine", "")
    metrics["num_prompts"] = system.get("num_prompts", "")
    metrics["client_success_count"] = system.get("client_success_count", "")
    metrics["client_fail_count"] = system.get("client_fail_count", "")

    for key in (
        "chunked_prefill_size",
        "max_prefill_tokens",
        "mem_fraction_static",
        "target_output_tokens",
    ):
        value = model_cfg.get(key)
        if value is None:
            value = dataset_cfg.get(key)
        if value is not None:
            metrics[key] = value

    disable_radix = system.get("disable_radix_cache")
    if disable_radix is not None:
        metrics["disable_radix_cache"] = disable_radix


def _aggregate_metric_values(values: list):
    """Average numeric duplicate cells; carry stable metadata values through."""
    numeric = [
        v for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    if len(numeric) == len(values):
        return sum(numeric) / len(numeric)

    non_empty = [v for v in values if v not in ("", None)]
    if not non_empty:
        return ""
    first = non_empty[0]
    if all(v == first for v in non_empty):
        return first
    return ";".join(str(v) for v in dict.fromkeys(non_empty))


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
            result[slug][bs] = {k: _aggregate_metric_values(v) for k, v in kdata.items()}
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


def aggregate_by_dataset(raw: list) -> dict:
    """Group metrics by dataset and model for per-dataset plotting.

    Args:
        raw: list of (slug, batch_size, dataset, metrics_dict) tuples.
             Duplicate (slug, bs, dataset) entries are averaged.

    Returns:
        {dataset: {slug: {batch_size: {metric_key: value}}}}
    """
    accum: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for slug, bs, dataset, metrics in raw:
        for k, v in metrics.items():
            accum[dataset][slug][bs][k].append(v)

    result: dict = {}
    for dataset, slug_data in accum.items():
        result[dataset] = {}
        for slug, bs_data in slug_data.items():
            result[dataset][slug] = {}
            for bs, kdata in bs_data.items():
                result[dataset][slug][bs] = {
                    k: _aggregate_metric_values(v) for k, v in kdata.items()
                }
    return result


def plot_metric_per_dataset(dataset: str, per_slug_bs_data: dict,
                            metric_label: str, metric_key: str, out_path: Path,
                            legacy_key: str = None) -> None:
    """Plot one metric vs batch size for every model, on a single figure.

    Args:
        per_slug_bs_data: {slug: {bs: {metric_key: value}}}
    """
    slugs = sorted(per_slug_bs_data.keys())
    if not slugs:
        return

    all_bs = sorted({bs for bs_data in per_slug_bs_data.values() for bs in bs_data.keys()})
    if not all_bs:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, slug in enumerate(slugs):
        bs_data = per_slug_bs_data[slug]
        bss = sorted(bs_data.keys())
        vals = [bs_data[bs].get(metric_key, 0) for bs in bss]
        color = color_cycle[i % len(color_cycle)]
        ax.plot(bss, vals, "o-", color=color, label=slug)

        if legacy_key and any(legacy_key in bs_data[bs] for bs in bss):
            leg_vals = [bs_data[bs].get(legacy_key, 0) for bs in bss]
            ax.plot(bss, leg_vals, "o:", color=color, alpha=0.4,
                    label=f"{slug} (legacy)")

    ax.set_xscale("log")
    ax.set_xticks(all_bs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{dataset} — {metric_label}")
    ax.legend(loc="best", fontsize="small")

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


def plot_smfu_smbu_for_model(slug: str, dataset: str, bs_data: dict, out_path: Path) -> None:
    """Plot S-MFU and S-MBU vs batch size for one model on one figure."""
    batch_sizes = sorted(bs_data.keys())
    if not batch_sizes:
        return

    smfu_vals = [bs_data[bs].get("prefill_smfu", 0) for bs in batch_sizes]
    smbu_vals = [bs_data[bs].get("prefill_smbu", 0) for bs in batch_sizes]
    y_vals = [v for v in smfu_vals + smbu_vals if isinstance(v, (int, float))]
    if not y_vals:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(batch_sizes, smfu_vals, "o-", color="tab:blue", label="S-MFU")
    ax.plot(batch_sizes, smbu_vals, "s-", color="tab:orange", label="S-MBU")

    if any("prefill_smfu_legacy" in bs_data[bs] for bs in batch_sizes):
        legacy_smfu = [bs_data[bs].get("prefill_smfu_legacy", 0) for bs in batch_sizes]
        ax.plot(batch_sizes, legacy_smfu, "o:", color="tab:blue", alpha=0.45,
                label="S-MFU (legacy)")
        y_vals.extend(legacy_smfu)

    if any("prefill_smbu_legacy" in bs_data[bs] for bs in batch_sizes):
        legacy_smbu = [bs_data[bs].get("prefill_smbu_legacy", 0) for bs in batch_sizes]
        ax.plot(batch_sizes, legacy_smbu, "s:", color="tab:orange", alpha=0.45,
                label="S-MBU (legacy)")
        y_vals.extend(legacy_smbu)

    ax.set_xscale("log")
    ax.set_xticks(batch_sizes)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Utilization (%)")
    ax.set_title(f"{slug} — S-MFU / S-MBU — {dataset}")
    ax.set_ylim(0, max(y_vals) + 20)
    ax.legend(loc="best")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_legacy_comparison(slug: str, dataset: str, bs_data: dict,
                           metric_label: str, current_key: str, legacy_key: str,
                           out_path: Path) -> None:
    """Plot current vs legacy calculation for a single model on one figure.

    Both lines solid (unlike the dotted legacy backdrop on combined plots) so
    the two paths can be compared directly.
    """
    batch_sizes = sorted(bs_data.keys())
    if not batch_sizes:
        return
    if not any(legacy_key in bs_data[bs] for bs in batch_sizes):
        return

    current_vals = [bs_data[bs].get(current_key, 0) for bs in batch_sizes]
    legacy_vals = [bs_data[bs].get(legacy_key, 0) for bs in batch_sizes]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(batch_sizes, current_vals, "o-", color="tab:blue", label="Current (Qwen3-Next path)")
    ax.plot(batch_sizes, legacy_vals, "s-", color="tab:orange", label="Legacy (Qwen3 path)")

    ax.set_xscale("log")
    ax.set_xticks(batch_sizes)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{slug} — {metric_label} — {dataset}")
    ax.legend(loc="best")

    if "%" in metric_label:
        ax.set_ylim(0, 100)
    else:
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def write_raw_values(raw: list, out_path: Path) -> None:
    """Write every computed metric cell to CSV.

    Args:
        raw: list of (slug, batch_size, dataset, metrics_dict) tuples.
    """
    metric_order = [
        # Run-shaping metadata
        "runner_mode", "inference_engine", "num_prompts",
        "client_success_count", "client_fail_count",
        "chunked_prefill_size", "max_prefill_tokens", "mem_fraction_static",
        "target_output_tokens",
        "disable_radix_cache",
        # MoE-CAP metrics and direct derivatives
        "prefill_tokens_per_sec", "decoding_tokens_per_sec",
        "prefill_raw_tflops", "decoding_raw_tflops",
        "prefill_smfu", "prefill_smbu", "decoding_smfu", "decoding_smbu",
        "ttft", "tpot", "num_gpus",
        # Legacy Qwen3-path derivations (Qwen3-Next only)
        "prefill_raw_tflops_legacy", "decoding_raw_tflops_legacy",
        "prefill_smfu_legacy", "prefill_smbu_legacy",
        "decoding_smfu_legacy", "decoding_smbu_legacy",
    ]

    rows = sorted(raw, key=lambda t: (t[2], t[0], t[1]))
    keys_present = [
        k for k in metric_order
        if any(k in metrics for _, _, _, metrics in rows)
    ]
    fieldnames = ["dataset", "slug", "batch_size"] + keys_present

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for slug, bs, dataset, metrics in rows:
            row = {"dataset": dataset, "slug": slug, "batch_size": bs}
            for k in keys_present:
                row[k] = metrics.get(k, "")
            writer.writerow(row)
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


def load_dataset_config_for_result(repo_dir: Path, dataset: str, slug: str) -> dict:
    """Best-effort load of configs/<dataset>_<slug>.yaml for result metadata."""
    config_path = repo_dir / "configs" / f"{dataset}_{slug}.yaml"
    if not config_path.exists():
        return {}
    try:
        return yaml.safe_load(config_path.read_text()) or {}
    except Exception as exc:
        warnings.warn(f"Could not read dataset config from {config_path}: {exc}")
        return {}


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

        prefill_records = [r for r in records if r.get("forward_mode") == "prefill"]
        if not prefill_records:
            warnings.warn(f"No prefill records in {slug} {bs_dir_name} {dataset} — skipping")
            continue
        normalized = normalize_records(records)
        metrics = compute_smfu_smbu(normalized, meta)
        if metrics is None:
            warnings.warn(f"Skipping {slug} {bs_dir_name} {dataset} — no valid metrics")
            continue

        repo_dir = Path(__file__).resolve().parent
        dataset_cfg = load_dataset_config_for_result(repo_dir, dataset, slug)
        _copy_run_metadata(metrics, meta, dataset_cfg)

        bs = meta["system_environment"]["batch_size"]
        raw.append((slug, bs, dataset, metrics))
        print(f"  {slug} bs={bs} {dataset}: "
              f"tok/s={metrics['prefill_tokens_per_sec']:.0f} "
              f"TFLOPS={metrics['prefill_raw_tflops']:.1f} "
              f"S-MFU={metrics['prefill_smfu']:.1f}% "
              f"S-MBU={metrics['prefill_smbu']:.1f}%")
        if metrics.get("decoding_tokens_per_sec", 0):
            print(f"    decoding: tok/s={metrics['decoding_tokens_per_sec']:.0f} "
                  f"TFLOPS={metrics['decoding_raw_tflops']:.1f} "
                  f"S-MFU={metrics['decoding_smfu']:.1f}% "
                  f"S-MBU={metrics['decoding_smbu']:.1f}% "
                  f"TPOT={metrics.get('tpot', 0):.4f}s")
        if "prefill_smfu_legacy" in metrics:
            print(f"    legacy: "
                  f"TFLOPS={metrics['prefill_raw_tflops_legacy']:.1f} "
                  f"S-MFU={metrics['prefill_smfu_legacy']:.1f}% "
                  f"S-MBU={metrics['prefill_smbu_legacy']:.1f}%")

    if not raw:
        print("No valid results found.", file=sys.stderr)
        sys.exit(1)

    # Raw-value dump first — always emitted, even if plotting fails.
    write_raw_values(raw, results_dir / "raw_values.csv")

    # One figure per (dataset, metric) with every model drawn as a line.
    per_dataset = aggregate_by_dataset(raw)
    for dataset in sorted(per_dataset.keys()):
        per_slug = per_dataset[dataset]
        plot_metric_per_dataset(
            dataset, per_slug, "Prefill S-MFU (%)", "prefill_smfu",
            results_dir / f"smfu_{dataset}.png",
            legacy_key="prefill_smfu_legacy",
        )
        plot_metric_per_dataset(
            dataset, per_slug, "Prefill S-MBU (%)", "prefill_smbu",
            results_dir / f"smbu_{dataset}.png",
            legacy_key="prefill_smbu_legacy",
        )
        plot_metric_per_dataset(
            dataset, per_slug, "Decoding S-MFU (%)", "decoding_smfu",
            results_dir / f"decoding_smfu_{dataset}.png",
            legacy_key="decoding_smfu_legacy",
        )
        plot_metric_per_dataset(
            dataset, per_slug, "Decoding S-MBU (%)", "decoding_smbu",
            results_dir / f"decoding_smbu_{dataset}.png",
            legacy_key="decoding_smbu_legacy",
        )
        plot_metric_per_dataset(
            dataset, per_slug, "Prefill TFLOPS", "prefill_raw_tflops",
            results_dir / f"raw_flops_{dataset}.png",
            legacy_key="prefill_raw_tflops_legacy",
        )
        plot_metric_per_dataset(
            dataset, per_slug, "Decoding TFLOPS", "decoding_raw_tflops",
            results_dir / f"decoding_raw_flops_{dataset}.png",
            legacy_key="decoding_raw_tflops_legacy",
        )
        plot_metric_per_dataset(
            dataset, per_slug, "Prefill tokens/sec", "prefill_tokens_per_sec",
            results_dir / f"tokens_per_sec_{dataset}.png",
        )
        plot_metric_per_dataset(
            dataset, per_slug, "Decoding tokens/sec", "decoding_tokens_per_sec",
            results_dir / f"decoding_tokens_per_sec_{dataset}.png",
        )
        for slug, bs_data in sorted(per_slug.items()):
            plot_smfu_smbu_for_model(
                slug, dataset, bs_data,
                results_dir / f"{slug}_smfu_smbu_{dataset}.png",
            )

        # Dedicated qwen3_next_80b current-vs-legacy comparison (one fig per metric).
        qn_slug = "qwen3_next_80b"
        if qn_slug in per_slug:
            qn_bs = per_slug[qn_slug]
            plot_legacy_comparison(
                qn_slug, dataset, qn_bs, "Prefill S-MFU (%)",
                "prefill_smfu", "prefill_smfu_legacy",
                results_dir / f"{qn_slug}_legacy_smfu_{dataset}.png",
            )
            plot_legacy_comparison(
                qn_slug, dataset, qn_bs, "Prefill S-MBU (%)",
                "prefill_smbu", "prefill_smbu_legacy",
                results_dir / f"{qn_slug}_legacy_smbu_{dataset}.png",
            )
            plot_legacy_comparison(
                qn_slug, dataset, qn_bs, "Prefill TFLOPS",
                "prefill_raw_tflops", "prefill_raw_tflops_legacy",
                results_dir / f"{qn_slug}_legacy_raw_flops_{dataset}.png",
            )


if __name__ == "__main__":
    main()
