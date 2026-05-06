#!/usr/bin/env python3
"""Post-processing script: recomputes S_MFU/S_MBU from sweep results and plots them."""

import glob
import csv
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
import yaml

from moe_cap.configs import CAPConfig
from moe_cap.model_loader import HFModelInfoRetriever
from moe_cap.utils.continuous_batching_utils import _calculate_continuous_metrics
from moe_cap.utils.hardware_utils import get_peak_bw, get_peak_flops

from sglang_metrics import (
    load_snapshots,
    server_tokens_per_sec,
    peak_running_reqs,
    peak_cache_hit_rate,
)

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


def _aggregate_raw_throughput(records: list) -> dict:
    """Sum tokens and time across prefill records. First-order primitives.

    Returns {total_tokens, total_latency, tokens_per_sec}, or zeros if empty.
    This is *raw* in the strict sense — only division of two measured quantities.
    """
    total_tokens = sum(r.get("seq_lens_sum", 0) for r in records)
    total_latency = sum(r.get("latency", 0) for r in records)
    tps = (total_tokens / total_latency) if total_latency > 0 else 0
    return {
        "total_tokens":  total_tokens,
        "total_latency": total_latency,
        "tokens_per_sec": tps,
    }


def _prefill_shape_metrics(records: list) -> dict:
    """Summarize the actual prefill forward-pass shape recorded by SGLang.

    For a batched prefill roofline, the relevant X coordinate is the amount of
    prefill work in the forward passes SGLang actually executed, not the sweep
    label. `seq_lens_sum` is already the packed B*S token mass for each prefill
    step, so a token-weighted average gives one representative roofline point
    for the run while de-emphasizing the final partial wave.
    """
    if not records:
        return {}

    batch_sizes = []
    seq_sums = []
    for r in records:
        try:
            rb = int(r.get("batch_size") or 0)
            seq = int(r.get("seq_lens_sum") or 0)
        except (TypeError, ValueError):
            continue
        if rb > 0:
            batch_sizes.append(rb)
        if seq > 0:
            seq_sums.append(seq)

    metrics = {"prefill_record_count": len(records)}
    if batch_sizes:
        metrics["prefill_actual_max_batch_size"] = max(batch_sizes)
        metrics["prefill_actual_avg_batch_size"] = sum(batch_sizes) / len(batch_sizes)
    if seq_sums:
        total_seq = sum(seq_sums)
        metrics["prefill_actual_max_seq_lens_sum"] = max(seq_sums)
        metrics["prefill_actual_avg_seq_lens_sum"] = total_seq / len(seq_sums)
        metrics["prefill_actual_token_weighted_seq_lens_sum"] = (
            sum(seq * seq for seq in seq_sums) / total_seq
        )
    return metrics


def _scale_tflops_with_server_tps(metrics: dict, server_tps: float) -> None:
    """Use server token throughput to correct achieved TFLOPS timing.

    MoE-CAP supplies the per-token FLOP model, but its per-forward timing can be
    wrong for chunked prefill. Scaling by server/client token throughput keeps
    the FLOP accounting while replacing the bad time denominator.
    """
    client_tps = metrics.get("prefill_tokens_per_sec", 0)
    raw_tflops = metrics.get("prefill_raw_tflops", 0)
    if not server_tps or not client_tps or not raw_tflops:
        return

    scale = server_tps / client_tps
    metrics["prefill_roofline_tflops"] = raw_tflops * scale

    legacy_tflops = metrics.get("prefill_raw_tflops_legacy")
    if legacy_tflops is not None:
        metrics["prefill_roofline_tflops_legacy"] = legacy_tflops * scale


def _finalize_roofline_metrics(metrics: dict, batch_size: int, target_input_tokens: Optional[int]) -> None:
    """Add roofline coordinates and MFU/MBU readings to a metrics dict.

    Roofline X uses the recorded prefill forward-pass token mass when present:
    AI ~= actual seq_lens_sum. This is the physical B*S that SGLang packed,
    not just the sweep label. For older records without shape data, it falls
    back to batch_size * target_input_tokens.

    Roofline Y is achieved dense-equivalent TFLOPS/s per GPU. If Tier-5 server
    throughput is available, Y is the server-corrected value; otherwise it
    falls back to the MoE-CAP raw TFLOPS value.
    """
    if not target_input_tokens or target_input_tokens <= 0:
        return

    peak_dense = metrics.get("peak_dense_tflops_per_gpu", 0)
    peak_bw = metrics.get("peak_memory_tbps_per_gpu", 0)
    num_gpus = metrics.get("num_gpus", 1)
    if not peak_dense or not peak_bw or not num_gpus:
        return

    actual_intensity = metrics.get("prefill_actual_token_weighted_seq_lens_sum", 0)
    intensity = actual_intensity if actual_intensity and actual_intensity > 0 else (
        batch_size * target_input_tokens
    )
    achieved = metrics.get("prefill_roofline_tflops")
    if achieved is None:
        achieved = metrics.get("prefill_raw_tflops", 0)

    achieved_per_gpu = achieved / num_gpus
    achieved_bw = (achieved_per_gpu / intensity) if intensity > 0 else 0

    metrics["prefill_target_input_tokens"] = target_input_tokens
    metrics["prefill_arithmetic_intensity"] = intensity
    metrics["prefill_intended_arithmetic_intensity"] = batch_size * target_input_tokens
    metrics["prefill_roofline_tflops_total"] = achieved
    metrics["prefill_roofline_tflops_per_gpu"] = achieved_per_gpu
    metrics["prefill_roofline_smfu"] = (achieved_per_gpu / peak_dense * 100) if peak_dense else 0
    metrics["prefill_roofline_bandwidth_tbps"] = achieved_bw
    metrics["prefill_roofline_smbu"] = (achieved_bw / peak_bw * 100) if peak_bw else 0

    legacy = metrics.get("prefill_roofline_tflops_legacy")
    if legacy is not None:
        legacy_per_gpu = legacy / num_gpus
        legacy_bw = legacy_per_gpu / intensity
        metrics["prefill_roofline_tflops_total_legacy"] = legacy
        metrics["prefill_roofline_tflops_per_gpu_legacy"] = legacy_per_gpu
        metrics["prefill_roofline_smfu_legacy"] = (
            legacy_per_gpu / peak_dense * 100 if peak_dense else 0
        )
        metrics["prefill_roofline_bandwidth_tbps_legacy"] = legacy_bw
        metrics["prefill_roofline_smbu_legacy"] = (
            legacy_bw / peak_bw * 100 if peak_bw else 0
        )


def _merge_tier5(metrics: dict, snaps: list, batch_size: int) -> None:
    """Augment metrics dict with server-side cross-checks from /metrics snapshots.

    Adds:
      - server_tokens_per_sec: Δprompt_tokens_total / Δt (monotonic counter).
      - client_vs_server_delta_pct: relative divergence vs client aggregate.
      - peak_running_reqs: max num_running_reqs observed.
      - peak_cache_hit_rate: max cache_hit_rate observed.
    Emits a warning if the client/server throughputs disagree by >5 %, or if
    peak_running_reqs exceeds batch_size+1 (serial-wave contract violated),
    or if cache_hit_rate is materially non-zero (contamination).
    """
    if not snaps:
        return
    server_tps = server_tokens_per_sec(snaps)
    peak_run = peak_running_reqs(snaps)
    peak_cache = peak_cache_hit_rate(snaps)
    metrics["server_tokens_per_sec"] = server_tps if server_tps is not None else 0
    metrics["peak_running_reqs"] = peak_run if peak_run is not None else 0
    metrics["peak_cache_hit_rate"] = peak_cache if peak_cache is not None else 0

    client_tps = metrics.get("prefill_tokens_per_sec", 0)
    if server_tps and client_tps:
        delta_pct = abs(client_tps - server_tps) / server_tps * 100
        metrics["client_vs_server_delta_pct"] = delta_pct
        _scale_tflops_with_server_tps(metrics, server_tps)
        if delta_pct > 5:
            warnings.warn(
                f"Tier 5: client tok/s={client_tps:.0f} vs server tok/s={server_tps:.0f} "
                f"— {delta_pct:.1f}% divergence (>5%)"
            )
    if peak_run is not None and peak_run > batch_size + 1:
        warnings.warn(
            f"Tier 5: peak_running_reqs={peak_run:.0f} > batch_size+1={batch_size+1} — "
            f"serial-wave contract violated (overlap still present)"
        )
    if peak_cache is not None and peak_cache > 0.05:
        warnings.warn(
            f"Tier 5: peak_cache_hit_rate={peak_cache:.3f} — prefix-cache contamination"
        )


def compute_smfu_smbu(records: list, metadata: dict) -> Optional[dict]:
    """Compute prefill metrics using MoE-CAP wherever possible.

    MoE-CAP is the ground truth for S-MFU, S-MBU, and prefill throughput
    (`prefill_tp`). analyze.py keeps aggregate token/time primitives as
    cross-checks because MoE-CAP does not expose those totals directly.
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

    raw = _aggregate_raw_throughput(records)

    result = _run_metrics(records, model_name, precision_str, num_gpus, gpu_raw_type, cap_config)
    if not result:
        return None

    try:
        peak_flops_raw = get_peak_flops(gpu_raw_type, precision_str)
        peak_tflops_sparse = peak_flops_raw / 1e12  # 2:4 sparse (e.g. H100 HBM3 = 1979)
    except KeyError:
        peak_tflops_sparse = 0
    peak_tflops_dense = peak_tflops_sparse / 2
    peak_memory_tbps = get_peak_bw(gpu_raw_type) / 1e12 if gpu_raw_type else 0

    prefill_smfu_frac = result.get("prefill_smfu", 0)
    moe_cap_prefill_tps = result.get("prefill_tp")
    if moe_cap_prefill_tps is None:
        moe_cap_prefill_tps = raw["tokens_per_sec"]

    metrics = {
        "prefill_tokens_per_sec": moe_cap_prefill_tps,
        "prefill_tokens_per_sec_aggregate": raw["tokens_per_sec"],
        "prefill_total_tokens":   raw["total_tokens"],
        "prefill_total_latency":  raw["total_latency"],
        "prefill_raw_tflops":     prefill_smfu_frac * num_gpus * peak_tflops_dense,
        "prefill_smfu":           prefill_smfu_frac * 100,
        "prefill_smbu":           result.get("prefill_smbu", 0) * 100,
        "peak_dense_tflops_per_gpu": peak_tflops_dense,
        "peak_memory_tbps_per_gpu": peak_memory_tbps,
        "num_gpus": num_gpus,
    }
    metrics.update(_prefill_shape_metrics(records))

    # For Qwen3-Next: also compute with legacy Qwen3 path for comparison
    if "Qwen3-Next" in model_name:
        legacy_name = model_name.replace("Qwen3-Next", "Qwen3")
        legacy_result = _run_metrics(records, legacy_name, precision_str, num_gpus, gpu_raw_type, cap_config)
        if legacy_result:
            leg_prefill = legacy_result.get("prefill_smfu", 0)
            metrics["prefill_raw_tflops_legacy"] = leg_prefill * num_gpus * peak_tflops_dense
            metrics["prefill_smfu_legacy"]       = leg_prefill * 100
            metrics["prefill_smbu_legacy"]       = legacy_result.get("prefill_smbu", 0) * 100

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
                result[dataset][slug][bs] = {k: sum(v) / len(v) for k, v in kdata.items()}
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


def _positive(values):
    return [v for v in values if isinstance(v, (int, float)) and v > 0]


def plot_roofline_per_dataset(dataset: str, per_slug_bs_data: dict, out_path: Path) -> None:
    """Plot prefill roofline: arithmetic intensity vs achieved TFLOPS.

    Points use MoE-CAP's prefill S-MFU-derived TFLOPS. S-MFU and S-MBU are
    shown as readings against the compute and memory roofs.
    """
    points = []
    for slug, bs_data in sorted(per_slug_bs_data.items()):
        for bs, metrics in sorted(bs_data.items()):
            x = metrics.get("prefill_arithmetic_intensity", 0)
            num_gpus = metrics.get("num_gpus", 1)
            y = metrics.get("prefill_raw_tflops", 0) / num_gpus if num_gpus else 0
            if x > 0 and y > 0:
                points.append((slug, bs, x, y, metrics))

    if not points:
        return

    peak_dense = max(
        _positive([
            m.get("peak_dense_tflops_per_gpu", 0)
            for _, _, _, _, m in points
        ]),
        default=0,
    )
    peak_bw = max(
        _positive([
            m.get("peak_memory_tbps_per_gpu", 0)
            for _, _, _, _, m in points
        ]),
        default=0,
    )
    if not peak_dense or not peak_bw:
        return

    xs = [p[2] for p in points]
    ys = [p[3] for p in points]
    ridge = peak_dense / peak_bw
    x_min_data = min(xs)
    x_max_data = max(xs)
    y_min_data = min(ys)
    y_max_data = max(ys)

    # Pad log axes multiplicatively and include the ridge only when doing so
    # does not dwarf the actual measured points.
    x_min = x_min_data / 1.5
    x_max = x_max_data * 1.5
    if x_min / 10 <= ridge <= x_max * 10:
        x_min = min(x_min, ridge / 1.5)
        x_max = max(x_max, ridge * 1.5)

    log_x_min = math.log10(x_min)
    log_x_max = math.log10(x_max)
    roof_x = [
        10 ** (log_x_min + (log_x_max - log_x_min) * i / 99)
        for i in range(100)
    ]
    roof_y = [min(peak_dense, peak_bw * x) for x in roof_x]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(roof_x, roof_y, color="black", linewidth=2.0, label="Roofline")
    if x_min <= ridge <= x_max:
        ax.axvline(ridge, color="0.45", linestyle=":", linewidth=1.2,
                   label=f"Ridge ~{ridge:.0f} FLOPs/byte")

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, slug in enumerate(sorted(per_slug_bs_data.keys())):
        slug_points = [p for p in points if p[0] == slug]
        if not slug_points:
            continue
        color = color_cycle[i % len(color_cycle)]
        slug_points.sort(key=lambda p: p[1])
        ax.plot(
            [p[2] for p in slug_points],
            [p[3] for p in slug_points],
            "o-",
            color=color,
            label=slug,
        )
        for _, bs, x, y, metrics in slug_points:
            smfu = metrics.get("prefill_smfu", 0)
            smbu = metrics.get("prefill_smbu", 0)
            ax.annotate(
                f"bs{bs}\nS-MFU {smfu:.1f}%\nS-MBU {smbu:.1f}%",
                xy=(x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                color=color,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOPs/byte, B x S projection)")
    ax.set_ylabel("Achieved prefill performance per GPU (TFLOPs/s)")
    ax.set_title(f"{dataset} — Prefill Roofline")
    ax.set_xlim(x_min, x_max)
    y_floor = max(y_min_data / 1.5, 1e-6)
    y_top = y_max_data * 1.5
    if y_floor <= peak_dense <= y_top * 10:
        y_top = max(y_top, peak_dense * 1.5)
    ax.set_ylim(y_floor, y_top)
    secax = ax.secondary_yaxis(
        "right",
        functions=(
            lambda tflops: tflops / peak_dense * 100,
            lambda pct: pct / 100 * peak_dense,
        ),
    )
    secax.set_ylabel("S-MFU equivalent (%)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.legend(loc="best", fontsize="small")

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
        # MoE-CAP throughput plus analyze.py aggregate cross-checks
        "prefill_tokens_per_sec", "prefill_tokens_per_sec_aggregate",
        "prefill_total_tokens", "prefill_total_latency",
        # Derived
        "prefill_raw_tflops", "prefill_smfu", "prefill_smbu",
        # Actual SGLang prefill shape used to validate the batch sweep.
        "prefill_record_count", "prefill_actual_max_batch_size",
        "prefill_actual_avg_batch_size", "prefill_actual_max_seq_lens_sum",
        "prefill_actual_avg_seq_lens_sum",
        "prefill_actual_token_weighted_seq_lens_sum",
        # Roofline coordinates/readings. MFU/MBU are relative to the roofs.
        "prefill_target_input_tokens", "prefill_arithmetic_intensity",
        "prefill_intended_arithmetic_intensity",
        "prefill_roofline_tflops_total", "prefill_roofline_tflops_per_gpu",
        "prefill_roofline_smfu",
        "prefill_roofline_bandwidth_tbps", "prefill_roofline_smbu",
        "peak_dense_tflops_per_gpu", "peak_memory_tbps_per_gpu", "num_gpus",
        # Tier 5 server-side cross-check (/metrics counters)
        "server_tokens_per_sec", "client_vs_server_delta_pct",
        "peak_running_reqs", "peak_cache_hit_rate",
        # Legacy Qwen3-path derivations (Qwen3-Next only)
        "prefill_raw_tflops_legacy", "prefill_smfu_legacy", "prefill_smbu_legacy",
        "prefill_roofline_tflops_total_legacy", "prefill_roofline_tflops_per_gpu_legacy",
        "prefill_roofline_smfu_legacy",
        "prefill_roofline_bandwidth_tbps_legacy", "prefill_roofline_smbu_legacy",
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


def resolve_target_input_tokens(
    metadata: dict,
    records: list,
    slug: str,
    dataset: str,
    repo_dir: Path,
) -> Optional[int]:
    """Find the fixed prefill sequence length for roofline X coordinates."""
    for section in ("benchmark_config", "dataset_config", "model_config", "system_environment"):
        value = metadata.get(section, {}).get("target_input_tokens")
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass

    config_path = repo_dir / "configs" / f"{dataset}_{slug}.yaml"
    if config_path.exists():
        try:
            cfg = yaml.safe_load(config_path.read_text()) or {}
            value = cfg.get("target_input_tokens")
            if value is not None:
                return int(value)
        except Exception as exc:
            warnings.warn(f"Could not read target_input_tokens from {config_path}: {exc}")

    # Last-resort inference for older folders: use the largest prefill step as
    # an approximate sequence length after dividing out the observed batch.
    candidates = []
    for r in records:
        seq = r.get("seq_lens_sum", 0)
        rb = r.get("batch_size") or metadata.get("system_environment", {}).get("batch_size") or 1
        try:
            if seq > 10 and rb > 0:
                candidates.append(int(round(seq / rb)))
        except TypeError:
            continue
    return max(candidates) if candidates else None


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
        # Tier 5: cross-check with server-side /metrics counters. Snapshot file
        # lives two levels above the <org>/<model> leaf (under the dataset dir).
        dataset_dir = leaf_dir.parent.parent
        snap_path = dataset_dir / f"sglang_metrics_bs{bs}.jsonl"
        snaps = load_snapshots(str(snap_path))
        _merge_tier5(metrics, snaps, bs)
        target_input_tokens = resolve_target_input_tokens(
            meta, normalized, slug, dataset, Path(__file__).resolve().parent
        )
        _finalize_roofline_metrics(metrics, bs, target_input_tokens)

        raw.append((slug, bs, dataset, metrics))
        print(f"  {slug} bs={bs} {dataset}: "
              f"tok/s={metrics['prefill_tokens_per_sec']:.0f} "
              f"TFLOPS={metrics['prefill_raw_tflops']:.1f} "
              f"S-MFU={metrics['prefill_smfu']:.1f}% "
              f"S-MBU={metrics['prefill_smbu']:.1f}%")
        if "prefill_roofline_tflops" in metrics:
            print(f"    roofline: AI={metrics['prefill_arithmetic_intensity']:.0f} "
                  f"(actual max batch={metrics.get('prefill_actual_max_batch_size', 0):.0f}) "
                  f"TFLOPS/GPU={metrics['prefill_roofline_tflops_per_gpu']:.1f} "
                  f"MFU={metrics['prefill_roofline_smfu']:.1f}% "
                  f"MBU={metrics['prefill_roofline_smbu']:.2f}%")
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
            dataset, per_slug, "Prefill TFLOPS", "prefill_raw_tflops",
            results_dir / f"raw_flops_{dataset}.png",
            legacy_key="prefill_raw_tflops_legacy",
        )
        plot_metric_per_dataset(
            dataset, per_slug, "Prefill tokens/sec", "prefill_tokens_per_sec",
            results_dir / f"tokens_per_sec_{dataset}.png",
        )
        plot_roofline_per_dataset(
            dataset, per_slug, results_dir / f"roofline_{dataset}.png",
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
