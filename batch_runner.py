"""Orchestrator-owned strict-serial batch runner.

Drop-in replacement for `moe_cap.runner.openai_api_profile` that fixes two
client-side pathologies in MoE-CAP's asyncio run_benchmark:

  1. bs=1 flood-fire — threshold = bs//2 = 0 means the client never waits
     before launching the next request, so all N prompts go in-flight at once.
  2. 50% inter-batch overlap — at bs≥2 the next wave is launched as soon as
     half the current wave completes, so peak concurrency is ~1.5 × bs.

This runner enforces: send N → await ALL N → send next N. No overlap.

It reuses MoE-CAP's dataset loaders so prompts are identical to the upstream
runner, and it pulls per-forward-pass records from the SGLang server via the
same /dump_expert_distribution_record endpoint so analyze.py reads the same
detailed_results_*.jsonl format.

CLI surface mirrors moe_cap.runner.openai_api_profile enough for the
orchestrator's run_benchmark() wrapper to call it unchanged.
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml

from moe_cap.configs import CAPConfig
from moe_cap.data_loader.loader_registry import get_loader_for_task


REQUEST_TIMEOUT = int(os.environ.get("BATCH_RUNNER_REQUEST_TIMEOUT", "3600"))


def _send_one(api_url: str, model: str, prompt: str, output_len: int) -> dict:
    """Issue one /v1/completions call and return client-side timing + usage."""
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": output_len,
        "stream": False,
        "ignore_eos": True,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(api_url, json=payload, timeout=REQUEST_TIMEOUT)
        latency = time.perf_counter() - t0
        resp.raise_for_status()
        body = resp.json()
        usage = body.get("usage", {}) or {}
        return {
            "success": True,
            "latency": latency,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
    except Exception as exc:
        return {
            "success": False,
            "latency": time.perf_counter() - t0,
            "error": str(exc),
        }


def run_serial_waves(
    api_url: str,
    model: str,
    prompts: list,
    batch_size: int,
    output_len: int,
) -> list:
    """Submit prompts in strict N-at-a-time waves. Waves do NOT overlap.

    Each wave uses a fresh ThreadPoolExecutor with exactly batch_size workers;
    the enclosing 'with' block doesn't return until all N have finished.
    """
    results = []
    n = len(prompts)
    wave_count = (n + batch_size - 1) // batch_size

    for w in range(wave_count):
        start = w * batch_size
        end = min(start + batch_size, n)
        wave = prompts[start:end]
        print(f"[batch_runner] wave {w+1}/{wave_count}: submitting {len(wave)} requests")
        wave_t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(wave)) as pool:
            futures = [pool.submit(_send_one, api_url, model, p, output_len)
                       for p in wave]
            wave_results = [f.result() for f in futures]
        wave_dt = time.perf_counter() - wave_t0
        ok = sum(1 for r in wave_results if r.get("success"))
        print(f"[batch_runner]   wave done in {wave_dt:.2f}s  ({ok}/{len(wave)} ok)")
        results.extend(wave_results)
    return results


def pull_server_records(base_url: str, model_name: str) -> list:
    """Trigger SGLang's expert-distribution dump and read the jsonl back."""
    try:
        r = requests.post(f"{base_url}/dump_expert_distribution_record", timeout=30)
        r.raise_for_status()
    except Exception as exc:
        print(f"[batch_runner] WARNING: dump endpoint failed: {exc}")
        return []

    base = os.environ.get(
        "SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR",
        os.path.join(os.getcwd(), "expert_records"),
    )
    rec_file = Path(base) / model_name / "expert_distribution_record.jsonl"
    if not rec_file.exists():
        print(f"[batch_runner] WARNING: record file not found: {rec_file}")
        return []

    records = []
    with rec_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[batch_runner] pulled {len(records)} server records")
    return records


def get_server_info(base_url: str) -> dict:
    try:
        r = requests.get(f"{base_url}/get_server_info", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def write_outputs(dest_dir: Path, dataset_name: str, server_records: list,
                  metadata_dict: dict) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    meta_path = dest_dir / f"metadata_{dataset_name}_{ts}.json"
    with meta_path.open("w") as f:
        json.dump(metadata_dict, f, indent=4)
    print(f"[batch_runner] wrote {meta_path}")

    det_path = dest_dir / f"detailed_results_{dataset_name}_{ts}.jsonl"
    with det_path.open("w") as f:
        for i, sr in enumerate(server_records):
            rec = {
                "index": i,
                "forward_mode": sr.get("forward_mode", "unknown"),
                "expert_activation": sr.get("expert_activation", 0),
                "batch_size": sr.get("batch_size", 0),
                "seq_lens_sum": sr.get("seq_lens_sum", 0),
                "gpu_raw_type": sr.get("gpu_raw_type"),
                "gpu_num": sr.get("gpu_num"),
            }
            if sr.get("forward_mode") == "prefill":
                rec["ttft"] = sr.get("latency", 0)
            else:
                rec["tpot"] = sr.get("latency", 0)
            f.write(json.dumps(rec) + "\n")
    print(f"[batch_runner] wrote {det_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--api-url", required=True,
                    help="Full completions URL, e.g. http://localhost:30000/v1/completions")
    ap.add_argument("--backend", default="sglang")
    ap.add_argument("--server-batch-size", type=int, required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    with open(args.config_file) as f:
        cfg = yaml.safe_load(f) or {}

    model_name = cfg["model_id"]
    precision = cfg.get("precision", "bfloat16")
    dataset_names = cfg.get("dataset_names", [])
    num_samples = int(cfg.get("num_samples", 200))
    output_len = int(cfg.get("target_output_tokens", 1))

    if not dataset_names:
        print("[batch_runner] ERROR: dataset_names is empty in config", file=sys.stderr)
        return 2

    # Reuse MoE-CAP's loader so prompts match upstream exactly.
    cap_config = CAPConfig(
        dataset_names=dataset_names,
        metrics=cfg.get("metrics", []),
        model_id=model_name,
        precision=precision,
        fixed_length_mode=cfg.get("fixed_length_mode", False),
        target_input_tokens=cfg.get("target_input_tokens"),
        target_output_tokens=output_len,
        dataset_split=cfg.get("dataset_split", "train"),
        num_samples=num_samples,
    )

    dataset_name = dataset_names[0]
    loader, _ = get_loader_for_task(dataset_name, cap_config)
    prompts_data = loader.get_input()
    prompts = [p["prompt"] if isinstance(p, dict) else p for p in prompts_data][:num_samples]
    if not prompts:
        print("[batch_runner] ERROR: loader returned no prompts", file=sys.stderr)
        return 3
    print(f"[batch_runner] loaded {len(prompts)} prompts for {dataset_name}")

    # Base URL = api_url without trailing path.
    base_url = args.api_url.rsplit("/v1/", 1)[0]

    t_total_0 = time.perf_counter()
    client_results = run_serial_waves(
        args.api_url, model_name, prompts, args.server_batch_size, output_len,
    )
    total_time = time.perf_counter() - t_total_0
    ok = sum(1 for r in client_results if r.get("success"))
    print(f"[batch_runner] all waves done in {total_time:.2f}s ({ok}/{len(prompts)} ok)")

    if ok == 0:
        print("[batch_runner] ERROR: no requests succeeded", file=sys.stderr)
        return 4

    server_records = pull_server_records(base_url, model_name)

    info = get_server_info(base_url)
    gpu_type = info.get("gpu_type", "unknown")
    num_gpus = info.get("tp_size", int(os.environ.get("TP_SIZE", "1")))

    metadata_dict = {
        "hardware": {"gpu_type": gpu_type, "num_gpus": num_gpus},
        "model_config": {"model_name": model_name, "precision": precision},
        "system_environment": {
            "inference_engine": args.backend,
            "inference_engine_version": info.get("version", "unknown"),
            "batch_size": args.server_batch_size,
            "batch_runner": "serial_waves_v1",
            "num_prompts": len(prompts),
            "total_wall_time_sec": total_time,
            "client_success_count": ok,
            "client_fail_count": len(prompts) - ok,
        },
    }

    dest_dir = Path(args.output_dir) / model_name
    write_outputs(dest_dir, dataset_name, server_records, metadata_dict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
