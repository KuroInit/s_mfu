"""MoE-CAP benchmark sweep orchestrator.

Manages the full sweep: model validation, SGLang subprocess lifecycle,
MoE-CAP runner invocation, and checkpoint/resume.
"""

import os
import sys
import time
import socket
import subprocess
import json
from itertools import combinations
from datetime import datetime
from pathlib import Path
import yaml
import requests
from typing import Optional
from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError

# ─── Configuration ───────────────────────────────────────────────────────────

SWEEP_CONFIG_PATH = os.environ.get("SWEEP_CONFIG", "sweep_config.yaml")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", os.path.join(RESULTS_DIR, "checkpoint.yaml"))
SGLANG_PORT = 30000
SGLANG_STARTUP_TIMEOUT = 1500  # 25 minutes — covers slow weight loads + warmup across all models/datasets
SGLANG_HEALTH_INTERVAL = 5     # seconds between health polls
SGLANG_SHUTDOWN_GRACE = 30     # seconds before SIGKILL
PORT_FREE_TIMEOUT = 60         # seconds to wait for port to clear
GPU_FREE_MEMORY_USED_MB = int(os.environ.get("GPU_FREE_MEMORY_USED_MB", "1024"))
GPU_RETRY_INTERVAL_SECONDS = int(os.environ.get("GPU_RETRY_INTERVAL_SECONDS", "15"))
GPU_MAX_IDLE_CHECKS = int(os.environ.get("GPU_MAX_IDLE_CHECKS", "3"))


def _disable_radix_cache_enabled() -> bool:
    """Return whether SGLang's prefix/radix cache should be disabled."""
    return os.environ.get("DISABLE_RADIX_CACHE", "1").lower() not in {"0", "false", "no"}


def _auto_select_gpus_enabled() -> bool:
    """Return whether the harness should bind each SGLang server to idle GPUs."""
    return os.environ.get("AUTO_SELECT_GPUS", "1").lower() not in {"0", "false", "no"}


def _cuda_visible_device_filter() -> Optional[set[str]]:
    """Return an integer CUDA_VISIBLE_DEVICES allow-list, or None for all GPUs.

    UUID/MIG forms are intentionally treated as unmanaged because nvidia-smi's
    index query cannot map those safely without more platform-specific handling.
    """
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return None
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        return set()
    if not all(device.isdigit() for device in devices):
        return None
    return set(devices)


def _query_gpu_memory_used_mb() -> Optional[list[tuple[str, int]]]:
    """Return [(gpu_index, memory_used_mb), ...], or None when nvidia-smi is unavailable."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None

    gpus: list[tuple[str, int]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        try:
            index, memory_used = [part.strip() for part in line.split(",", 1)]
            gpus.append((index, int(memory_used)))
        except ValueError:
            return None
    return gpus


def _query_gpu_p2p_write_matrix() -> Optional[dict[tuple[str, str], bool]]:
    """Return GPU peer-write compatibility from nvidia-smi, or None if unavailable.

    NCCL tensor parallel startup can fail with opaque internal errors when the
    selected GPUs cannot do peer access. The topo query keeps auto-selection
    from handing SGLang an incompatible TP group on mixed/shared hosts.
    """
    cmd = ["nvidia-smi", "topo", "-p2p", "w"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None

    lines = [line.split() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    headers = [item.removeprefix("GPU") for item in lines[0] if item.startswith("GPU")]
    if not headers:
        return None

    matrix: dict[tuple[str, str], bool] = {}
    for parts in lines[1:]:
        row_label = parts[0]
        if not row_label.startswith("GPU"):
            continue
        row_gpu = row_label.removeprefix("GPU")
        values = parts[1:1 + len(headers)]
        if len(values) != len(headers):
            return None
        for col_gpu, value in zip(headers, values):
            if row_gpu == col_gpu:
                continue
            matrix[(row_gpu, col_gpu)] = value.upper() == "OK"
    return matrix


def _select_p2p_compatible_gpus(idle: list[str], required: int) -> list[str]:
    """Return an idle GPU group that supports peer writes, or [] if none exists."""
    if required <= 1:
        return idle[:required]

    p2p = _query_gpu_p2p_write_matrix()
    if p2p is None:
        return idle[:required]

    for candidate in combinations(idle, required):
        if all(
            p2p.get((left, right), False) and p2p.get((right, left), False)
            for left, right in combinations(candidate, 2)
        ):
            return list(candidate)
    return []


def select_idle_gpus(required: int) -> Optional[list[str]]:
    """Pick physical GPU IDs with low memory use.

    Returns:
        - list[str]: enough idle GPU IDs were found
        - []: GPU auto-selection works, but not enough GPUs are idle
        - None: auto-selection is disabled or nvidia-smi is unavailable
    """
    if required <= 0 or not _auto_select_gpus_enabled():
        return None

    gpus = _query_gpu_memory_used_mb()
    if gpus is None:
        print("[gpu] nvidia-smi unavailable; leaving CUDA_VISIBLE_DEVICES unchanged")
        return None

    visible_filter = _cuda_visible_device_filter()
    idle = [
        index for index, memory_used in gpus
        if (visible_filter is None or index in visible_filter)
        and memory_used <= GPU_FREE_MEMORY_USED_MB
    ]
    if len(idle) < required:
        return []

    selected = _select_p2p_compatible_gpus(idle, required)
    if selected == [] and required > 1:
        print(
            f"[gpu] Found {len(idle)} idle GPU(s), but no peer-access-compatible "
            f"group for tp={required}"
        )
    return selected


def wait_for_idle_gpus(required: int) -> Optional[list[str]]:
    """Wait for enough idle GPUs, then return [] if the retry budget is exhausted."""
    failed_checks = 0
    while True:
        selected = select_idle_gpus(required)
        if selected != []:
            return selected
        failed_checks += 1
        if failed_checks >= GPU_MAX_IDLE_CHECKS:
            print(
                f"[gpu] No free idle GPU available for tp={required} after "
                f"{failed_checks} check(s) "
                f"(threshold={GPU_FREE_MEMORY_USED_MB}MB used)"
            )
            return []
        print(
            f"[gpu] Waiting {GPU_RETRY_INTERVAL_SECONDS}s for {required} idle GPU(s) "
            f"(check {failed_checks}/{GPU_MAX_IDLE_CHECKS}, "
            f"threshold={GPU_FREE_MEMORY_USED_MB}MB used)"
        )
        time.sleep(GPU_RETRY_INTERVAL_SECONDS)


# ─── Checkpoint ──────────────────────────────────────────────────────────────

class Checkpoint:
    """Tracks completed (model, batch_size, dataset) triples with success/failed status.

    Persists to YAML at self.path after every mark() call so that a container
    restart resumes from where it left off.
    """

    def __init__(self, path: str = CHECKPOINT_PATH) -> None:
        self.path = path
        self._entries: list = self._load()

    def _load(self) -> list:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r") as f:
            data = yaml.safe_load(f) or {}
        return data.get("completed", [])

    def _save(self) -> None:
        dir_part = os.path.dirname(self.path)
        if dir_part:
            os.makedirs(dir_part, exist_ok=True)
        with open(self.path, "w") as f:
            yaml.dump({"completed": self._entries}, f, default_flow_style=False)

    def is_done(self, slug: str, batch_size: int, dataset: str) -> bool:
        """Return True only if the run completed with status=success.

        Keyed on slug (not model_id) so multiple tp variants of the same model
        can coexist in the same checkpoint.
        """
        return any(
            e["slug"] == slug
            and e["batch_size"] == batch_size
            and e["dataset"] == dataset
            and e["status"] == "success"
            for e in self._entries
        )

    def mark(
        self,
        slug: str,
        batch_size: int,
        dataset: str,
        status: str,
        error: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> None:
        """Append an entry and persist immediately. Keyed on slug."""
        entry: dict = {
            "slug": slug,
            "batch_size": batch_size,
            "dataset": dataset,
            "status": status,
        }
        if model_id is not None:
            entry["model"] = model_id
        if error is not None:
            entry["error"] = error
        # Replace any existing entry for this triple so restarts don't accumulate stale entries
        self._entries = [
            e for e in self._entries
            if not (e["slug"] == slug and e["batch_size"] == batch_size
                    and e["dataset"] == dataset)
        ]
        self._entries.append(entry)
        self._save()

# ─── SGLang Lifecycle ─────────────────────────────────────────────────────────

def start_sglang(
    model_id: str,
    tp: int,
    batch_size: int,
    port: int = SGLANG_PORT,
    chunked_prefill_size: Optional[int] = None,
    max_prefill_tokens: Optional[int] = None,
    mem_fraction_static: Optional[float] = None,
    gpu_ids: Optional[list[str]] = None,
) -> subprocess.Popen:
    """Launch the MoE-CAP SGLang server as a background subprocess."""
    cmd = [
        sys.executable, "-m", "moe_cap.systems.sglang",
        "--model-path", model_id,
        "--port", str(port),
        "--expert-distribution-recorder-mode", "stat",
        "--tp-size", str(tp),
        "--max-running-requests", str(batch_size),
        "--enable-metrics",
        "--enable-expert-distribution-metrics",
    ]
    if _disable_radix_cache_enabled():
        cmd += ["--disable-radix-cache"]
    if chunked_prefill_size is not None and chunked_prefill_size > 0:
        cmd += ["--chunked-prefill-size", str(chunked_prefill_size)]
    if max_prefill_tokens is not None and max_prefill_tokens > 0:
        cmd += ["--max-prefill-tokens", str(max_prefill_tokens)]
    if mem_fraction_static is not None and mem_fraction_static > 0:
        cmd += ["--mem-fraction-static", str(mem_fraction_static)]
    env = None
    if gpu_ids is not None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        print(f"[gpu] Binding SGLang to physical GPU(s): {env['CUDA_VISIBLE_DEVICES']}")

    print(f"[sglang] Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env)


def wait_for_health(
    port: int = SGLANG_PORT,
    timeout: int = SGLANG_STARTUP_TIMEOUT,
    interval: int = SGLANG_HEALTH_INTERVAL,
    proc: Optional[subprocess.Popen] = None,
) -> bool:
    """Poll GET /health until the server responds 200, exits, or times out."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            print(f"[sglang] Process exited before health check passed (code={proc.returncode})")
            return False
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def kill_sglang(proc: subprocess.Popen, grace: int = SGLANG_SHUTDOWN_GRACE) -> None:
    """Gracefully stop SGLang: SIGTERM → wait → SIGKILL."""
    if proc.poll() is not None:
        return  # already dead
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        print("[sglang] Grace period expired, sending SIGKILL")
        proc.kill()
        proc.wait()


def wait_port_free(
    port: int = SGLANG_PORT,
    timeout: int = PORT_FREE_TIMEOUT,
    interval: int = 2,
) -> bool:
    """Wait until TCP port is no longer in use."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return True  # connection refused → port free
        time.sleep(interval)
    return False

# ─── Pre-flight Validation ────────────────────────────────────────────────────

def validate_models(models: list) -> None:
    """Check all model IDs exist on HuggingFace before starting any SGLang instance.

    Calls sys.exit() immediately on the first missing or unreachable model.
    """
    print("Pre-flight: validating model IDs on HuggingFace...")
    for m in models:
        model_id = m["id"]
        try:
            model_info(model_id)
            print(f"  ✓  {model_id}")
        except RepositoryNotFoundError:
            sys.exit(f"ERROR: Model not found on HuggingFace: {model_id!r}")
        except Exception as exc:
            sys.exit(f"ERROR: Could not validate {model_id!r}: {exc}")

# ─── Runner Invocation ────────────────────────────────────────────────────────

def run_benchmark(
    config_file: str,
    batch_size: int,
    output_dir: str,
    port: int = SGLANG_PORT,
) -> int:
    """Invoke MoE-CAP's OpenAI API profiler and return its exit code."""
    cmd = [
        sys.executable, "-m", "moe_cap.runner.openai_api_profile",
        "--config-file", config_file,
        "--api-url", f"http://localhost:{port}/v1/completions",
        "--backend", "sglang",
        "--server-batch-size", str(batch_size),
        "--output_dir", output_dir,
    ]
    print(f"[runner] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def persist_moe_cap_server_records(model_id: str, output_dir: str, dataset: str) -> Optional[Path]:
    """Copy MoE-CAP's full SGLang server records into this result leaf.

    MoE-CAP computes continuous-batching metrics from full server records,
    including fields such as per_req_info. Its detailed_results export is a
    reduced view, so the harness preserves the full file for post-processing.
    """
    base = os.environ.get(
        "SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR",
        os.path.join(RESULTS_DIR, "expert_records"),
    )
    src = Path(base) / model_id / "expert_distribution_record.jsonl"
    if not src.exists():
        print(f"[runner] WARNING: MoE-CAP server record file not found: {src}")
        return None
    if src.stat().st_size == 0:
        print(f"[runner] WARNING: MoE-CAP server record file is empty: {src}")
        return None

    dest_dir = Path(output_dir) / model_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = dest_dir / f"server_records_{dataset}_{ts}.jsonl"
    dest.write_text(src.read_text())
    print(f"[runner] preserved MoE-CAP server records at {dest}")
    return dest


def persist_failure_record(
    model_id: str,
    slug: str,
    output_dir: str,
    dataset: str,
    batch_size: int,
    tp: int,
    status: str,
    error: str,
    gpu_ids: Optional[list[str]] = None,
) -> Path:
    """Write a minimal result artifact for runs that fail before metrics exist."""
    dest_dir = Path(output_dir) / model_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = dest_dir / f"failure_{dataset}_{ts}.json"
    payload = {
        "status": status,
        "error": error,
        "dataset": dataset,
        "slug": slug,
        "model": model_id,
        "batch_size": batch_size,
        "tp": tp,
        "cuda_visible_devices": ",".join(gpu_ids) if gpu_ids else "",
        "timestamp": ts,
    }
    dest.write_text(json.dumps(payload, indent=2))
    print(f"[runner] preserved failure record at {dest}")
    return dest


# ─── Config Loading ───────────────────────────────────────────────────────────

def _get_max_batch_size(config_file: str) -> Optional[int]:
    """Read max_batch_size from a dataset config YAML, or None if not set."""
    try:
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("max_batch_size")
    except FileNotFoundError:
        return None


def _load_dataset_config(config_file: str) -> dict:
    """Load one MoE-CAP benchmark YAML."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f) or {}


def _get_explicit_chunked_prefill_size(model: dict, dataset_cfg: dict) -> Optional[int]:
    """Return an explicit chunked-prefill override, if one was configured.

    The harness should not infer chunking from target_input_tokens. MoE-CAP and
    SGLang own scheduling; this knob is only for operator-specified memory workarounds.
    """
    value = dataset_cfg.get("chunked_prefill_size", model.get("chunked_prefill_size"))
    return int(value) if value is not None else None


def _get_explicit_max_prefill_tokens(model: dict, dataset_cfg: dict) -> Optional[int]:
    """Return an explicit max-prefill-token override, if one was configured."""
    value = dataset_cfg.get("max_prefill_tokens", model.get("max_prefill_tokens"))
    return int(value) if value is not None else None


def _get_explicit_mem_fraction_static(model: dict, dataset_cfg: dict) -> Optional[float]:
    """Return an explicit SGLang static-memory fraction, if one was configured."""
    value = dataset_cfg.get("mem_fraction_static", model.get("mem_fraction_static"))
    return float(value) if value is not None else None


def _effective_batch_sizes(batch_sizes: list, dataset_cfg: dict) -> list:
    """Return sweep batch sizes that are not skipped by max_batch_size."""
    max_bs = dataset_cfg.get("max_batch_size")
    if max_bs is None:
        return batch_sizes
    return [bs for bs in batch_sizes if bs <= int(max_bs)]


def _estimate_per_gpu_memory_gb(model: dict, dataset_cfg: dict, batch_size: int) -> Optional[float]:
    """Conservative weights + KV estimate for one GPU.

    The optional metadata is deliberately simple and per-GPU after tensor
    parallelism. It is a preflight guardrail, not part of MoE-CAP metric math.
    """
    weight_gb = model.get("weight_gb_per_gpu")
    kv_bytes = model.get("kv_bytes_per_token_per_gpu")
    if weight_gb is None or kv_bytes is None:
        return None
    tokens = int(dataset_cfg.get("target_input_tokens", 0)) + int(dataset_cfg.get("target_output_tokens", 0))
    kv_gb = tokens * batch_size * int(kv_bytes) / (1024 ** 3)
    return float(weight_gb) + kv_gb


def validate_sweep_configs(config: dict) -> None:
    """Fail fast before launching any server if sweep dataset configs are invalid."""
    datasets = config.get("datasets", [])
    batch_sizes = config.get("batch_sizes", [])
    gpu_memory_gb = float(config.get("gpu_memory_gb", 94))
    for model in config.get("models", []):
        model_id = model["id"]
        config_slug = model.get("config_slug", model["slug"])
        for dataset in datasets:
            config_file = f"configs/{dataset}_{config_slug}.yaml"
            if not os.path.exists(config_file):
                sys.exit(f"ERROR: Missing benchmark config: {config_file}")
            cfg = _load_dataset_config(config_file)
            if cfg.get("model_id") != model_id:
                sys.exit(
                    f"ERROR: {config_file} model_id={cfg.get('model_id')!r} "
                    f"does not match sweep model {model_id!r}"
                )
            if not cfg.get("fixed_length_mode", False):
                sys.exit(f"ERROR: {config_file} must set fixed_length_mode: true")
            if cfg.get("target_input_tokens") is None:
                sys.exit(f"ERROR: {config_file} must set target_input_tokens")
            if int(cfg.get("target_output_tokens", 0)) <= 0:
                sys.exit(f"ERROR: {config_file} must set positive target_output_tokens")
            max_context = model.get("max_context_tokens")
            total_tokens = int(cfg["target_input_tokens"]) + int(cfg["target_output_tokens"])
            if max_context is not None and total_tokens > int(max_context):
                sys.exit(
                    f"ERROR: {config_file} requests {total_tokens} total tokens, "
                    f"but {model_id} max_context_tokens={max_context}"
                )
            for bs in _effective_batch_sizes(batch_sizes, cfg):
                estimated = _estimate_per_gpu_memory_gb(model, cfg, int(bs))
                if estimated is not None and estimated > gpu_memory_gb:
                    sys.exit(
                        f"ERROR: {config_file} bs={bs} estimated per-GPU memory "
                        f"{estimated:.1f}GB exceeds {gpu_memory_gb:.1f}GB. "
                        f"Lower max_batch_size or target_input_tokens."
                    )


def load_sweep_config(path: str = SWEEP_CONFIG_PATH) -> dict:
    """Load and return the sweep configuration YAML."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─── Sweep Loop ───────────────────────────────────────────────────────────────

def run_sweep(config: dict, checkpoint: Checkpoint) -> None:
    """Execute the full model × batch_size × dataset sweep.

    SGLang is restarted per (model, bs, dataset) triple. This eliminates the
    detailed_results/expert_records carryover that contaminated dataset #2 when
    one server lifetime spanned multiple datasets (Audit Finding #21).
    """
    port = config.get("port", SGLANG_PORT)
    models = config["models"]
    batch_sizes = config["batch_sizes"]
    datasets = config["datasets"]

    total = len(models) * len(batch_sizes) * len(datasets)
    done = 0

    for model in models:
        model_id = model["id"]
        slug = model["slug"]
        tp = model["tp"]
        # config_slug controls which configs/<dataset>_<config_slug>.yaml files are
        # read. Defaults to slug — set explicitly when multiple sweep entries share
        # the same dataset configs (e.g., tp=1 and tp=2 variants of one model).
        config_slug = model.get("config_slug", slug)

        for batch_size in batch_sizes:
            for dataset in datasets:
                if checkpoint.is_done(slug, batch_size, dataset):
                    print(f"[sweep] Skipping {slug} bs={batch_size} {dataset} — done")
                    done += 1
                    continue

                config_file = f"configs/{dataset}_{config_slug}.yaml"
                output_dir = f"{RESULTS_DIR}/{slug}/bs{batch_size}/{dataset}/"
                dataset_cfg = _load_dataset_config(config_file) if os.path.exists(config_file) else {}

                max_bs = _get_max_batch_size(config_file)
                if max_bs is not None and batch_size > max_bs:
                    print(f"[sweep] Skipping {slug} bs={batch_size} {dataset} — bs > max_batch_size={max_bs}")
                    checkpoint.mark(slug, batch_size, dataset, "skipped",
                                    f"batch_size {batch_size} > max_batch_size {max_bs}",
                                    model_id=model_id)
                    done += 1
                    continue

                prefill_size = _get_explicit_chunked_prefill_size(model, dataset_cfg)
                max_prefill_tokens = _get_explicit_max_prefill_tokens(model, dataset_cfg)
                mem_fraction_static = _get_explicit_mem_fraction_static(model, dataset_cfg)

                sep = "=" * 60
                print(f"\n{sep}")
                print(f"[sweep] [{done+1}/{total}] {slug}  bs={batch_size}  {dataset}")
                print(
                    f"[sweep]   tp={tp}  chunked_prefill_size={prefill_size} "
                    f"max_prefill_tokens={max_prefill_tokens} "
                    f"mem_fraction_static={mem_fraction_static}"
                )
                print(sep)

                selected_gpus = wait_for_idle_gpus(tp)
                if selected_gpus == []:
                    print(
                        f"[gpu] Skipping {slug} bs={batch_size} {dataset}: "
                        f"no free idle GPU available for tp={tp}"
                    )
                    done += 1
                    continue

                proc = start_sglang(
                    model_id,
                    tp,
                    batch_size,
                    port,
                    prefill_size,
                    max_prefill_tokens,
                    mem_fraction_static,
                    selected_gpus,
                )
                try:
                    if not wait_for_health(port, proc=proc):
                        exit_code = proc.poll()
                        if exit_code is None:
                            error = f"SGLang failed to start within {SGLANG_STARTUP_TIMEOUT}s"
                            status = "startup_failed"
                        else:
                            error = (
                                f"SGLang exited during startup with code {exit_code}; "
                                "likely startup OOM or scheduler initialization failure"
                            )
                            status = "oom"
                        print(f"[sweep] ERROR: {error}")
                        persist_failure_record(
                            model_id, slug, output_dir, dataset, batch_size,
                            tp, status, error, selected_gpus,
                        )
                        checkpoint.mark(
                            slug, batch_size, dataset, "failed",
                            error,
                            model_id=model_id,
                        )
                        done += 1
                        continue

                    rc = run_benchmark(config_file, batch_size, output_dir, port)

                    if rc == 0:
                        preserved = persist_moe_cap_server_records(model_id, output_dir, dataset)
                        if preserved is None:
                            error = (
                                "Runner completed but no valid SGLang server records were found; "
                                "continuous-batching metrics would be invalid"
                            )
                            persist_failure_record(
                                model_id, slug, output_dir, dataset, batch_size,
                                tp, "missing_server_records", error, selected_gpus,
                            )
                            checkpoint.mark(
                                slug, batch_size, dataset, "failed",
                                error,
                                model_id=model_id,
                            )
                            print(f"[sweep]   ✗ {dataset} ({error})")
                        else:
                            checkpoint.mark(slug, batch_size, dataset, "success", model_id=model_id)
                            print(f"[sweep]   ✓ {dataset}")
                    else:
                        error = f"Runner exited with code {rc}"
                        persist_failure_record(
                            model_id, slug, output_dir, dataset, batch_size,
                            tp, "failed", error, selected_gpus,
                        )
                        checkpoint.mark(
                            slug, batch_size, dataset, "failed",
                            error,
                            model_id=model_id,
                        )
                        print(f"[sweep]   ✗ {dataset} (exit code {rc})")
                    done += 1

                finally:
                    print(f"[sweep] Shutting down SGLang...")
                    kill_sglang(proc)
                    if not wait_port_free(port):
                        print(f"[sweep] WARNING: port {port} still in use after shutdown")

    print(f"\n{'='*60}")
    print(f"[sweep] Sweep complete. {done}/{total} runs processed.")


def main() -> None:
    config = load_sweep_config()
    checkpoint = Checkpoint()
    validate_sweep_configs(config)
    validate_models(config["models"])
    run_sweep(config, checkpoint)


if __name__ == "__main__":
    main()
