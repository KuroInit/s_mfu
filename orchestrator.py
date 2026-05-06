"""MoE-CAP benchmark sweep orchestrator.

Manages the full sweep: model validation, SGLang subprocess lifecycle,
MoE-CAP runner invocation, and checkpoint/resume.
"""

import os
import sys
import time
import socket
import subprocess
from datetime import datetime
from pathlib import Path
import yaml
import requests
from typing import Optional
from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError

from sglang_metrics import SGLangMetricsPoller

# ─── Configuration ───────────────────────────────────────────────────────────

SWEEP_CONFIG_PATH = os.environ.get("SWEEP_CONFIG", "sweep_config.yaml")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", os.path.join(RESULTS_DIR, "checkpoint.yaml"))
SGLANG_PORT = 30000
SGLANG_STARTUP_TIMEOUT = 1500  # 25 minutes — covers slow weight loads + warmup across all models/datasets
SGLANG_HEALTH_INTERVAL = 5     # seconds between health polls
SGLANG_SHUTDOWN_GRACE = 30     # seconds before SIGKILL
PORT_FREE_TIMEOUT = 60         # seconds to wait for port to clear
METRICS_POLL_INTERVAL = 1.0    # seconds; set METRICS_POLL_INTERVAL=0 to disable Tier 5


def _disable_radix_cache_enabled() -> bool:
    """Return whether SGLang's prefix/radix cache should be disabled."""
    return os.environ.get("DISABLE_RADIX_CACHE", "1").lower() not in {"0", "false", "no"}


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
    ]
    if _disable_radix_cache_enabled():
        cmd += ["--disable-radix-cache"]
    if chunked_prefill_size is not None and chunked_prefill_size > 0:
        cmd += ["--chunked-prefill-size", str(chunked_prefill_size)]
    print(f"[sglang] Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


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
    """Invoke the configured runner and return its exit code.

    Default is the harness-owned strict-serial runner because this sweep needs
    the client-side contract "batch_size = one request wave". Set
    BATCH_RUNNER=upstream to use MoE-CAP's openai_api_profile runner.
    """
    mode = os.environ.get("BATCH_RUNNER", "strict").lower()
    if mode == "strict":
        cmd = [
            sys.executable, "batch_runner.py",
            "--config-file", config_file,
            "--api-url", f"http://localhost:{port}/v1/completions",
            "--backend", "sglang",
            "--server-batch-size", str(batch_size),
            "--output_dir", output_dir,
        ]
    else:
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

    dest_dir = Path(output_dir) / model_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = dest_dir / f"server_records_{dataset}_{ts}.jsonl"
    dest.write_text(src.read_text())
    print(f"[runner] preserved MoE-CAP server records at {dest}")
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
            if int(cfg.get("target_output_tokens", 0)) != 1:
                sys.exit(f"ERROR: {config_file} must set target_output_tokens: 1")
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

                sep = "=" * 60
                print(f"\n{sep}")
                print(f"[sweep] [{done+1}/{total}] {slug}  bs={batch_size}  {dataset}")
                print(f"[sweep]   tp={tp}  chunked_prefill_size={prefill_size}")
                print(sep)

                proc = start_sglang(model_id, tp, batch_size, port, prefill_size)
                try:
                    if not wait_for_health(port, proc=proc):
                        exit_code = proc.poll()
                        if exit_code is None:
                            error = f"SGLang failed to start within {SGLANG_STARTUP_TIMEOUT}s"
                        else:
                            error = (
                                f"SGLang exited during startup with code {exit_code}; "
                                "likely startup OOM or scheduler initialization failure"
                            )
                        print(f"[sweep] ERROR: {error}")
                        checkpoint.mark(
                            slug, batch_size, dataset, "failed",
                            error,
                            model_id=model_id,
                        )
                        done += 1
                        continue

                    # Tier 5: server-side ground-truth throughput + serial-wave proof.
                    poller = None
                    interval = float(os.environ.get("METRICS_POLL_INTERVAL", METRICS_POLL_INTERVAL))
                    if interval > 0:
                        metrics_path = os.path.join(
                            output_dir, f"sglang_metrics_bs{batch_size}.jsonl"
                        )
                        poller = SGLangMetricsPoller(
                            base_url=f"http://localhost:{port}",
                            output_path=metrics_path,
                            interval=interval,
                            label=f"{slug}_bs{batch_size}_{dataset}",
                        )
                        poller.start()

                    try:
                        rc = run_benchmark(config_file, batch_size, output_dir, port)
                    finally:
                        if poller is not None:
                            poller.stop()

                    if rc == 0:
                        persist_moe_cap_server_records(model_id, output_dir, dataset)
                        checkpoint.mark(slug, batch_size, dataset, "success", model_id=model_id)
                        print(f"[sweep]   ✓ {dataset}")
                    else:
                        checkpoint.mark(
                            slug, batch_size, dataset, "failed",
                            f"Runner exited with code {rc}",
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
