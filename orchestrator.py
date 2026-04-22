"""MoE-CAP benchmark sweep orchestrator.

Manages the full sweep: model validation, SGLang subprocess lifecycle,
MoE-CAP runner invocation, and checkpoint/resume.
"""

import os
import sys
import time
import socket
import subprocess
import yaml
import requests
from typing import Optional
from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError

from sglang_metrics import SGLangMetricsPoller

# ─── Configuration ───────────────────────────────────────────────────────────

SWEEP_CONFIG_PATH = os.environ.get("SWEEP_CONFIG", "sweep_config.yaml")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/results")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", os.path.join(RESULTS_DIR, "checkpoint.yaml"))
SGLANG_PORT = 30000
SGLANG_STARTUP_TIMEOUT = 1500  # 25 minutes — covers slow weight loads + warmup across all models/datasets
SGLANG_HEALTH_INTERVAL = 5     # seconds between health polls
SGLANG_SHUTDOWN_GRACE = 30     # seconds before SIGKILL
PORT_FREE_TIMEOUT = 60         # seconds to wait for port to clear
METRICS_POLL_INTERVAL = 1.0    # seconds; set METRICS_POLL_INTERVAL=0 to disable Tier 5


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
    if chunked_prefill_size is not None and chunked_prefill_size > 0:
        cmd += ["--chunked-prefill-size", str(chunked_prefill_size)]
    print(f"[sglang] Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def wait_for_health(
    port: int = SGLANG_PORT,
    timeout: int = SGLANG_STARTUP_TIMEOUT,
    interval: int = SGLANG_HEALTH_INTERVAL,
) -> bool:
    """Poll GET /health until the server responds 200 or timeout expires."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
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

    Default is `moe_cap.runner.openai_api_profile`. Set BATCH_RUNNER=strict to
    use the orchestrator-owned `batch_runner.py` which issues strict-serial
    waves (no bs=1 flood-fire, no 50% inter-batch overlap). Both runners write
    the same detailed_results/metadata output schema.
    """
    mode = os.environ.get("BATCH_RUNNER", "").lower()
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


# ─── Config Loading ───────────────────────────────────────────────────────────

def _get_max_batch_size(config_file: str) -> Optional[int]:
    """Read max_batch_size from a dataset config YAML, or None if not set."""
    try:
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("max_batch_size")
    except FileNotFoundError:
        return None


CHUNK_SIZE_CAP = 32768  # Global default; per-model override via sweep_config.chunk_size_cap.


def _required_chunk_tokens(config_file: str, batch_size: int) -> int:
    """Return required chunked-prefill tokens for one dataset config, or 0 if N/A.

    prefill_mode=batched scales with batch_size; default=single uses one prefill
    per step. Returns 0 if the dataset disallows this batch_size or has no config.
    """
    try:
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return 0
    max_bs = cfg.get("max_batch_size")
    if max_bs is not None and batch_size > max_bs:
        return 0
    tokens = cfg.get("target_input_tokens", 0)
    mode = cfg.get("prefill_mode", "single")
    return tokens * batch_size if mode == "batched" else tokens


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
        chunk_cap = model.get("chunk_size_cap", CHUNK_SIZE_CAP)

        for batch_size in batch_sizes:
            for dataset in datasets:
                if checkpoint.is_done(slug, batch_size, dataset):
                    print(f"[sweep] Skipping {slug} bs={batch_size} {dataset} — done")
                    done += 1
                    continue

                config_file = f"configs/{dataset}_{config_slug}.yaml"
                output_dir = f"{RESULTS_DIR}/{slug}/bs{batch_size}/{dataset}/"

                max_bs = _get_max_batch_size(config_file)
                if max_bs is not None and batch_size > max_bs:
                    print(f"[sweep] Skipping {slug} bs={batch_size} {dataset} — bs > max_batch_size={max_bs}")
                    checkpoint.mark(slug, batch_size, dataset, "skipped",
                                    f"batch_size {batch_size} > max_batch_size {max_bs}",
                                    model_id=model_id)
                    done += 1
                    continue

                required = _required_chunk_tokens(config_file, batch_size)
                prefill_size = min(required + 50, chunk_cap) if required > 0 else None

                sep = "=" * 60
                print(f"\n{sep}")
                print(f"[sweep] [{done+1}/{total}] {slug}  bs={batch_size}  {dataset}")
                print(f"[sweep]   tp={tp}  chunked_prefill_size={prefill_size}  chunk_cap={chunk_cap}")
                print(sep)

                proc = start_sglang(model_id, tp, batch_size, port, prefill_size)
                try:
                    if not wait_for_health(port):
                        print(f"[sweep] ERROR: SGLang failed to start within {SGLANG_STARTUP_TIMEOUT}s")
                        checkpoint.mark(
                            slug, batch_size, dataset, "failed",
                            f"SGLang failed to start within {SGLANG_STARTUP_TIMEOUT}s",
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
    validate_models(config["models"])
    run_sweep(config, checkpoint)


if __name__ == "__main__":
    main()
