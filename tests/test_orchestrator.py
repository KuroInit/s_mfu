import os
import sys
import subprocess
import yaml
import pytest
import tempfile
from unittest.mock import patch, MagicMock, call

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── Checkpoint Tests ────────────────────────────────────────────────────────

class TestCheckpoint:
    def test_load_returns_empty_when_file_missing(self, tmp_path):
        from orchestrator import Checkpoint
        ckpt = Checkpoint(path=str(tmp_path / "checkpoint.yaml"))
        assert ckpt._entries == []

    def test_is_done_returns_false_when_empty(self, tmp_path):
        from orchestrator import Checkpoint
        ckpt = Checkpoint(path=str(tmp_path / "checkpoint.yaml"))
        assert not ckpt.is_done("modelA", 1, "gsm8k")

    def test_mark_success_and_is_done(self, tmp_path):
        from orchestrator import Checkpoint
        path = str(tmp_path / "checkpoint.yaml")
        ckpt = Checkpoint(path=path)
        ckpt.mark("modelA", 1, "gsm8k", "success")
        assert ckpt.is_done("modelA", 1, "gsm8k")

    def test_mark_failed_is_not_done(self, tmp_path):
        from orchestrator import Checkpoint
        path = str(tmp_path / "checkpoint.yaml")
        ckpt = Checkpoint(path=path)
        ckpt.mark("modelA", 1, "gsm8k", "failed", error="Runner crashed")
        assert not ckpt.is_done("modelA", 1, "gsm8k")

    def test_mark_persists_to_disk(self, tmp_path):
        from orchestrator import Checkpoint
        path = str(tmp_path / "checkpoint.yaml")
        ckpt = Checkpoint(path=path)
        ckpt.mark("modelA", 32, "numinamath", "success")

        # Load a fresh instance from the same file
        ckpt2 = Checkpoint(path=path)
        assert ckpt2.is_done("modelA", 32, "numinamath")

    def test_checkpoint_yaml_structure(self, tmp_path):
        from orchestrator import Checkpoint
        path = str(tmp_path / "checkpoint.yaml")
        ckpt = Checkpoint(path=path)
        ckpt.mark("modelA", 1, "gsm8k", "success")
        ckpt.mark("modelA", 1, "numinamath", "failed", error="exit code 1")

        with open(path) as f:
            data = yaml.safe_load(f)

        assert "completed" in data
        assert len(data["completed"]) == 2
        assert data["completed"][0]["status"] == "success"
        assert data["completed"][1]["error"] == "exit code 1"

    def test_is_done_does_not_match_different_batch_size(self, tmp_path):
        from orchestrator import Checkpoint
        path = str(tmp_path / "checkpoint.yaml")
        ckpt = Checkpoint(path=path)
        ckpt.mark("modelA", 1, "gsm8k", "success")
        assert not ckpt.is_done("modelA", 32, "gsm8k")

    def test_is_done_does_not_match_different_dataset(self, tmp_path):
        from orchestrator import Checkpoint
        path = str(tmp_path / "checkpoint.yaml")
        ckpt = Checkpoint(path=path)
        ckpt.mark("modelA", 1, "gsm8k", "success")
        assert not ckpt.is_done("modelA", 1, "numinamath")

    def test_mark_replaces_existing_entry_on_retry(self, tmp_path):
        from orchestrator import Checkpoint
        path = str(tmp_path / "checkpoint.yaml")
        ckpt = Checkpoint(path=path)
        ckpt.mark("modelA", 1, "gsm8k", "failed", error="Runner crashed")
        ckpt.mark("modelA", 1, "gsm8k", "success")
        # Only one entry should remain — no stale failed entry
        triples = [(e["model"], e["batch_size"], e["dataset"]) for e in ckpt._entries]
        assert triples.count(("modelA", 1, "gsm8k")) == 1
        assert ckpt.is_done("modelA", 1, "gsm8k")

# ─── SGLang Lifecycle Tests ──────────────────────────────────────────────────

class TestWaitForHealth:
    def test_returns_true_when_server_responds_200(self):
        from orchestrator import wait_for_health
        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch("orchestrator.requests.get", return_value=mock_response):
            result = wait_for_health(port=30000, timeout=10, interval=0)
        assert result is True

    def test_returns_false_on_timeout(self):
        from orchestrator import wait_for_health
        # Use timeout=1 so the loop runs but always fails, exhausting the deadline
        with patch("orchestrator.requests.get", side_effect=ConnectionRefusedError("refused")):
            with patch("orchestrator.time.sleep"):  # skip actual sleeping
                result = wait_for_health(port=30000, timeout=1, interval=0)
        assert result is False

    def test_retries_until_success(self):
        from orchestrator import wait_for_health
        ok = MagicMock(status_code=200)
        # First two calls raise ConnectionRefusedError (common during SGLang startup),
        # third call returns 200
        with patch("orchestrator.requests.get",
                   side_effect=[ConnectionRefusedError(), ConnectionRefusedError(), ok]):
            result = wait_for_health(port=30000, timeout=60, interval=0)
        assert result is True


class TestStartSglang:
    def test_invokes_correct_command(self):
        from orchestrator import start_sglang
        with patch("orchestrator.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            start_sglang(model_id="org/model", tp=2, batch_size=64, port=30000)
        cmd = mock_popen.call_args[0][0]
        assert "--model-path" in cmd
        assert "org/model" in cmd
        assert "--tp-size" in cmd
        assert "2" in cmd
        assert "--max-running-requests" in cmd
        assert "64" in cmd
        assert "--expert-distribution-recorder-mode" in cmd
        assert "stat" in cmd


class TestKillSglang:
    def test_terminates_running_process(self):
        from orchestrator import kill_sglang
        proc = MagicMock()
        proc.poll.return_value = None          # process is running
        proc.wait.return_value = None
        kill_sglang(proc, grace=0)
        proc.terminate.assert_called_once()

    def test_skips_already_dead_process(self):
        from orchestrator import kill_sglang
        proc = MagicMock()
        proc.poll.return_value = 0             # already exited
        kill_sglang(proc, grace=0)
        proc.terminate.assert_not_called()

    def test_sigkills_if_terminate_times_out(self):
        from orchestrator import kill_sglang
        proc = MagicMock()
        proc.poll.return_value = None
        proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="sglang", timeout=0), None]
        kill_sglang(proc, grace=0)
        proc.kill.assert_called_once()


class TestWaitPortFree:
    def test_returns_true_when_port_is_free(self):
        from orchestrator import wait_port_free
        with patch("orchestrator.socket.socket") as mock_sock_cls:
            mock_sock = MagicMock()
            mock_sock.__enter__ = lambda s: s
            mock_sock.__exit__ = MagicMock(return_value=False)
            mock_sock.connect_ex.return_value = 111  # connection refused → port free
            mock_sock_cls.return_value = mock_sock
            result = wait_port_free(port=30000, timeout=5, interval=0)
        assert result is True

    def test_returns_false_when_port_stays_occupied(self):
        from orchestrator import wait_port_free
        with patch("orchestrator.socket.socket") as mock_sock_cls:
            mock_sock = MagicMock()
            mock_sock.__enter__ = lambda s: s
            mock_sock.__exit__ = MagicMock(return_value=False)
            mock_sock.connect_ex.return_value = 0   # connection succeeded → port in use
            mock_sock_cls.return_value = mock_sock
            result = wait_port_free(port=30000, timeout=0, interval=0)
        assert result is False

# ─── Pre-flight Validation Tests ────────────────────────────────────────────

class TestValidateModels:
    def test_passes_when_all_models_found(self):
        from orchestrator import validate_models
        models = [
            {"id": "Qwen/Qwen1.5-MoE-A2.7B-Chat", "slug": "q", "tp": 1},
            {"id": "deepseek-ai/DeepSeek-V2-Lite-Chat", "slug": "d", "tp": 1},
        ]
        with patch("orchestrator.model_info") as mock_mi:
            mock_mi.return_value = MagicMock()
            validate_models(models)  # should not raise
            assert mock_mi.call_count == 2

    def test_exits_on_missing_model(self):
        from orchestrator import validate_models
        from huggingface_hub.utils import RepositoryNotFoundError
        models = [{"id": "org/does-not-exist", "slug": "x", "tp": 1}]
        with patch("orchestrator.model_info", side_effect=RepositoryNotFoundError("404", response=MagicMock())):
            with pytest.raises(SystemExit):
                validate_models(models)

    def test_exits_on_unexpected_hf_error(self):
        from orchestrator import validate_models
        models = [{"id": "org/model", "slug": "x", "tp": 1}]
        with patch("orchestrator.model_info", side_effect=Exception("network error")):
            with pytest.raises(SystemExit):
                validate_models(models)

# ─── Runner Invocation Tests ─────────────────────────────────────────────────

class TestRunBenchmark:
    def test_returns_zero_on_success(self):
        from orchestrator import run_benchmark
        with patch("orchestrator.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            rc = run_benchmark(
                config_file="configs/gsm8k_qwen3_30b.yaml",
                batch_size=32,
                output_dir="/results/qwen3_30b/bs32/gsm8k/",
                port=30000,
            )
        assert rc == 0

    def test_returns_nonzero_on_failure(self):
        from orchestrator import run_benchmark
        with patch("orchestrator.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            rc = run_benchmark(
                config_file="configs/gsm8k_qwen3_30b.yaml",
                batch_size=32,
                output_dir="/results/qwen3_30b/bs32/gsm8k/",
                port=30000,
            )
        assert rc == 1

    def test_invokes_correct_command(self):
        from orchestrator import run_benchmark
        with patch("orchestrator.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            run_benchmark(
                config_file="configs/gsm8k_qwen3_30b.yaml",
                batch_size=64,
                output_dir="/results/qwen3_30b/bs64/gsm8k/",
                port=30000,
            )
        cmd = mock_run.call_args[0][0]
        assert "--config-file" in cmd
        assert "configs/gsm8k_qwen3_30b.yaml" in cmd
        assert "--server-batch-size" in cmd
        assert "64" in cmd
        assert "--backend" in cmd
        assert "sglang" in cmd
        assert "--output_dir" in cmd
        assert "/results/qwen3_30b/bs64/gsm8k/" in cmd


# ─── Config Loading Tests ────────────────────────────────────────────────────

class TestLoadSweepConfig:
    def test_loads_models_and_batch_sizes(self, tmp_path):
        from orchestrator import load_sweep_config
        cfg_content = """
batch_sizes: [1, 32]
datasets: [gsm8k]
port: 30000
models:
  - id: Qwen/ModelA
    tp: 1
    slug: model_a
"""
        cfg_path = tmp_path / "sweep_config.yaml"
        cfg_path.write_text(cfg_content)
        cfg = load_sweep_config(str(cfg_path))
        assert cfg["batch_sizes"] == [1, 32]
        assert len(cfg["models"]) == 1
        assert cfg["models"][0]["slug"] == "model_a"


# ─── Sweep Loop Tests ────────────────────────────────────────────────────────

class TestRunSweep:
    def _make_config(self):
        return {
            "port": 30000,
            "batch_sizes": [1],
            "datasets": ["gsm8k", "numinamath"],
            "models": [{"id": "org/modelA", "slug": "model_a", "tp": 1}],
        }

    def test_skips_sglang_when_all_datasets_done(self, tmp_path):
        from orchestrator import run_sweep, Checkpoint
        ckpt = Checkpoint(path=str(tmp_path / "ckpt.yaml"))
        ckpt.mark("org/modelA", 1, "gsm8k", "success")
        ckpt.mark("org/modelA", 1, "numinamath", "success")
        with patch("orchestrator.start_sglang") as mock_start:
            run_sweep(self._make_config(), ckpt)
        mock_start.assert_not_called()

    def test_marks_all_datasets_failed_when_sglang_wont_start(self, tmp_path):
        from orchestrator import run_sweep, Checkpoint
        ckpt = Checkpoint(path=str(tmp_path / "ckpt.yaml"))
        with patch("orchestrator.start_sglang", return_value=MagicMock()):
            with patch("orchestrator.wait_for_health", return_value=False):
                with patch("orchestrator.kill_sglang"):
                    with patch("orchestrator.wait_port_free", return_value=True):
                        run_sweep(self._make_config(), ckpt)
        assert not ckpt.is_done("org/modelA", 1, "gsm8k")
        assert not ckpt.is_done("org/modelA", 1, "numinamath")
        # Both should be recorded as failed
        assert any(
            e["model"] == "org/modelA" and e["dataset"] == "gsm8k" and e["status"] == "failed"
            for e in ckpt._entries
        )
        assert any(
            e["model"] == "org/modelA" and e["dataset"] == "numinamath" and e["status"] == "failed"
            for e in ckpt._entries
        )

    def test_marks_dataset_failed_on_nonzero_runner_exit(self, tmp_path):
        from orchestrator import run_sweep, Checkpoint
        ckpt = Checkpoint(path=str(tmp_path / "ckpt.yaml"))
        with patch("orchestrator.start_sglang", return_value=MagicMock()):
            with patch("orchestrator.wait_for_health", return_value=True):
                with patch("orchestrator.run_benchmark", return_value=1):
                    with patch("orchestrator.kill_sglang"):
                        with patch("orchestrator.wait_port_free", return_value=True):
                            run_sweep(self._make_config(), ckpt)
        assert not ckpt.is_done("org/modelA", 1, "gsm8k")
        assert not ckpt.is_done("org/modelA", 1, "numinamath")

    def test_marks_dataset_success_on_zero_runner_exit(self, tmp_path):
        from orchestrator import run_sweep, Checkpoint
        ckpt = Checkpoint(path=str(tmp_path / "ckpt.yaml"))
        with patch("orchestrator.start_sglang", return_value=MagicMock()):
            with patch("orchestrator.wait_for_health", return_value=True):
                with patch("orchestrator.run_benchmark", return_value=0):
                    with patch("orchestrator.kill_sglang"):
                        with patch("orchestrator.wait_port_free", return_value=True):
                            run_sweep(self._make_config(), ckpt)
        assert ckpt.is_done("org/modelA", 1, "gsm8k")
        assert ckpt.is_done("org/modelA", 1, "numinamath")

    def test_kill_sglang_called_in_finally_even_on_health_failure(self, tmp_path):
        from orchestrator import run_sweep, Checkpoint
        ckpt = Checkpoint(path=str(tmp_path / "ckpt.yaml"))
        mock_proc = MagicMock()
        with patch("orchestrator.start_sglang", return_value=mock_proc):
            with patch("orchestrator.wait_for_health", return_value=False):
                with patch("orchestrator.kill_sglang") as mock_kill:
                    with patch("orchestrator.wait_port_free", return_value=True):
                        run_sweep(self._make_config(), ckpt)
        mock_kill.assert_called_once_with(mock_proc)
