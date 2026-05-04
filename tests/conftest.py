import pytest


@pytest.fixture(autouse=True)
def disable_metrics_poller_by_default(monkeypatch):
    """Keep unit tests from starting background /metrics polling threads."""
    monkeypatch.setenv("METRICS_POLL_INTERVAL", "0")
