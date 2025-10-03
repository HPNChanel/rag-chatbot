from pathlib import Path

from experiments.tracking import Tracker


def test_tracker_logs(tmp_path: Path):
    tracker = Tracker(tmp_path)
    tracker.log_params(alpha=1)
    tracker.log_metric("recall", 0.5, step=1)
    artifact = tmp_path / "foo.txt"
    artifact.write_text("hello", encoding="utf-8")
    copied = tracker.log_artifact(artifact)
    assert copied.exists()
    assert "recall" in tracker.metrics_csv.read_text(encoding="utf-8")
