import json
import random

from experiments.seed import fix_seeds
from repro.env_capture import write_environment_snapshot


def test_fix_seeds(tmp_path):
    fix_seeds(123)
    values = [random.random() for _ in range(3)]
    fix_seeds(123)
    assert values == [random.random() for _ in range(3)]


def test_env_capture(tmp_path):
    path = tmp_path / "env.json"
    metadata = write_environment_snapshot(path)
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == metadata
