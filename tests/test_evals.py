import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_eval_set_is_consistent():
    result = subprocess.run([sys.executable, str(ROOT / "evals" / "run_evals.py"), "--check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all cases ok" in result.stdout
