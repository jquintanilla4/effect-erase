from pathlib import Path
import tomllib
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "worker" / "pyproject.toml"


class WorkerDependencyTests(unittest.TestCase):
    def test_supervision_dependency_stays_on_headless_compatible_series(self):
        pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
        dependencies = pyproject["project"]["dependencies"]

        self.assertIn("supervision>=0.23.0,<0.24.0", dependencies)

    def test_worker_declares_google_genai_dependency_for_void_phase_two(self):
        pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
        dependencies = pyproject["project"]["dependencies"]

        self.assertIn("google-genai>=1.33.0", dependencies)


if __name__ == "__main__":
    unittest.main()
