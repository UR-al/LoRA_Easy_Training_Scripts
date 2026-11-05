"""The Linux entry scripts the README tells users to run directly (./install.sh, ./install311.sh, run.sh, update.sh)
must be committed with the executable bit, or a fresh clone fails with "Permission denied" (exit 126); external review
2026-11-05. Git stores the mode, so the check reads the index; on Linux it also checks the checked-out file.

    venv\\Scripts\\python.exe -m unittest tests.test_script_modes
"""

import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ("install.sh", "install311.sh", "run.sh", "update.sh")


class ScriptModeTest(unittest.TestCase):
    def test_entry_scripts_are_executable(self):
        try:
            result = subprocess.run(["git", "ls-files", "-s", "--", *SCRIPTS], cwd=ROOT, capture_output=True,
                                    text=True, encoding="utf-8")
        except FileNotFoundError:
            self.skipTest("git is not installed")
        if result.returncode != 0:
            self.skipTest("not a git checkout")
        modes = {}
        for line in result.stdout.splitlines():
            meta, path = line.split("\t", 1)
            modes[path] = meta.split()[0]
        self.assertEqual(modes, {script: "100755" for script in SCRIPTS})
        if os.name == "posix":
            for script in SCRIPTS:
                self.assertTrue(os.access(ROOT / script, os.X_OK), script)


if __name__ == "__main__":
    unittest.main()
