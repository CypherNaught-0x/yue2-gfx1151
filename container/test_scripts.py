"""CPU-only shell-wrapper contract tests; container engines are recording fakes.

Run: python -m unittest discover -s container -p 'test_*.py' -v
No actual container or GPU process is launched by these tests.
"""
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="yue2-wrapper-test-")
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.bin = self.work / "bin"
        self.bin.mkdir()
        python = sys.executable
        for engine in ("podman", "docker"):
            fake = self.bin / engine
            fake.write_text(f"#!{python}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
            fake.chmod(0o755)
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", IMAGE="localhost/yue2-gfx1151:test", CONTAINER_ENGINE="podman")
        self.env.pop("BASE_IMAGE", None)
        for name in ("model weights", "vae", "output", "source"):
            (self.work / name).mkdir()
        (self.work / "request.json").write_text("{}\n")
        self.options = ["--model", str(self.work / "model weights"), "--vae", str(self.work / "vae"), "--request", str(self.work / "request.json"), "--output", str(self.work / "output")]

    def run_script(self, script, args=()):
        return subprocess.run(["bash", str(ROOT / "scripts" / script), *args], env=self.env, text=True, capture_output=True)

    def test_missing_base_fails_before_engine(self):
        result = self.run_script("build-container.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Set BASE_IMAGE explicitly", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_short_base_rejected(self):
        self.env["BASE_IMAGE"] = "unqualified:latest"
        result = self.run_script("build-container.sh")
        self.assertEqual(result.returncode, 2)
        self.assertIn("fully qualified", result.stderr)

    def test_explicit_paths_required(self):
        result = self.run_script("run-container.sh", ["--dry-run"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("Explicit", result.stderr)

    def test_dry_run_has_only_selected_mounts_and_no_gpu(self):
        result = self.run_script("run-container.sh", [*self.options, "--dry-run"])
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)
        self.assertEqual(args.count("--mount"), 4)
        self.assertNotIn("--device", args)
        self.assertNotIn("--ipc=host", args)
        self.assertIn("--userns=keep-id", args)
        self.assertIn("--network=none", args)
        self.assertEqual(args[-1], "--dry-run")
        self.assertIn(f"type=bind,src={self.work / 'model weights'},dst=/inputs/model,readonly", args)
        self.assertIn(f"type=bind,src={self.work / 'output'},dst=/output", args)

    def test_smoke_flag_is_forwarded(self):
        result = self.run_script("run-container.sh", [*self.options, "--smoke", "--dry-run"])
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)
        self.assertIn("--smoke", args)
        self.assertNotIn("--device", args)

    def test_source_is_narrow_readonly_mount(self):
        result = self.run_script("run-container.sh", [*self.options, "--source", str(self.work / "source"), "--dry-run"])
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)
        self.assertEqual(args.count("--mount"), 5)
        self.assertIn(f"type=bind,src={self.work / 'source'},dst=/inputs/source,readonly", args)
        self.assertEqual(args[args.index("--source") + 1], "/inputs/source")

    def test_docker_uses_numeric_identity_without_podman_flags(self):
        self.env["CONTAINER_ENGINE"] = "docker"
        result = self.run_script("run-container.sh", [*self.options, "--dry-run"])
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)
        self.assertNotIn("--userns=keep-id", args)
        self.assertEqual(args[args.index("--user") + 1], f"{os.getuid()}:{os.getgid()}")
        self.assertNotIn("--device", args)

    def test_overlapping_output_rejected(self):
        options = list(self.options)
        options[-1] = str(self.work / "model weights")
        result = self.run_script("run-container.sh", [*options, "--dry-run"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("must not overlap", result.stderr)

    def test_missing_request_does_not_create_output(self):
        options = list(self.options)
        options[options.index("--request") + 1] = str(self.work / "missing.json")
        options[-1] = str(self.work / "new-output")
        result = self.run_script("run-container.sh", [*options, "--dry-run"])
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.work / "new-output").exists())


if __name__ == "__main__":
    unittest.main()
