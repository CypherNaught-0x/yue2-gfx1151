import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from yue2_gfx1151 import cli

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('model', 'vae'):
            (self.root/name).mkdir()
            (self.root/name/'config.json').write_text('{}')
        self.request = self.root/'request.json'
        self.row = {'id': 'public-example', 'style': 'Instrumental piano', 'lyrics': '[Instrumental]', 'seed': 42}
        self.request.write_text(json.dumps(self.row))
        self.argv = ['generate', '--model', str(self.root/'model'), '--vae', str(self.root/'vae'),
                     '--request', str(self.request), '--output', str(self.root/'output')]

    def test_dry_run_is_cpu_only_and_does_not_create_output(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(self.argv + ['--dry-run']), 0)
        report = json.loads(output.getvalue())
        self.assertFalse(report['gpu_exercised'])
        self.assertFalse(report['model_weights_verified'])
        self.assertFalse((self.root/'output').exists())
        self.assertNotIn('torch', sys.modules)

    def test_invalid_ids(self):
        for value in ('../escape', '/absolute', 'a/b', 'a\\b', '.', '..', '', None, 'x\n'):
            with self.subTest(value=value), self.assertRaises(ValueError): cli.safe_name(value)

    def test_duplicate_and_oversized_batches(self):
        for rows in ([self.row, self.row], [dict(self.row, id=str(i)) for i in range(5)], []):
            self.request.write_text(json.dumps(rows))
            with self.assertRaises(ValueError): cli.load_requests(self.request)

    def test_cfg_and_unknown_keys(self):
        for extra in ({'cfg_scale': 1.01}, {'abc_path': '../secret'}, {'seed': True}, {'abc': 3}):
            self.request.write_text(json.dumps(dict(self.row, **extra)))
            with self.assertRaises(ValueError): cli.load_requests(self.request)

    def test_off_mode_is_explicit_cfg_one(self):
        self.request.write_text(json.dumps(dict(self.row, cot='off')))
        self.assertEqual(cli.load_requests(self.request)[0]['cfg_scale'], 1.0)

    def test_nonempty_output(self):
        output = self.root/'output'; output.mkdir(); (output/'file').write_text('fixture')
        with self.assertRaises(FileExistsError): cli.validate(cli.parser().parse_args(self.argv))

    def test_input_output_overlap(self):
        args = cli.parser().parse_args(self.argv); args.output = args.model
        with self.assertRaises(ValueError): cli.validate(args)

    def test_source_resolution(self):
        self.assertEqual(cli.source_root(ROOT), ROOT/'src')
        self.assertEqual(cli.source_root(ROOT/'src'), ROOT/'src')
        with self.assertRaises(ValueError): cli.source_root(self.root)

    def test_smoke_and_override_conflict(self):
        args = cli.parser().parse_args(self.argv + ['--smoke', '--generation-config', 'unused'])
        with self.assertRaises(ValueError): cli.validate(args)

    def test_help_from_arbitrary_working_directory(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT/'src'))
        result = subprocess.run([sys.executable, '-S', '-m', 'yue2_gfx1151', '--help'],
                                cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('generate', result.stdout)

    def test_model_download_revisions_and_explicit_paths(self):
        calls = []
        module = types.SimpleNamespace(snapshot_download=lambda **kw: calls.append(kw))
        args = types.SimpleNamespace(model=str(self.root/'download-model'), vae=str(self.root/'download-vae'))
        with patch.dict(sys.modules, {'huggingface_hub': module}), contextlib.redirect_stdout(io.StringIO()):
            cli.download(args)
        self.assertEqual([c['revision'] for c in calls], [cli.MODEL_REVISION, cli.VAE_REVISION])
        self.assertTrue(all(len(c['revision']) == 40 for c in calls))
        self.assertEqual(calls[0]['repo_id'], 'm-a-p/YuE2-3B')
        self.assertEqual(calls[1]['local_dir'], args.vae)

    def test_pinned_vendored_source(self):
        provenance = json.loads((ROOT/'docs/source-provenance.json').read_text())
        files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'src/yue2').glob('*.py')}
        self.assertEqual(files, {k: v['sha256'] for k, v in provenance['files'].items()})

    def test_public_benchmark_help_without_dependencies(self):
        for file in (ROOT/'benchmarks').glob('*.py'):
            if file.name.startswith('_'): continue
            with self.subTest(file=file.name):
                result = subprocess.run([sys.executable, '-S', str(file), '--help'], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_importable_native_protocol(self):
        from yue2.protocol import GenerationConfig
        cfg = GenerationConfig()
        self.assertEqual(cfg.ode_steps, 32)
        self.assertEqual(cfg.ode_method, 'midpoint')
        with self.assertRaises(ValueError): GenerationConfig.from_dict({'semantic': {'max_tokens': 0}})


if __name__ == '__main__': unittest.main()
