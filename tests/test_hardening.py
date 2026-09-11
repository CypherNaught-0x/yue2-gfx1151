"""CPU regression fixtures for resume/manifest/path gates, not GPU evidence."""
import contextlib
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

from yue2_gfx1151 import cli, campaign, verification
import test_verification as audio_fixture


class HardeningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def manifest(self):
        value: dict = {key: {} for key in ('generation', 'source', 'launcher', 'weights', 'auxiliary', 'env')}
        value.update(schema_version=1, requests=[], torch='fixture', hip='fixture',
                     sequential=False, smoke=False, threads=8, budget=48)
        value['signature'] = verification.manifest_signature(value)
        return value

    def test_manifest_hash_is_recomputable_and_covers_all_fields(self):
        file = self.root/'campaign.json'
        manifest = self.manifest()
        campaign.write(file, manifest)
        self.assertEqual(verification.checked_manifest(file), manifest)
        for field, changed in (('smoke', True), ('threads', 2), ('budget', 10),
                               ('env', {'MIOPEN_FIND_MODE': 'NORMAL'}), ('requests', [{'id': 'other'}])):
            campaign.write(file, dict(manifest, **{field: changed}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                verification.checked_manifest(file)

    def test_legacy_incomplete_identity_is_not_resumable(self):
        file = self.root/'campaign.json'
        campaign.write(file, {'signature': 'copied-old-signature'})
        with self.assertRaises(ValueError): verification.checked_manifest(file)

    def test_nonempty_song_without_checkpoint_is_not_replanned(self):
        (self.root/'prefix.npy').write_bytes(b'partial output')
        with self.assertRaises(ValueError): campaign.checked(self.root)

    def test_empty_song_is_valid_planning_candidate(self):
        self.assertIsNone(campaign.checked(self.root))

    def test_verified_planned_resume_and_corruption(self):
        folder = self.root/'song'; folder.mkdir()
        names = ['plan.json', 'plan_manifest.json', 'prefix.npy', 'abc_tokens.npy']
        for name in names: (folder/name).write_bytes(b'hash-only checkpoint fixture')
        campaign.checkpoint(folder, 'planned', names)
        campaign.preflight_output(self.root, {'song'}, True)
        checked = campaign.checked(folder)
        assert checked is not None
        self.assertEqual(checked['stage'], 'planned')
        (folder/'prefix.npy').write_bytes(b'changed')
        with self.assertRaises(ValueError): campaign.preflight_output(self.root, {'song'}, True)

    def test_hardlinked_output_cannot_overwrite_another_file(self):
        out = self.root/'output'; out.mkdir()
        folder = out/'song'; folder.mkdir()
        victim = self.root/'victim'; victim.write_text('unchanged')
        os.link(victim, folder/'semantic.npy')
        with self.assertRaises(ValueError): campaign.preflight_output(out, {'song'}, True)
        self.assertEqual(victim.read_text(), 'unchanged')

    def test_unlisted_symlink_is_rejected_before_stage_writes(self):
        folder = self.root/'song'; folder.mkdir()
        outside = self.root.parent/(self.root.name + '-missing')
        (folder/'semantic.npy').symlink_to(outside)
        with self.assertRaises(ValueError): campaign.preflight_output(self.root, {'song'}, True)
        self.assertFalse(outside.exists())

    def test_dangling_checkpoint_symlink_is_not_missing_checkpoint(self):
        (self.root/'checkpoint.json').symlink_to(self.root/'missing')
        with self.assertRaises(ValueError): campaign.checked(self.root)

    def test_atomic_writer_does_not_follow_predictable_temp_symlink(self):
        victim = self.root/'victim'; victim.write_text('unchanged')
        (self.root/'status.json.tmp').symlink_to(victim)
        campaign.write(self.root/'status.json', {'state': 'fixture'})
        self.assertEqual(victim.read_text(), 'unchanged')
        self.assertEqual(verification.read_json(self.root/'status.json')['state'], 'fixture')

    def test_atomic_writer_rejects_destination_symlink(self):
        victim = self.root/'victim'; victim.write_text('unchanged')
        (self.root/'status.json').symlink_to(victim)
        with self.assertRaises(ValueError): campaign.write(self.root/'status.json', {})
        self.assertEqual(victim.read_text(), 'unchanged')

    def test_source_override_rejects_loaded_other_checkout(self):
        fake = types.SimpleNamespace(__file__=str(self.root/'elsewhere/yue2/__init__.py'))
        with patch.dict(sys.modules, {'yue2': fake}):
            with self.assertRaises(ValueError): cli.check_source_modules(self.root/'requested')

    def test_invalid_requests_rejected_in_cpu_validation(self):
        file = self.root/'request.json'
        base = {'id': 'song', 'style': 'Piano', 'lyrics': '[Instrumental]'}
        for change in ({'id': 'campaign.json'}, {'seed': -1}, {'seed': 2**63},
                       {'abc': ''}, {'abc': 'notes', 'cot': 'off'}):
            file.write_text(json.dumps(dict(base, **change)))
            with self.subTest(change=change), self.assertRaises(ValueError): cli.load_requests(file)

    def test_nonfinite_budget_and_output_input_overlap(self):
        for model in ('model', 'vae'):
            (self.root/model).mkdir(); (self.root/model/'config.json').write_text('{}')
        request = self.root/'request.json'
        request.write_text(json.dumps({'id': 'song', 'style': 'Piano', 'lyrics': '[Instrumental]'}))
        argv = ['generate', '--model', str(self.root/'model'), '--vae', str(self.root/'vae'),
                '--request', str(request), '--output', str(self.root/'output')]
        for option in (['--budget', 'nan'], ['--budget', 'inf'],
                       ['--gpu-lock', str(self.root/'output/lock')],
                       ['--generation-config', str(self.root/'output/config.json')]):
            with self.subTest(option=option), self.assertRaises(ValueError):
                cli.validate(cli.parser().parse_args(argv + option))
        (self.root/'model/pipeline.json').write_text('{"model":"../other"}')
        with self.assertRaises(ValueError): cli.validate(cli.parser().parse_args(argv))

    def test_real_dependency_free_cli_process(self):
        for model in ('model', 'vae'):
            (self.root/model).mkdir(); (self.root/model/'config.json').write_text('{}')
        request = self.root/'request.json'
        request.write_text(json.dumps({'id': 'cpu-fixture', 'style': 'Synthetic validation only', 'lyrics': '[Instrumental]'}))
        argv = [sys.executable, '-S', '-m', 'yue2_gfx1151', 'generate',
                '--model', str(self.root/'model'), '--vae', str(self.root/'vae'),
                '--request', str(request), '--output', str(self.root/'output'), '--dry-run']
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
        result = subprocess.run(argv, cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report['gpu_exercised'])
        self.assertFalse(report['model_weights_verified'])
        self.assertFalse((self.root/'output').exists())
        rejected = subprocess.run(argv + ['--budget', 'nan'], cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(rejected.returncode, 2, rejected.stderr)


@unittest.skipUnless(audio_fixture.HAS_AUDIO, 'CPU audio tests need numpy and soundfile')
class AudioHardeningTests(unittest.TestCase):
    root: Path
    song: Path

    def setUp(self):
        fixture = audio_fixture.VerificationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.song = fixture.root, fixture.song

    def mutate_result(self, **changes):
        result = verification.read_json(self.song/'result.json')
        result.update(changes)
        campaign.write(self.song/'result.json', result)
        checkpoint = verification.read_json(self.song/'checkpoint.json')
        checkpoint['files']['result.json'] = cli.digest(self.song/'result.json')
        campaign.write(self.song/'checkpoint.json', checkpoint)

    def test_missing_or_nonboolean_truncation_cannot_pass(self):
        for value in ({}, {'abc': False}, {'abc': 0, 'semantic': False}):
            self.mutate_result(truncated=value)
            with self.subTest(value=value), self.assertRaises(ValueError): verification.verify(self.root, 1)

    def test_truncation_requires_explicit_acceptance(self):
        self.mutate_result(truncated={'abc': False, 'semantic': True})
        with self.assertRaises(ValueError): verification.verify(self.root, 1)
        self.assertEqual(verification.verify(self.root, 1, True)['technical_validation'], 'passed')

    def test_nonfinite_duration_cannot_pass(self):
        # Deliberately construct malformed JSON accepted by Python's decoder.
        file = self.song/'result.json'
        result = verification.read_json(file); result['audio_seconds'] = float('nan')
        file.write_text(json.dumps(result))
        checkpoint = verification.read_json(self.song/'checkpoint.json')
        checkpoint['files']['result.json'] = cli.digest(file)
        campaign.write(self.song/'checkpoint.json', checkpoint)
        with self.assertRaises(ValueError): verification.verify(self.root, 1)

    def test_failed_attempt_cannot_reuse_old_completed_summary(self):
        campaign.write(self.root/'status.json', {'state': 'failed'})
        with self.assertRaises(ValueError): verification.verify(self.root, 1)

    def test_symlink_metadata_cannot_pass(self):
        for name in ('result.json', 'checkpoint.json'):
            file = self.song/name; backup = self.root/('backup-' + name)
            file.rename(backup); file.symlink_to(backup)
            with self.subTest(name=name), self.assertRaises(ValueError): verification.verify(self.root, 1)
            file.unlink(); backup.rename(file)

    def test_mutable_checkpoint_is_not_an_audio_artifact(self):
        artifacts = verification.read_json(self.song/'result.json')['artifacts']
        artifacts['checkpoint.json'] = {'sha256': cli.digest(self.song/'checkpoint.json'),
                                        'bytes': (self.song/'checkpoint.json').stat().st_size}
        self.mutate_result(artifacts=artifacts)
        with self.assertRaises(ValueError): verification.verify(self.root, 1)

    def test_campaign_binding_and_hash_round_trip(self):
        from yue2.protocol import SongRequest
        request = SongRequest(id='fixture', style='Synthetic test', lyrics='[Instrumental]').to_dict()
        campaign.write(self.song/'request.json', request)
        result = verification.read_json(self.song/'result.json')
        result['artifacts']['request.json'] = {'sha256': cli.digest(self.song/'request.json'),
                                              'bytes': (self.song/'request.json').stat().st_size}
        result['weights'] = {}
        result['identity'] = verification.manifest_signature({'request': request, 'config': {}, 'weights': {}})
        self.mutate_result(**result)
        checkpoint = verification.read_json(self.song/'checkpoint.json')
        checkpoint['files']['request.json'] = cli.digest(self.song/'request.json')
        campaign.write(self.song/'checkpoint.json', checkpoint)
        manifest = HardeningTests.manifest(self)
        manifest['requests'] = [request]; manifest['signature'] = verification.manifest_signature(manifest)
        campaign.write(self.root/'campaign.json', manifest)
        self.assertTrue(verification.verify(self.root, 1)['campaign_identity_verified'])
        manifest['requests'][0]['style'] = 'Changed request'
        manifest['signature'] = verification.manifest_signature(manifest)
        campaign.write(self.root/'campaign.json', manifest)
        with self.assertRaises(ValueError): verification.verify(self.root, 1)


if __name__ == '__main__': unittest.main()
