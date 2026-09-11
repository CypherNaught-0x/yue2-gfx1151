"""Synthetic CPU-only audio fixtures; not generated-song evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from yue2_gfx1151.cli import digest
from yue2_gfx1151.verification import verify

HAS_AUDIO = all(importlib.util.find_spec(x) is not None for x in ('numpy', 'soundfile'))

@unittest.skipUnless(HAS_AUDIO, 'CPU audio tests need numpy and soundfile')
class VerificationTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import soundfile as sf
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.song = self.root/'fixture'; self.song.mkdir()
        # A synthetic tone is solely a format/hash gate fixture, never a YuE2 result.
        wave = .1*np.sin(2*np.pi*440*np.arange(6*48000)/48000)
        sf.write(self.song/'audio.flac', np.column_stack((wave, wave)), 48000)
        for name in ('prefix.npy','semantic.npy','latent.npy'):
            np.save(self.song/name, np.zeros(4, dtype=np.float32))
        for name in ('request.json','config.json'): (self.song/name).write_text('{}')
        result = {'status':'complete','audio_seconds':6.0,'truncated':{'abc':False,'semantic':False},
                  'artifacts':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in self.song.iterdir()}}
        (self.song/'result.json').write_text(json.dumps(result))
        (self.song/'checkpoint.json').write_text(json.dumps({'stage':'complete','files':{p.name:digest(p) for p in self.song.iterdir()}}))
        (self.root/'summary.json').write_text(json.dumps({'state':'completed','songs':[{'id':'fixture'}]}))

    def test_full_decode_and_hashes(self):
        self.assertEqual(verify(self.root, 1)['technical_validation'], 'passed')

    def test_corruption_rejected(self):
        (self.song/'semantic.npy').write_bytes(b'corrupt fixture')
        with self.assertRaises(ValueError): verify(self.root, 1)

    def test_count_mismatch_rejected(self):
        with self.assertRaises(ValueError): verify(self.root, 2)
