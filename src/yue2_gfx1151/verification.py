"""Full CPU artifact validation; signal checks are not listening acceptance."""
import hashlib
import json
import math
from pathlib import Path
from .cli import safe_name, digest


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Missing or unsafe metadata: ' + str(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('Metadata must be a JSON object: ' + str(path))
    return value


def manifest_signature(manifest):
    """Hash all persisted identity fields, never the signature itself."""
    fields = {k: v for k, v in manifest.items() if k != 'signature'}
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def checked_manifest(path):
    manifest = read_json(path)
    required = {'schema_version', 'requests', 'generation', 'source', 'launcher', 'weights',
                'auxiliary', 'torch', 'hip', 'env', 'sequential', 'smoke', 'threads', 'budget', 'signature'}
    if not required <= manifest.keys() or manifest['schema_version'] != 1:
        raise ValueError('Unsupported campaign identity; use a fresh output directory')
    if manifest_signature(manifest) != manifest['signature']:
        raise ValueError('Corrupt campaign identity')
    return manifest


def verify_song(folder, allow_truncated=False, request=None):
    import numpy as np
    import soundfile as sf
    folder = Path(folder)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError('Missing or symlink song directory')
    result = read_json(folder / 'result.json')
    if result.get('status') != 'complete':
        raise ValueError('Incomplete song')
    truncated = result.get('truncated')
    if not isinstance(truncated, dict) or set(truncated) != {'abc', 'semantic'} or any(type(v) is not bool for v in truncated.values()):
        raise ValueError('Missing or invalid truncation flags')
    if not allow_truncated and any(truncated.values()):
        raise ValueError('Truncated output; only explicit smoke acceptance permits it')
    required = {'audio.flac', 'prefix.npy', 'semantic.npy', 'latent.npy', 'request.json', 'config.json'}
    artifacts = result.get('artifacts', {})
    if not isinstance(artifacts, dict) or not required <= artifacts.keys():
        raise ValueError('Missing required artifacts')
    if {'checkpoint.json', 'result.json'} & artifacts.keys():
        raise ValueError('Mutable/self-referential metadata is not a result artifact')
    for name, value in artifacts.items():
        file = folder / safe_name(name)
        if not isinstance(value, dict) or file.is_symlink() or not file.is_file() or digest(file) != value.get('sha256') or file.stat().st_size != value.get('bytes'):
            raise ValueError('Corrupt result artifact: ' + name)
    checkpoint = read_json(folder / 'checkpoint.json')
    if checkpoint.get('stage') != 'complete':
        raise ValueError('Incomplete checkpoint')
    files = checkpoint.get('files', {})
    if not isinstance(files, dict) or not {'audio.flac', 'result.json'} <= files.keys():
        raise ValueError('Missing checkpoint audio/result')
    if 'checkpoint.json' in files:
        raise ValueError('Self-referential checkpoint')
    for name, expected_hash in files.items():
        file = folder / safe_name(name)
        if file.is_symlink() or not file.is_file() or digest(file) != expected_hash:
            raise ValueError('Corrupt checkpoint: ' + name)
    if request is not None:
        if read_json(folder / 'request.json') != request:
            raise ValueError('Saved result request disagrees with campaign')
        config = read_json(folder / 'config.json')
        fields = {'request': request, 'config': config, 'weights': result.get('weights')}
        if manifest_signature(fields) != result.get('identity'):
            raise ValueError('Saved result identity mismatch')
    seconds = result.get('audio_seconds')
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or result.get('sample_rate', 48000) != 48000:
        raise ValueError('Invalid audio metadata')
    energy = 0.0; samples = 0; peak = 0.0
    with sf.SoundFile(folder / 'audio.flac') as stream:
        if stream.samplerate != 48000 or stream.channels != 2:
            raise ValueError('Expected stereo 48 kHz audio')
        frames = len(stream)
        for block in stream.blocks(blocksize=48000, dtype='float64', always_2d=True):
            if not np.isfinite(block).all():
                raise ValueError('Nonfinite audio')
            energy += float(np.square(block).sum()); samples += block.size
            peak = max(peak, float(np.abs(block).max()))
    duration = frames / 48000
    if samples != frames * 2 or duration <= 5 or abs(duration - seconds) >= .002:
        raise ValueError('Audio duration/decode mismatch')
    rms = float(np.sqrt(energy / samples))
    if rms < 1e-5:
        raise ValueError('Silent audio')
    return {'id': folder.name, 'audio_seconds': duration, 'rms': rms, 'peak': peak,
            'hashes_verified': True, 'full_flac_decode_verified': True, 'truncated': truncated}


def verify(output, expected, allow_truncated=False):
    root = Path(output).resolve()
    summary = read_json(root / 'summary.json')
    if type(expected) is not int or expected < 1 or summary.get('state') != 'completed':
        raise ValueError('Campaign did not complete or invalid expected count')
    status = root / 'status.json'
    if (status.exists() or status.is_symlink()) and read_json(status).get('state') != 'completed':
        raise ValueError('Campaign status is not completed')
    songs = summary.get('songs', [])
    if not isinstance(songs, list) or any(not isinstance(r, dict) or 'id' not in r for r in songs):
        raise ValueError('Invalid song summary')
    ids = [safe_name(r['id']) for r in songs]
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError('Unexpected or duplicate song count')
    manifest = root / 'campaign.json'
    campaign_verified = manifest.exists() or manifest.is_symlink()
    requests = {}
    if campaign_verified:
        campaign = checked_manifest(manifest)
        from yue2.protocol import SongRequest
        for row in campaign['requests']:
            request = SongRequest(**row).to_dict()
            if request['id'] in requests:
                raise ValueError('Duplicate campaign request')
            requests[request['id']] = request
        if set(requests) != set(ids):
            raise ValueError('Campaign and summary request IDs disagree')
    rows = [verify_song(root / rid, allow_truncated, requests.get(rid)) for rid in ids]
    return {'technical_validation': 'passed', 'expected_count': expected,
            'campaign_identity_verified': campaign_verified,
            'subjective_listening_verified': False, 'songs': rows}
