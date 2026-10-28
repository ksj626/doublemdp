"""Read-only verification before queued OGBench jobs may claim data.

Quiescent byte checks are cached by filesystem identity, including ctime, for
at most one hour. Recent writes are rechecked because a filesystem may expose
coarse timestamps. This metadata cache is not a guarantee against deliberately
hidden in-place changes. Large-shard loading and checkpoint resume perform
additional identity checks. No downloader is invoked here.
"""
from __future__ import annotations
import hashlib
import json
import time
import zipfile
from pathlib import Path
import numpy as np

BASE = 'https://rail.eecs.berkeley.edu/datasets/ogbench/'
QUIESCENCE_SECONDS = 2.
MAX_CACHE_AGE_SECONDS = 3600.


def signature(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class DatasetReadiness:
    def __init__(self):
        self.checked = {}
        self.manifests = {}

    def _file(self, path, *, url=None, sha256=None, transitions=None):
        path = Path(path)
        before = signature(path)
        sidecar = path.with_suffix('.download.json')
        record_signature = signature(sidecar) if url else None
        key = (str(path.resolve()), before, record_signature, url, sha256, transitions)
        stamps = [before] + ([record_signature] if record_signature else [])
        quiescent = all(time.time()-max(stamp[3],stamp[4])/1e9 >= QUIESCENCE_SECONDS for stamp in stamps)
        cached = self.checked.get(key)
        if cached and quiescent and time.monotonic()-cached[1] < MAX_CACHE_AGE_SECONDS:
            return cached[0]
        if url:
            record = json.loads(sidecar.read_text())
            if record.get('url') != url or record.get('size') != before[2]:
                raise ValueError(f'Download identity/size mismatch: {path}')
            sha256 = record.get('sha256')
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f'Missing byte checksum: {path}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b''):
                digest.update(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError(f'Dataset checksum mismatch: {path}')
        with np.load(path, allow_pickle=False) as data:
            if not {'observations', 'actions', 'terminals'} <= set(data.files):
                raise ValueError(f'Incomplete OGBench archive: {path}')
            terminals = data['terminals']
            if terminals.ndim != 1 or not np.all((terminals == 0) | (terminals == 1)):
                raise ValueError(f'Invalid trajectory boundaries: {path}')
            count = int(np.sum(1-terminals))
        if transitions is not None and count != transitions:
            raise ValueError(f'Shard transition count differs: {path}')
        if signature(path) != before or (url and signature(sidecar) != record_signature):
            raise ValueError(f'Dataset changed during verification: {path}')
        value = (digest.hexdigest(), count)
        self.checked[key] = (value, time.monotonic())
        return value

    def _shards(self, config):
        path = Path(config['shard_manifest'])
        stamp = signature(path)
        old = self.manifests.get(str(path))
        quiescent = time.time()-max(stamp[3],stamp[4])/1e9 >= QUIESCENCE_SECONDS
        if (old is None or old[0] != stamp or not quiescent
                or time.monotonic()-old[2] >= MAX_CACHE_AGE_SECONDS):
            manifest = json.loads(path.read_text())
            if signature(path) != stamp:
                raise ValueError('Shard manifest changed while reading')
            self.manifests[str(path)] = (stamp, manifest, time.monotonic())
        manifest = self.manifests[str(path)][1]
        family = 'puzzle-4x4' if 'puzzle-4x4' in config['env'] else 'cube-quadruple'
        source = BASE + family + '-play-100m-v0/'
        if manifest.get('source') != source or manifest.get('total_transitions', 0) < 100_000_000:
            raise ValueError('Invalid source or total in 100M manifest')
        resolve = lambda value: (path.parent / value).resolve()
        shards = [resolve(value) for value in manifest['shards']]
        records = {resolve(r['path']): r for r in manifest['records']}
        if len(shards) != 100 or len(set(shards)) != 100 or len(manifest['records']) != 100 or set(records) != set(shards):
            raise ValueError('100M manifest requires 100 distinct training shards and records')
        validation = {resolve(value) for value in manifest['validation_shards']}
        expected_validation = {p.with_name(p.stem+'-val.npz') for p in shards}
        if validation != expected_validation or len(validation) != 100:
            raise ValueError('100M manifest requires every validation partner')
        hashes, total = set(), 0
        for shard in shards:
            record = records[shard]
            digest, count = self._file(shard, sha256=record['sha256'], transitions=record['transitions'])
            if digest in hashes:
                raise ValueError('Duplicate shard contents cannot certify 100M transitions')
            hashes.add(digest); total += count
            partner = shard.with_name(shard.stem+'-val.npz')
            self._file(partner, url=source+partner.name)
        if total != manifest['total_transitions'] or total < 100_000_000:
            raise ValueError('Actual shard total differs from 100M manifest')

    def __call__(self, config):
        """Return None when ready, otherwise a reason suitable for queue state."""
        try:
            if config.get('shard_manifest'):
                self._shards(config)
            elif 'singletask' in config.get('env', '') and not config.get('dataset'):
                parts = config['env'].split('-')
                base = '-'.join(parts[:parts.index('singletask')] + parts[-1:])
                for suffix in ('.npz', '-val.npz'):
                    path = Path(config['dataset_dir']) / (base+suffix)
                    self._file(path, url=BASE+path.name)
            return None
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
            return f'Dataset not ready: {error}'
