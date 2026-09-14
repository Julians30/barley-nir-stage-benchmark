"""Reconstruct the byte-identical prepared NPZ from hash-verified binary parts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile

DATA = Path(__file__).resolve().parents[1] / '01_DATASET'


def main():
    info = json.loads((DATA / 'barley_nir_long.parts.json').read_text())
    target = DATA / info['filename']
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != info['sha256']:
            raise RuntimeError('Existing NPZ differs from the archive; refusing to overwrite it.')
        print('Dataset already present; original SHA-256 verified.')
        return
    digest, size = hashlib.sha256(), 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=DATA, prefix='dataset_restore_', delete=False) as tmp:
            temporary = Path(tmp.name)
            for part in info['parts']:
                raw = (DATA / part['path']).read_bytes()
                if len(raw) != part['bytes'] or hashlib.sha256(raw).hexdigest() != part['sha256']:
                    raise RuntimeError(f"Dataset part failed integrity checks: {part['path']}")
                tmp.write(raw)
                digest.update(raw)
                size += len(raw)
        if size != info['bytes'] or digest.hexdigest() != info['sha256']:
            raise RuntimeError('Reconstructed dataset failed total size or SHA-256 verification.')
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f'Reconstructed {target.name}: {size} bytes; original SHA-256 verified.')


if __name__ == '__main__':
    main()
