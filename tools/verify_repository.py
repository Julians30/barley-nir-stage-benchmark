"""Verify the archived files and saved scientific results without model fitting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def verify_files():
    manifest = json.loads((ROOT / 'REPOSITORY_MANIFEST.json').read_text())
    for record in manifest['files']:
        path = ROOT / record['path']
        raw = path.read_bytes()
        assert len(raw) == record['bytes'], path
        assert hashlib.sha256(raw).hexdigest() == record['sha256'], path
        header = f'blob {len(raw)}\0'.encode()
        assert hashlib.sha1(header + raw).hexdigest() == record['git_blob_sha'], path
    original = json.loads((ROOT / '03_RESULTADOS/results_manifest.json').read_text())
    for record in original['files']:
        raw = (ROOT / '03_RESULTADOS' / record['name']).read_bytes()
        assert len(raw) == record['bytes'], record['name']
        assert hashlib.sha256(raw).hexdigest() == record['sha256'], record['name']
    print(f"Verified {len(manifest['files'])} repository files and {len(original['files'])} original result digests.", flush=True)


def main():
    subprocess.run([sys.executable, str(ROOT / 'tools/restore_dataset.py')], cwd=ROOT, check=True)
    verify_files()
    code = ROOT / '02_CODIGO_NOTEBOOKS'
    for script in ['audit_completed_results.py', 'audit_review_results.py', 'test_review_diagnostics.py']:
        subprocess.run([sys.executable, str(code / script)], cwd=ROOT, check=True)
    verify_files()
    print('PASS: archived integrity, prediction audits and VIP tests; no model fitting.')


if __name__ == '__main__':
    main()
