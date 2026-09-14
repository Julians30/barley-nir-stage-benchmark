"""Prepare a new output project without copying any archived result checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def prepare(destination):
    destination = Path(destination).resolve()
    if destination == ROOT or ROOT in destination.parents:
        raise RuntimeError('Destination must be outside the archived project.')
    if destination.exists():
        raise RuntimeError('Destination already exists; choose a new directory to preserve prior runs.')
    subprocess.run([sys.executable, str(ROOT/'tools/restore_dataset.py')], check=True)
    data = ROOT/'01_DATASET/barley_nir_long.npz'
    expected = json.loads((ROOT/'01_DATASET/barley_nir_long.parts.json').read_text())['sha256']
    if hashlib.sha256(data.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Dataset digest differs from the archive.')
    destination.mkdir(parents=True)
    try:
        (destination/'01_DATASET').mkdir()
        (destination/'02_CODIGO_NOTEBOOKS').mkdir()
        (destination/'03_RESULTADOS').mkdir()
        shutil.copy2(data,destination/'01_DATASET'/data.name)
        for path in (ROOT/'01_DATASET').iterdir():
            if path.is_file() and path.suffix in ['.txt','.md']:
                shutil.copy2(path,destination/'01_DATASET'/path.name)
        scripts = {}
        for path in (ROOT/'02_CODIGO_NOTEBOOKS').glob('*.py'):
            shutil.copy2(path,destination/'02_CODIGO_NOTEBOOKS'/path.name)
            scripts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(ROOT/'requirements.txt',destination/'requirements.txt')
        provenance = {'prepared_utc':datetime.now(timezone.utc).isoformat(),
                      'dataset_sha256':expected,'script_sha256':scripts,
                      'archived_results_or_checkpoints_copied':False,
                      'training_started':False,
                      'scope':'Fresh output project; original archive is untouched. No pre-executed notebook or archived integrity manifest is copied.'}
        (destination/'RECOMPUTATION_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    except Exception:
        shutil.rmtree(destination)
        raise
    return destination


def main(destination,prepare_only=False):
    destination = prepare(destination)
    print(f'Fresh output project prepared: {destination}',flush=True)
    if prepare_only:
        return
    provenance_path = destination/'RECOMPUTATION_PROVENANCE.json'
    provenance = json.loads(provenance_path.read_text())
    provenance['training_started'] = True
    provenance_path.write_text(json.dumps(provenance,indent=2)+'\n')
    code = destination/'02_CODIGO_NOTEBOOKS'
    subprocess.run([sys.executable,str(code/'run_nested_validation.py'),'--project-dir',str(destination),'--force'],check=True)
    metadata_path = destination/'03_RESULTADOS/nested_run_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['released_script_sha256'] = provenance['script_sha256']['run_nested_validation.py']
    metadata['released_script_note'] = 'Fresh fit from copied release script; no archived result checkpoints copied.'
    metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
    subprocess.run([sys.executable,str(code/'run_review_additions.py'),'--project-dir',str(destination)],check=True)
    for script in ['audit_completed_results.py','audit_review_results.py','test_review_diagnostics.py']:
        subprocess.run([sys.executable,str(code/script)],check=True)
    provenance['training_and_saved_result_audits_completed'] = True
    provenance['completed_utc'] = datetime.now(timezone.utc).isoformat()
    provenance_path.write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',required=True,type=Path,help='New directory outside the archived project.')
    parser.add_argument('--prepare-only',action='store_true',help='Copy verified data and scripts without fitting models.')
    args=parser.parse_args();main(args.destination,args.prepare_only)
