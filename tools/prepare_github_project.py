"""Package an existing completed analysis without changing its scientific files."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

from lxml import etree as E

URL = 'https://github.com/Julians30/barley-nir-stage-benchmark'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG = 'http://schemas.openxmlformats.org/package/2006/relationships'


def update_word(source, target):
    with ZipFile(source) as zin:
        document = E.fromstring(zin.read('word/document.xml'))
        rels = E.fromstring(zin.read('word/_rels/document.xml.rels'))
        matches = [t for t in document.iter(f'{{{W}}}t')
                   if '[GitHub repository URL to be inserted]' in (t.text or '')]
        assert len(matches) == 1
        text_node = matches[0]
        run = text_node.getparent()
        paragraph = run.getparent()
        assert run.tag == f'{{{W}}}r' and paragraph.tag == f'{{{W}}}p'
        text = text_node.text.replace('are prepared for release at', 'are archived in the project repository at')
        before, after = text.split('[GitHub repository URL to be inserted]')
        text_node.text = before
        text_node.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        used = {x.get('Id') for x in rels}
        count = 1
        while f'rIdGitHub{count}' in used:
            count += 1
        rid = f'rIdGitHub{count}'
        E.SubElement(rels, f'{{{PKG}}}Relationship', Id=rid,
                     Type=R + '/hyperlink', Target=URL, TargetMode='External')
        hyperlink = E.Element(f'{{{W}}}hyperlink', {f'{{{R}}}id': rid})
        link_run = E.SubElement(hyperlink, f'{{{W}}}r')
        if run.find(f'{{{W}}}rPr') is not None:
            link_run.append(deepcopy(run.find(f'{{{W}}}rPr')))
        E.SubElement(link_run, f'{{{W}}}t').text = URL
        index = paragraph.index(run)
        paragraph.insert(index + 1, hyperlink)
        suffix = deepcopy(run)
        suffix.find(f'{{{W}}}t').text = after
        paragraph.insert(index + 2, suffix)
        xml = E.tostring(document, xml_declaration=True, encoding='UTF-8', standalone=True)
        rel_xml = E.tostring(rels, xml_declaration=True, encoding='UTF-8', standalone=True)
        with ZipFile(target, 'w') as zout:
            for entry in zin.infolist():
                content = zin.read(entry.filename)
                if entry.filename == 'word/document.xml':
                    content = xml
                elif entry.filename == 'word/_rels/document.xml.rels':
                    content = rel_xml
                zout.writestr(entry, content)


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--target', required=True, type=Path)
    args = parser.parse_args()
    root, repo = args.source_root.resolve(), args.target.resolve()
    out = root / 'project_artifacts'
    manifest = json.loads((out / 'results_manifest.json').read_text())
    for item in manifest['files']:
        source = out / item['name']
        raw = source.read_bytes()
        assert len(raw) == item['bytes'], source
        assert hashlib.sha256(raw).hexdigest() == item['sha256'], source
        copy_file(source, repo / '03_RESULTADOS' / item['name'])
    copy_file(out / 'results_manifest.json', repo / '03_RESULTADOS/results_manifest.json')
    for name in ['barley_nir_long.npz', 'DATASET_NOTES.md', 'DATASET_LICENSE.txt']:
        copy_file(out / name, repo / '01_DATASET' / name)
    for name in ['README.txt', 'LICENSE.txt']:
        copy_file(root / 'barley_data_raw' / name, repo / '01_DATASET' / ('SOURCE_' + name))
    for name in ['run_nested_validation.py', 'run_review_additions.py', 'audit_completed_results.py',
                 'audit_review_results.py', 'test_review_diagnostics.py', 'build_methodology_workflow.py',
                 'build_validation_notebook.py']:
        copy_file(root / name, repo / '02_CODIGO_NOTEBOOKS' / name)
    copy_file(out / '02_VALIDACION_ANIDADA_CEBADA_NIR.ipynb', repo / '02_CODIGO_NOTEBOOKS/02_VALIDACION_ANIDADA_CEBADA_NIR.ipynb')
    copy_file(out / 'requirements_reproduction.txt', repo / 'requirements.txt')
    copy_file(Path(__file__), repo / 'tools/prepare_github_project.py')
    dataset = (repo / '01_DATASET/barley_nir_long.npz').read_bytes()
    parts = []
    for index, start in enumerate(range(0, len(dataset), 1048576), 1):
        raw = dataset[start:start + 1048576]
        part = Path('parts') / f'barley_nir_long.npz.part{index:03d}'
        path = repo / '01_DATASET' / part
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        parts.append({'path': str(part), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    info = {'filename': 'barley_nir_long.npz', 'bytes': len(dataset),
            'sha256': hashlib.sha256(dataset).hexdigest(), 'parts': parts}
    (repo / '01_DATASET/barley_nir_long.parts.json').write_text(json.dumps(info, indent=2) + '\n')
    updated = out / 'ARTICULO_CEBADA_NIR_GITHUB.docx'
    update_word(out / 'ARTICULO_CEBADA_NIR_CORREGIDO_WORKFLOW.docx', updated)
    copy_file(updated, repo / '04_MANUSCRITO_CCIS' / updated.name)
    status = {'repository_url': URL, 'visibility': 'private',
              'branch': 'main', 'primary_run_signature': manifest['run_signature'],
              'zenodo_published_version_doi': None,
              'public_release_authorized': False,
              'software_license_confirmed': False,
              'historical_review_templates_preserved_unchanged': True,
              'current_manuscript': '04_MANUSCRITO_CCIS/' + updated.name}
    (repo / 'REPOSITORY_STATUS.json').write_text(json.dumps(status, indent=2) + '\n')
    records = []
    for path in sorted(repo.rglob('*')):
        if not path.is_file() or path.name == 'REPOSITORY_MANIFEST.json' or path.suffix in ['.npz', '.pyc']:
            continue
        raw = path.read_bytes()
        records.append({'path': str(path.relative_to(repo)), 'bytes': len(raw),
                        'sha256': hashlib.sha256(raw).hexdigest(),
                        'git_blob_sha': hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()})
    (repo / 'REPOSITORY_MANIFEST.json').write_text(json.dumps({'files': records}, indent=2) + '\n')
    print(json.dumps({'files': len(records) + 1, 'bytes': sum(x['bytes'] for x in records),
                      'updated_word': str(updated), 'repository_root': str(repo)}))


if __name__ == '__main__':
    main()
