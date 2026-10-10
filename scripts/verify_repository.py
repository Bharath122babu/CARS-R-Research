"""Check documentation links/figures, standalone paths and optional asset hashes."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
errors = []
markdown = [ROOT/'README.md', ROOT/'CONTRIBUTING.md', *sorted((ROOT/'docs').glob('*.md'))]
for path in markdown:
    value = path.read_text(encoding='utf-8')
    links = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', value)
    links += re.findall(r'(?:src|href)="([^"]+)"', value)
    for link in links:
        if link.startswith(('https://','http://','#','mailto:')): continue
        destination = (path.parent / link.split('#',1)[0]).resolve()
        if not destination.exists(): errors.append(f'{path.relative_to(ROOT)}: missing {link}')
    if re.search(r'[A-Z]:[\\/]|/Users/|/home/', value):
        errors.append(f'{path.relative_to(ROOT)}: personal absolute path')
for name in ('hero','architecture','operator','speedup','ownership','workflow'):
    ET.parse(ROOT/f'docs/assets/{name}.svg')
    if not (ROOT/f'docs/assets/{name}.png').exists(): errors.append(f'Missing PNG: {name}')
for name in ('cars_fast.py','cars_epoch.py'):
    source = (ROOT/name).read_text(encoding='utf-8')
    compile(source, name, 'exec')
    if 'PROTECTED_ROOT' in source or 'ROOT.parents[' in source:
        errors.append(f'{name}: refers outside the standalone repository')
assets = json.loads((ROOT/'docs/local_assets_manifest.json').read_text())
checked, missing = 0, 0
for relative, digest in assets.items():
    path = ROOT/relative
    if not path.exists(): missing += 1; continue
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        errors.append(f'Asset hash changed: {relative}')
    checked += 1
if errors: raise SystemExit('\n'.join(errors))
print(json.dumps({'status':'PASS','markdown_documents':len(markdown), 'diagram_pairs':6,
                  'local_assets_verified':checked, 'optional_local_assets_absent':missing},indent=2))
