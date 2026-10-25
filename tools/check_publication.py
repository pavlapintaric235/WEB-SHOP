"""Inspect publication files without printing secrets or uploading anything."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PATTERNS={
 'AWS access key':re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
 'private key':re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----'),
 'literal service credential':re.compile(r'''(?i)(?:aws_secret_access_key|runpod_api_key|WEBSHOP_DINO_TOKEN|CLOUDFLARE_API_TOKEN)\s*[=:]\s*["']([A-Za-z0-9_+=/-]{24,})["']''')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='docs/release_files.json');parser.add_argument('--tracked',action='store_true')
    args=parser.parse_args();manifest=ROOT/args.manifest
    proposed=set(json.loads(manifest.read_text()) if manifest.exists() else []);files=set(proposed)
    if args.tracked:files.update(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
    findings=[];warnings=[];checked=0
    for name in sorted(files-{''}):
        path=ROOT/name
        if not path.exists():
            if name in proposed:findings.append(dict(file=name,reason='missing release file'))
            continue
        if path.stat().st_size>5*1024*1024:findings.append(dict(file=name,reason='over 5 MiB; explicit artifact review required'))
        if path.suffix.lower() in {'.sqlite','.sqlite3','.db','.pth','.pt','.engine','.safetensors','.npy','.npz','.pem','.pfx','.key'}:findings.append(dict(file=name,reason='private/generated artifact'))
        if path.suffix.lower() not in {'.py','.md','.json','.yml','.yaml','.toml','.ps1','.txt','.html','.js','.css','.svg'}:continue
        checked+=1
        try:text=path.read_text(encoding='utf-8')
        except UnicodeError:continue
        for kind,pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                if 'EXAMPLE' not in match.group():findings.append(dict(file=name,line=text.count('\n',0,match.start())+1,reason=kind))
        if path.suffix=='.md' and name in proposed:
            for target in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)\)',text):
                if target.startswith(('https://','http://','#','mailto:')):continue
                target=target.split('#')[0]
                if target and not (path.parent/target).exists():findings.append(dict(file=name,reason='broken local link',target=target))
        if re.search(r'C:[/\\]Users[/\\]U938|(?:213\.173\.110\.227)',text):warnings.append(dict(file=name,reason='historical workstation/endpoint metadata; review before broader redistribution'))
    print(json.dumps(dict(checked_text_files=checked,findings=findings,warnings=warnings,secret_values_printed=False),indent=2))
    return 1 if findings else 0


if __name__=='__main__':raise SystemExit(main())
