"""Opt-in email-restricted HTTPS sharing. Never exposes the private 8509 app."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def validate_emails(values):
    emails=sorted(set(v.strip().lower() for value in values for v in value.split(',')))
    if not emails or any(not re.fullmatch(r'[a-z0-9.!#$%&\x27+/=?^_`{|}~-]+@[a-z0-9.-]+\.[a-z]{2,}',v) or '*' in v for v in emails):
        raise ValueError('Supply explicit allowed email addresses; broad domains and wildcards are disabled.')
    return emails


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allowed-email',action='append',required=True)
    args=parser.parse_args()
    emails=validate_emails(args.allowed_email)
    os.environ['WEBSHOP_DATA_DIR']=str(ROOT/'data/webshop_judge_demo')
    from judge_release import validate_sandbox
    validate_sandbox()
    folder=ROOT/'data/judge_release'
    metadata=json.loads((folder/'cloudflared_verified.json').read_text())
    binary=folder/'bin/cloudflared-windows-amd64.exe'
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=metadata['sha256']:
        raise RuntimeError('Cloudflared checksum does not match the verified release.')
    help_text=subprocess.run([str(binary),'tunnel','--help'],capture_output=True,text=True,check=True).stdout
    if '--allowed-mail' not in help_text:raise RuntimeError('This client cannot restrict Quick Tunnels by email; no tunnel was opened.')
    if urlopen('http://127.0.0.1:8511/_stcore/health',timeout=10).status!=200:
        raise RuntimeError('Start tools/start_webshop_judge.ps1 first.')
    command=[str(binary),'tunnel','--url','http://127.0.0.1:8511','--loglevel','info']
    for email in emails:command.extend(['--allowed-mail',email])
    print('Starting email-restricted HTTPS for the isolated judge sandbox. Keep this process running.')
    print('The changing hostname, no uptime guarantee and laptop dependency make this a supervised demo, not permanent hosting.')
    subprocess.run(command,check=True)


if __name__=='__main__':main()
