"""Download verified official Windows tooling; never starts a public tunnel."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[1]


def main():
    request=Request('https://api.github.com/repos/cloudflare/cloudflared/releases/latest',headers={'User-Agent':'WEB-SHOP-release-check'})
    release=json.load(urlopen(request,timeout=30))
    asset=next(a for a in release['assets'] if a['name']=='cloudflared-windows-amd64.exe')
    digest=asset.get('digest','')
    if not digest.startswith('sha256:'):raise RuntimeError('Official release has no verifiable SHA-256 digest.')
    if not asset['browser_download_url'].startswith('https://github.com/cloudflare/cloudflared/releases/'):raise RuntimeError('Unexpected binary source.')
    folder=ROOT/'data/judge_release/bin';folder.mkdir(parents=True,exist_ok=True)
    path=folder/asset['name']
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest.split(':',1)[1]:
        temporary=path.with_suffix('.download')
        try:
            with urlopen(asset['browser_download_url'],timeout=60) as source,temporary.open('wb') as target:
                while chunk:=source.read(1024*1024):target.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest()!=digest.split(':',1)[1]:raise RuntimeError('Download checksum mismatch.')
            temporary.replace(path)
        finally:temporary.unlink(missing_ok=True)
    (folder.parent/'cloudflared_verified.json').write_text(json.dumps(dict(release=release['tag_name'],sha256=digest.split(':',1)[1],source=asset['browser_download_url']),indent=2))
    print('Official cloudflared binary verified. No tunnel opened.')


if __name__=='__main__':main()
