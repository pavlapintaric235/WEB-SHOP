"""Copy existing assets over approved SSH and start a private test worker.

Does not create/start pods, install SSH, alter keys, or expose HTTP ports.
"""
import argparse
import json
import os
import secrets
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--known-hosts', type=Path, help='Host key verified through the authenticated RunPod gateway')
    args = parser.parse_args()
    source = ROOT / 'experiments/first_two_train_third_test_20261020/dino/.build_dino_eval/inventorylens-dino-eval'
    assets = [source / 'model_epoch_011.pth', source / 'experiment.yaml', ROOT / 'tools/tao_dino_worker.py']
    for asset in assets:
        if not asset.is_file():
            raise FileNotFoundError(asset)
    folder = ROOT / 'data/webshop_demo'
    folder.mkdir(parents=True, exist_ok=True)
    config_file = folder / 'worker_connection.json'
    # Reconnect the same worker without rotating its already configured secret.
    config = json.loads(config_file.read_text()) if config_file.exists() else {
        'url': 'http://127.0.0.1:8765', 'token': secrets.token_urlsafe(32)}
    config.update(host=args.host, port=args.port)
    if len(config.get('token', '')) < 24:
        raise ValueError('Existing worker token is invalid; it was left unchanged.')
    host_options = ['-o', 'StrictHostKeyChecking=yes']
    if args.known_hosts:
        host_options += ['-o', f'UserKnownHostsFile={args.known_hosts.resolve()}']
    ssh = ['ssh', *host_options, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-p', str(args.port), f'root@{args.host}']
    subprocess.run(ssh + ['mkdir -p /workspace/webshop-worker'], check=True)
    config_file.write_text(json.dumps(config))
    subprocess.run(['scp', *host_options, '-q', '-P', str(args.port), *map(str, assets), str(config_file),
                    f'root@{args.host}:/workspace/webshop-worker/'], check=True)
    remote = """import json,os,pathlib,subprocess,nvidia_tao_pytorch
root=pathlib.Path('/workspace/webshop-worker')
(pathlib.Path(nvidia_tao_pytorch.__file__).parent/'cv/dino/experiment_specs').mkdir(parents=True,exist_ok=True)
secret=root/'worker_connection.json'
secret.chmod(0o600)
env=dict(os.environ,WEBSHOP_DINO_TOKEN=json.loads(secret.read_text())['token'])
with (root/'worker.log').open('ab') as log:
 p=subprocess.Popen(['python3',str(root/'tao_dino_worker.py'),'--spec',str(root/'experiment.yaml'),'--checkpoint',str(root/'model_epoch_011.pth')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'worker.pid').write_text(str(p.pid))
print(json.dumps({'worker_pid':p.pid,'bound_to':'127.0.0.1:8765'}))
"""
    subprocess.run(ssh + ['python3 -c ' + shlex.quote(remote)], check=True)
    print('Private worker installed. Connection secret saved locally; not printed.')


if __name__ == '__main__':
    main()
