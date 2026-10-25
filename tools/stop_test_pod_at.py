"""Stop one explicitly selected temporary test pod at an absolute UTC deadline."""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pod-id', required=True)
    parser.add_argument('--deadline', required=True)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline).astimezone(timezone.utc)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    def record(status, **fields):
        with args.log.open('a') as stream:
            stream.write(json.dumps({'time_utc': datetime.now(timezone.utc).isoformat(),
                'pod_id': args.pod_id, 'status': status, **fields}) + '\n')

    record('scheduled', deadline=deadline.isoformat())
    while (remaining := (deadline - datetime.now(timezone.utc)).total_seconds()) > 0:
        time.sleep(min(30, remaining))
    for attempt in range(3):
        result = subprocess.run(['runpodctl', 'pod', 'stop', args.pod_id], capture_output=True, text=True, timeout=45)
        if result.returncode == 0:
            check = subprocess.run(['runpodctl', 'pod', 'get', args.pod_id], capture_output=True, text=True, timeout=45)
            try:
                data = json.loads(check.stdout)
                state = data.get('status') or data.get('runtimeStatus') or data.get('desiredStatus')
            except ValueError:
                state = 'readback_unavailable'
            record('stop_requested', readback_state=state)
            return
        record('stop_attempt_failed', attempt=attempt + 1, exit_code=result.returncode)
        time.sleep(15)
    record('stop_failed_requires_operator')
    raise SystemExit(1)


if __name__ == '__main__':
    main()
