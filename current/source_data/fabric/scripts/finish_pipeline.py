"""Finish evaluation and independent artifact checks once active compute jobs finish."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parents[1]


def main():
    begun=time.time()
    required=[OUT/'cache'/f'{name}.done.json' for name in ['resnet50','dinov2','clip']]
    required.append(OUT/'results/adaptation.done.json')
    while not all(path.exists() for path in required):
        if time.time()-begun>21600:
            raise TimeoutError('Six-hour completion wait expired; existing artifacts retained')
        pending=[str(path.relative_to(OUT)) for path in required if not path.exists()]
        print('pending:',', '.join(pending),flush=True)
        time.sleep(30)
    for script in ['evaluate_results.py','verify_results.py']:
        print('running',script,flush=True)
        result=subprocess.run([sys.executable,'-u',str(OUT/'scripts'/script)],cwd=str(OUT),timeout=1800)
        if result.returncode:
            raise RuntimeError(f'{script} failed with exit {result.returncode}; inspect logs, no automatic retry')
    artifacts=[]
    for directory in ['scripts','protocol','results','predictions','audit']:
        for path in sorted((OUT/directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.name!='artifact_manifest.json':
                hasher=hashlib.sha256()
                with path.open('rb') as handle:
                    for chunk in iter(lambda:handle.read(1048576),b''):hasher.update(chunk)
                artifacts.append(dict(path=str(path.relative_to(OUT)).replace('\\','/'),bytes=path.stat().st_size,sha256=hasher.hexdigest()))
    (OUT/'audit/artifact_manifest.json').write_text(json.dumps(dict(status='completed',files=artifacts),indent=2),encoding='utf-8')
    print('COMPLETE EVALUATION AND INDEPENDENT ARTIFACT CHECKS',flush=True)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    main()
