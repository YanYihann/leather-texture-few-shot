from pathlib import Path
import requests,json
out=Path('output/Sensors_revision_20261011/external')
u='https://thor.robots.ox.ac.uk/dtd/dtd-r1.0.1.tar.gz'
r=requests.get(u,stream=True,timeout=60);r.raise_for_status()
p=out/'dtd-r1.0.1.tar.gz'
with p.open('wb') as f:
 for c in r.iter_content(1024*1024):f.write(c)
print('downloaded',p.stat().st_size,flush=True)
import tarfile
with tarfile.open(p) as t:t.extractall(out,filter='data')
print('extracted',flush=True)
