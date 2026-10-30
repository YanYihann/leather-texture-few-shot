import requests,json,zipfile
from pathlib import Path
root=Path('output/Sensors_revision_20261011');out=root/'external/matsim';out.mkdir(exist_ok=True)
code='kZIiSQZCYU5M4HOvnQykql9jxF4h0KiC5MX'
for fid,name in [(19334468029,'benchmark_structure.pdf'),(18721235096,'benchmark_set2.zip'),(19176773763,'evaluation_scripts.zip')]:
 r=requests.get('https://eapi.pcloud.com/getpublinkdownload',params={'code':code,'fileid':fid},timeout=30);j=r.json();assert j['result']==0,j
 u='https://'+j['hosts'][0]+j['path'];s=requests.get(u,timeout=60);s.raise_for_status();(out/name).write_bytes(s.content);print(name,len(s.content),flush=True)
 if name.endswith('.zip'):
  with zipfile.ZipFile(out/name) as z:z.extractall(out/name[:-4])
print('COMPLETE MatSim download',flush=True)
