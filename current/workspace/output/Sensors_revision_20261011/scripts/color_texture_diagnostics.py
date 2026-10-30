"""Reference-defined visual cue challenges; no unverified semantic texture labels."""
from pathlib import Path
import csv,json
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1];OLD=ROOT/'output/Sensors_revision_20261010';DATA=ROOT/'IC_project/01_核心代码与数据集'
classes=sorted(p.name for p in (DATA/'dataset_train').iterdir() if p.is_dir())
paths=[next((DATA/'dataset_train'/c).glob('*')) for c in classes]
color=[];texture=[]
for p in paths:
    im=Image.open(p).convert('RGB').resize((224,224));a=np.array(im);h=np.histogramdd(a.reshape(-1,3),bins=(8,)*3,range=((0,256),)*3)[0].ravel();color.append(np.sqrt(h/h.sum()))
    g=np.array(im.convert('L'));c=g[1:-1,1:-1];lbp=np.zeros(c.shape,np.uint8)
    for k,(dy,dx) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):lbp|=((g[1+dy:223+dy,1+dx:223+dx]>=c).astype(np.uint8)<<k)
    h=np.bincount(lbp.ravel(),minlength=256);texture.append(np.sqrt(h/h.sum()))
color=np.stack(color);texture=np.stack(texture);cs=color@color.T;ts=texture@texture.T
# Percentile-based reference-only operational strata. These are visual descriptor proxies.
ii,jj=np.triu_indices(203,1);chi=np.quantile(cs[ii,jj],.90);tlo=np.quantile(ts[ii,jj],.25);thi=np.quantile(ts[ii,jj],.90);clo=np.quantile(cs[ii,jj],.25)
groups={'similar_color_different_lbp':[(i,j) for i,j in zip(ii,jj) if cs[i,j]>=chi and ts[i,j]<=tlo], 'similar_lbp_different_color':[(i,j) for i,j in zip(ii,jj) if ts[i,j]>=thi and cs[i,j]<=clo]}
manifest=[]
for group,pairs in groups.items():
    for i,j in pairs:manifest.append(dict(group=group,class_a=classes[i],class_b=classes[j],color_similarity=float(cs[i,j]),lbp_similarity=float(ts[i,j])))
with (OUT/'new_experiments/reference_challenge_pairs.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['group','class_a','class_b','color_similarity','lbp_similarity']);w.writeheader();w.writerows(manifest)
(OUT/'new_experiments/reference_challenge_protocol.json').write_text(json.dumps(dict(color_high=float(chi),color_low=float(clo),lbp_high=float(thi),lbp_low=float(tlo),counts={k:len(v) for k,v in groups.items()},caution='Descriptor-defined challenges, not human-certified same-grain groups. Thresholds computed from all reference pairs; no query labels or performance used.'),indent=2))
for group,pairs in groups.items():
    pairs=sorted(pairs,key=lambda ij: cs[ij] if group.startswith('similar_color') else ts[ij],reverse=True)[:12]
    canvas=Image.new('RGB',(1000,190*len(pairs)), 'white');draw=ImageDraw.Draw(canvas)
    for n,(i,j) in enumerate(pairs):
        for k,t in enumerate([i,j]):
            im=Image.open(paths[t]).convert('RGB');im.thumbnail((400,150));canvas.paste(im,(k*450,n*190));draw.text((k*450,n*190+153),classes[t],fill='black')
        draw.text((0,n*190+172),f'color={cs[i,j]:.3f}, LBP={ts[i,j]:.3f}',fill='black')
    canvas.save(OUT/f'audit/{group}.jpg')
print({k:len(v) for k,v in groups.items()})
