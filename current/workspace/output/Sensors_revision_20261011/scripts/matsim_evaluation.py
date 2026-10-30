"""MatSim Set2: fixed external masked-crop protocol, macro accuracy, no tuning."""
from pathlib import Path
import json,hashlib
import numpy as np,torch
from PIL import Image
from torchvision import transforms
from additional_experiments import load,encode
from recompute_diagnostics import OUT,DEST,resize,write_csv
torch.set_num_threads(2)
root=OUT/'external/matsim/benchmark_set2/Set2_MatSim_BenchMark_Uncorrelated_materials_objects'
allfiles=sorted(root.glob('*/*.jpg'));files=[p for p in allfiles if p.with_name(p.stem+'_MASK.png').exists()]
(DEST/'matsim_exclusions.json').write_text(json.dumps([dict(path=str(p.relative_to(root)),reason='corresponding foreground mask absent in the distributed archive; no substitute mask inferred') for p in allfiles if p not in files],indent=2))
classes=sorted({p.parent.name for p in files});y=np.array([classes.index(p.parent.name) for p in files]);images=[];manifest=[]
for p in files:
    im=Image.open(p).convert('RGB');maskpath=p.with_name(p.stem+'_MASK.png');mask=np.array(Image.open(maskpath).convert('L'))>127
    assert mask.shape==np.array(im).shape[:2] and mask.any()
    a=np.array(im);a[~mask]=127;yy,xx=np.where(mask);im=Image.fromarray(a).crop((xx.min(),yy.min(),xx.max()+1,yy.max()+1));images.append(im)
    manifest.append(dict(path=str(p.relative_to(root)),class_id=p.parent.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),mask_sha256=hashlib.sha256(maskpath.read_bytes()).hexdigest()))
write_csv(DEST/'matsim_manifest.csv',manifest)
plans=[]
for seed in range(10):
    rng=np.random.default_rng(20261011+seed);refs=np.array([rng.choice(np.where(y==c)[0]) for c in range(len(classes))]);query=np.array([i for i in range(len(files)) if i not in refs]);plans.append((seed,refs,query))
(DEST/'matsim_protocol.json').write_text(json.dumps(dict(source='https://github.com/ZuseZ4/MatSim-Dataset-Generator-Scripts-And-Neural-net',subset='Set2, uncorrelated materials/objects',n_images=len(files),n_classes=len(classes),preprocessing='provided foreground mask >127, background127, tight foreground bounding box, Resize256/CenterCrop224',evaluation='16-way one-reference, all remaining images queried; macro class accuracy; 10 deterministic reference draws; same masks and references for all methods; no selection/tuning on this dataset',additional='all-to-all retrieval with diagonal excluded, both macro/micro; differs from one-reference protocol',reference_plans=[dict(seed=k,reference_indices=r.tolist(),query_indices=q.tolist()) for k,r,q in plans],seeds=list(range(20261011,20261021))),indent=2))
rows=[];pred=[];aug=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.4,1.)),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ColorJitter(.2,.2,.2,0)])
for name in ['rgb_histogram','resnet50','dinov2_s14']:
    if name=='rgb_histogram':
        ff=[]
        for im in images:
            h=np.histogramdd(np.array(resize(im)).reshape(-1,3),bins=(8,)*3,range=((0,256),)*3)[0].ravel();ff.append(np.sqrt(h/h.sum()))
        f=np.stack(ff);prototypes=None
    else:
        model=load(name);fp=DEST/f'matsim_{name}_features.npz'
        if fp.exists():d=np.load(fp);f=d['features'];prototypes=d['augmented']
        else:
            f=encode(model,[resize(im) for im in images]);augims=[]
            for i,im in enumerate(images):
                for v in range(5):
                    seed=int.from_bytes(hashlib.sha256(f'matsim|{manifest[i]["path"]}|{v}'.encode()).digest()[:4],'little');torch.manual_seed(seed);augims.append(aug(im))
            prototypes=encode(model,augims).reshape(len(files),5,-1).mean(1);prototypes/=np.linalg.norm(prototypes,axis=1,keepdims=True);np.savez_compressed(fp,features=f,augmented=prototypes,labels=y)
    score=f@f.T;np.fill_diagonal(score,-np.inf);correct=y[score.argmax(1)]==y
    rows.append(dict(method=name,protocol='all_to_all',seed=-1,n_queries=len(y),macro_top1=100*np.mean([correct[y==c].mean() for c in range(len(classes))]),micro_top1=100*correct.mean()))
    for variant,proto in [(name,f)]+([(name+'_A5',prototypes)] if prototypes is not None else []):
        for seed,refs,query in plans:
            score=f[query]@proto[refs].T;order=np.argsort(-score,axis=1);correct=order[:,0]==y[query]
            rows.append(dict(method=variant,protocol='one_reference',seed=seed,n_queries=len(query),macro_top1=100*np.mean([correct[y[query]==c].mean() for c in range(len(classes))]),micro_top1=100*correct.mean()))
            pred.extend(dict(method=variant,seed=seed,path=manifest[i]['path'],true_class=classes[y[i]],predicted=classes[order[j,0]],score=float(score[j,order[j,0]])) for j,i in enumerate(query))
    write_csv(DEST/'matsim_results.csv',rows);write_csv(DEST/'matsim_predictions.csv',pred);print('MatSim done',name,flush=True)
print('COMPLETE MatSim',flush=True)
