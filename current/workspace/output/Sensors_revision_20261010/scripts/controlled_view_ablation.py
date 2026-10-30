"""Matched ResNet/DINO image views, nested N, cached embeddings and metric controls.
Default is a bounded diagnostic (N=1,5,10; seeds 42,43,44), not a replacement
for independent acquisition. --max-views 50 --seeds 42 43 44 45 46 extends it.
"""
from pathlib import Path
import argparse,csv,json,hashlib,time
import numpy as np
from PIL import Image
import torch
from torchvision import transforms,models
from recompute_diagnostics import ROOT,OUT,DATA,CLASSES,DEST,norm,write_csv,evaluate
parser=argparse.ArgumentParser();parser.add_argument('--max-views',type=int,default=10);parser.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);args=parser.parse_args()
cache=DEST/'controlled_views_hue0';cache.mkdir(exist_ok=True)
aug=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.4,1.)),transforms.RandomHorizontalFlip(.5),transforms.RandomVerticalFlip(.5),transforms.ColorJitter(.2,.2,.2,0.)])
start=time.time();view_manifest=[]
for seed in args.seeds:
    for c in CLASSES:
        p=next((DATA/'dataset_train'/c).glob('*'));im=Image.open(p).convert('RGB')
        for v in range(args.max_views):
            sd=int.from_bytes(hashlib.sha256(f'{seed}|{c}|{v}'.encode()).digest()[:4],'little')
            target=cache/f'{seed}_{c}_{v:02d}.png'
            if not target.exists():
                torch.manual_seed(sd);aug(im).save(target)
            view_manifest.append(dict(seed=seed,class_id=c,view=v,transform_seed=sd,path=str(target.relative_to(OUT)),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    print('cached shared views',seed,flush=True)
write_csv(DEST/'controlled_view_manifest.csv',view_manifest)
with (OUT/'audit/image_manifest.csv').open(encoding='utf-8-sig') as f:manifest=list(csv.DictReader(f))
qinfo={s:sorted([r for r in manifest if r['split']==s],key=lambda r:r['class_id']) for s in ['dataset_val','my_dataset_test_filtered']}
rows=[];ps={}
for name in ['resnet50','dinov2_s14']:
    if name=='resnet50':
        m=models.resnet50(weights=None);m.load_state_dict(torch.load(Path.home()/'.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth',map_location='cpu'));m.fc=torch.nn.Identity()
    else:m=torch.hub.load(str(Path.home()/'.cache/torch/hub/facebookresearch_dinov2_main'),'dinov2_vits14',source='local',pretrained=True)
    m.eval();qcache=np.load(DEST/f'{name}_rgb_features.npz')
    for seed in args.seeds:
        fp=DEST/f'controlled_{name}_seed{seed}_N{args.max_views}_embeddings.npz'
        if fp.exists():features=np.load(fp)['features']
        else:
            ims=[cache/f'{seed}_{c}_{v:02d}.png' for c in CLASSES for v in range(args.max_views)];ff=[]
            with torch.inference_mode():
                for pos in range(0,len(ims),16):
                    batch=torch.stack([norm(Image.open(p).convert('RGB')) for p in ims[pos:pos+16]])
                    ff.append(torch.nn.functional.normalize(m(batch),dim=1).numpy())
            features=np.concatenate(ff).reshape(len(CLASSES),args.max_views,-1);np.savez_compressed(fp,features=features)
        for n in [x for x in [1,5,10,20,30,50] if x<=args.max_views]:
            proto=features[:,:n].mean(1);proto/=np.maximum(np.linalg.norm(proto,axis=1,keepdims=True),1e-12);ps[(name,seed,n)]=proto
            for split,items in qinfo.items():
                q=qcache[split];labels=np.array([CLASSES.index(r['class_id']) for r in items]);score=q@proto.T
                cosine=np.argsort(-score,axis=1,kind='stable');dist=(q*q).sum(1)[:,None]+(proto*proto).sum(1)[None,:]-2*score
                # Ranking identity is exact mathematically; float ties may reorder. Verify score equality.
                assert np.allclose(dist,2-2*score,atol=2e-6)
                metrics=['cosine','euclidean']
                for metric in metrics:
                    order=cosine[:,:3] if metric=='cosine' else np.argsort(dist,axis=1,kind='stable')[:,:3]
                    k1=int((order[:,0]==labels).sum());k3=int((order==labels[:,None]).any(1).sum())
                    rows.append(dict(model=name,seed=seed,n_views=n,metric=metric,split=split,n_queries=len(labels),top1=k1/len(labels)*100,top3=k3/len(labels)*100,top1_correct=k1,top3_correct=k3,embedding_sha256=hashlib.sha256(fp.read_bytes()).hexdigest()))
                pred=[dict(class_id=CLASSES[y],top1=CLASSES[o[0]],top2=CLASSES[o[1]],top3=CLASSES[o[2]],score1=float(score[i,o[0]])) for i,(y,o) in enumerate(zip(labels,cosine))]
                write_csv(DEST/f'controlled_{name}_{seed}_N{n}_{split}_predictions.csv',pred)
        write_csv(DEST/'controlled_ablation_results.csv',rows);print('completed',name,seed,flush=True)
stability=[]
for name in ['resnet50','dinov2_s14']:
    for n in [x for x in [1,5,10,20,30,50] if x<=args.max_views]:
        for i,s1 in enumerate(args.seeds):
            for s2 in args.seeds[i+1:]:
                ang=np.degrees(np.arccos(np.clip((ps[(name,s1,n)]*ps[(name,s2,n)]).sum(1),-1,1)))
                stability.append(dict(model=name,n_views=n,seed1=s1,seed2=s2,mean_angle_deg=float(ang.mean()),median_angle_deg=float(np.median(ang))))
write_csv(DEST/'prototype_angular_variation.csv',stability)
(DEST/'controlled_protocol.json').write_text(json.dumps(dict(seeds=args.seeds,max_views=args.max_views,view_counts=[x for x in [1,5,10,20,30,50] if x<=args.max_views],shared_views=True,nested_subsets=True,query_preprocessing='same RGB cache as color controls',elapsed_seconds=time.time()-start),indent=2))
print('COMPLETE controlled',time.time()-start,flush=True)
