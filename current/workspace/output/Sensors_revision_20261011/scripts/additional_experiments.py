"""Fixed-protocol sensitivity, complete perturbation caches and external DTD transfer."""
from pathlib import Path
import sys,json,time,hashlib,csv
import numpy as np,torch
from PIL import Image
from torchvision import transforms,models
from recompute_diagnostics import ROOT,OUT,DATA,DEST,CLASSES,norm,resize,write_csv
sys.path.insert(0,str(DATA))
from evaluate_augmented_robustness import apply_perturbation,PERTURBATIONS,SEVERITY_LEVELS
torch.set_num_threads(3)
OLD=ROOT/'output/Sensors_revision_20261010/new_experiments'
def load(name):
    if name=='resnet50':
        m=models.resnet50(weights=None);m.load_state_dict(torch.load(Path.home()/'.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth',map_location='cpu'));m.fc=torch.nn.Identity()
    else:m=torch.hub.load(str(Path.home()/'.cache/torch/hub/facebookresearch_dinov2_main'),'dinov2_vits14',source='local',pretrained=True)
    return m.eval()
def encode(m,ims,local=False):
    out=[]
    with torch.inference_mode():
        for i in range(0,len(ims),16):
            x=torch.stack([norm(im) for im in ims[i:i+16]])
            if local:
                f=m.forward_features(x)['x_norm_patchtokens'];b,n,d=f.shape
                f=torch.nn.functional.adaptive_avg_pool2d(f.transpose(1,2).reshape(b,d,16,16),(4,4)).flatten(2).transpose(1,2)
            else:f=m(x)
            out.append(torch.nn.functional.normalize(f,dim=-1).numpy())
    return np.concatenate(out)
def scores_report(score,labels,method,split,rows):
    order=np.argsort(-score,axis=1,kind='stable');a=(order[:,0]==labels);b=(order[:,:3]==np.array(labels)[:,None]).any(1)
    rows.append(dict(method=method,split=split,n=len(labels),top1=100*a.mean(),top3=100*b.mean(),top1_correct=int(a.sum()),top3_correct=int(b.sum())))
    return order
def leather():
    items={s:[next((DATA/s/c).glob('*')) for c in CLASSES if (DATA/s/c).exists()] for s in ['dataset_train','dataset_val','my_dataset_test_filtered']}
    labels={s:np.array([CLASSES.index(p.parent.name) for p in ps]) for s,ps in items.items()};rows=[]
    for name in ['resnet50','dinov2_s14']:
        m=load(name)
        recommended=transforms.Compose([transforms.Resize(232 if name=='resnet50' else 256,interpolation=transforms.InterpolationMode.BILINEAR if name=='resnet50' else transforms.InterpolationMode.BICUBIC),transforms.CenterCrop(224)])
        sens=DEST/f'{name}_recommended_features.npz'
        if not sens.exists():np.savez_compressed(sens,**{s:encode(m,[recommended(Image.open(p).convert('RGB')) for p in ps]) for s,ps in items.items()})
        f=np.load(sens)
        for s in list(items)[1:]:scores_report(f[s]@f['dataset_train'].T,labels[s],name+'_recommended',s,rows)
        if name=='dinov2_s14':
            pf=DEST/'dino_local4x4_features.npz'
            if not pf.exists():np.savez_compressed(pf,**{s:encode(m,[resize(Image.open(p).convert('RGB')) for p in ps],True) for s,ps in items.items()})
            ff=np.load(pf);r=torch.from_numpy(ff['dataset_train'])
            for s in list(items)[1:]:
                scores=[]
                for q in torch.from_numpy(ff[s]):
                    sim=torch.einsum('pd,crd->cpr',q,r);scores.append(((sim.max(1).values.mean(1)+sim.max(2).values.mean(1))/2).numpy())
                score=np.stack(scores);scores_report(score,labels[s],'dinov2_local_symmetric',s,rows);np.save(DEST/f'local_{s}_scores.npy',score)
        write_csv(DEST/'sensitivity_local_results.csv',rows)
        hashes=[]
        hashfile=DEST/'robust_resnet50_input_hashes.csv'
        prior_hashes=[]
        if hashfile.exists():
            with hashfile.open(encoding='utf-8-sig') as handle:prior_hashes=list(csv.DictReader(handle))
        for severity in SEVERITY_LEVELS:
            for perturbation in PERTURBATIONS:
                fp=DEST/f'robust_{name}_{perturbation}_{severity}.npz'
                prior=[dict(r,model=name) for r in prior_hashes if r['severity']==severity and r['perturbation']==perturbation]
                if fp.exists() and len(prior)==201:
                    hashes.extend(prior);continue
                ims=[]
                for p in items['my_dataset_test_filtered']:
                    seed=int.from_bytes(hashlib.sha256(f'20261011|{p.parent.name}|{perturbation}|{severity}'.encode()).digest()[:8],'little')
                    with Image.open(p) as source:
                        im=source.convert('RGB');x=resize(apply_perturbation(im,perturbation,severity,np.random.default_rng(seed)))
                    del im
                    ims.append(x)
                    hashes.append(dict(model=name,class_id=p.parent.name,perturbation=perturbation,severity=severity,seed=seed,pixel_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
                if not fp.exists():np.savez_compressed(fp,features=encode(m,ims),labels=labels['my_dataset_test_filtered'])
                print('robust',name,perturbation,severity,flush=True)
        if hashes:write_csv(DEST/f'robust_{name}_input_hashes.csv',hashes)
    print('COMPLETE leather diagnostics',flush=True)
def external():
    dtd=OUT/'external/dtd';files=sorted((dtd/'images').glob('*/*.jpg'))
    assert len(files)==5640,len(files)
    names=[str(p.relative_to(dtd/'images')).replace('\\','/') for p in files];ix={n:i for i,n in enumerate(names)};cats=sorted({p.parent.name for p in files});y=np.array([cats.index(p.parent.name) for p in files])
    plan=[]
    for split in range(1,11):
        train=(dtd/f'labels/train{split}.txt').read_text().splitlines();test=(dtd/f'labels/test{split}.txt').read_text().splitlines();rng=np.random.default_rng(20261011+split)
        refs=[str(rng.choice(sorted(n for n in train if n.split('/')[0]==c))) for c in cats]
        plan.append(dict(split=split,refs=refs,queries=test))
    (DEST/'dtd_protocol.json').write_text(json.dumps(dict(source='https://www.robots.ox.ac.uk/~vgg/data/dtd/',protocol='47-way one-reference attribute-category classification; not material identity matching; official train reference/test query pools',splits=plan,seed_base=20261011,selection='no DTD tuning; N=5 and RGB hist8 selected in advance; 10 overlapping official splits are descriptive reference/split variation'),indent=2))
    hist=[]
    for p in files:
        a=np.asarray(resize(Image.open(p).convert('RGB')));h=np.histogramdd(a.reshape(-1,3),bins=(8,)*3,range=((0,256),)*3)[0].ravel();hist.append(np.sqrt(h/h.sum()))
    hist=np.stack(hist);rows=[]
    for method in ['rgb_histogram','resnet50','dinov2_s14']:
        if method=='rgb_histogram':f=hist;m=None
        else:
            m=load(method);fp=DEST/f'dtd_{method}_features.npz'
            if fp.exists():f=np.load(fp)['features']
            else:
                ff=[]
                for pos in range(0,len(files),160):
                    ff.append(encode(m,[resize(Image.open(p).convert('RGB')) for p in files[pos:pos+160]]));print('external encode',method,pos,flush=True)
                f=np.concatenate(ff);np.savez_compressed(fp,features=f,names=np.array(names),labels=y)
        for p in plan:
            ri=[ix[n] for n in p['refs']];qi=[ix[n] for n in p['queries']];score=f[qi]@f[ri].T
            scores_report(score,y[qi],method,p['split'],rows)
            if m is not None:
                af=DEST/f'dtd_{method}_split{p["split"]}_A5.npz'
                if af.exists():proto=np.load(af)['prototypes']
                else:
                    ims=[]
                    aug=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.4,1.)),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ColorJitter(.2,.2,.2,0)])
                    for n in p['refs']:
                        for v in range(5):
                            seed=int.from_bytes(hashlib.sha256(f'dtd|{p["split"]}|{n}|{v}'.encode()).digest()[:4],'little');torch.manual_seed(seed);ims.append(aug(Image.open(dtd/'images'/n).convert('RGB')))
                    ef=encode(m,ims).reshape(47,5,-1);proto=ef.mean(1);proto/=np.linalg.norm(proto,axis=1,keepdims=True);np.savez_compressed(af,prototypes=proto)
                scores_report(f[qi]@proto.T,y[qi],method+'_A5',p['split'],rows)
            write_csv(DEST/'dtd_results.csv',rows);print('external scored',method,p['split'],flush=True)
    print('COMPLETE external',flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['leather','external']);a=p.parse_args()
    leather() if a.mode=='leather' else external()
