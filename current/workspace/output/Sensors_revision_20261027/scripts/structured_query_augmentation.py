"""Fixed synthetic query sensitivity; original-query clusters remain the analysis unit.

Uses the previously verified frozen encoders and reference pools without retuning.
Run from any cwd. No source image, reference, or previous deliverable is modified.
"""
from pathlib import Path
import sys, json, csv, hashlib, io, time, datetime
import numpy as np
from PIL import Image

O = Path(__file__).resolve().parents[1]
ROOT = O.parents[1]
PREV = ROOT/'output/Sensors_revision_20261011'
OLD = ROOT/'output/Sensors_revision_20261010/new_experiments'
sys.path.insert(0, str(PREV/'scripts'))
from additional_experiments import load, encode
from recompute_diagnostics import DATA, CLASSES, resize

D = O/'new_experiments'; D.mkdir(parents=True, exist_ok=True)
FAMILIES = ['regional_crop', 'perspective', 'resolution', 'illumination', 'device_pipeline', 'compound']
LEVELS = ['mild', 'moderate']
REPEATS = 3

def sha(x): return hashlib.sha256(x).hexdigest()
def write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def transform(im, family, level, rng):
    """All operations act on the available 224px field, never new surface content."""
    mild = level == 'mild'; p = {}
    if family == 'regional_crop':
        fraction=float(rng.uniform(.8,.95) if mild else rng.uniform(.55,.75))
        n=im.width; k=round(n*fraction)
        x,y=[int(v) for v in rng.integers(0,n-k+1,size=2)]
        im=im.crop((x,y,x+k,y+k)).resize((n,n),Image.Resampling.BICUBIC)
        p=dict(linear_fraction=fraction,x=x,y=y)
    elif family == 'perspective':
        rho=.15 if mild else .28; n=im.width-1
        j=rng.uniform(0,rho,size=8)
        quad=(j[0]*n,j[1]*n,j[2]*n,(1-j[3])*n,(1-j[4])*n,(1-j[5])*n,(1-j[6])*n,j[7]*n)
        im=im.transform(im.size,Image.Transform.QUAD,quad,resample=Image.Resampling.BILINEAR)
        p=dict(inset_bound=rho,input_quad=quad)
    elif family == 'resolution':
        k=int(rng.integers(112,169) if mild else rng.integers(56,97))
        im=im.resize((k,k),Image.Resampling.BILINEAR).resize(im.size,Image.Resampling.BICUBIC)
        p=dict(intermediate_side=k)
    elif family == 'illumination':
        a=np.asarray(im,dtype=np.float32)/255
        y,x=np.mgrid[-1:1:complex(im.height),-1:1:complex(im.width)]
        theta=float(rng.uniform(0,2*np.pi)); amp=float(rng.uniform(.2,.35) if mild else rng.uniform(.4,.6))
        exposure=float(rng.uniform(.85,1.15) if mild else rng.uniform(.65,1.35))
        center=rng.uniform(-.7,.7,size=2); sigma=float(rng.uniform(.18,.35))
        gloss=float(rng.uniform(.08,.18) if mild else rng.uniform(.2,.35))
        gradient=exposure*(1+amp*(x*np.cos(theta)+y*np.sin(theta))/np.sqrt(2))
        highlight=gloss*np.exp(-((x-center[0])**2+(y-center[1])**2)/(2*sigma**2))
        a=np.clip(a*gradient[:,:,None],0,1); a=a*(1-highlight[:,:,None])+highlight[:,:,None]
        im=Image.fromarray(np.uint8(np.clip(a,0,1)*255))
        p=dict(theta=theta,gradient_amplitude=amp,exposure=exposure,highlight_center=center.tolist(),sigma=sigma,highlight_opacity=gloss)
    elif family == 'device_pipeline':
        gains=rng.uniform(.9,1.1,size=3) if mild else rng.uniform(.75,1.25,size=3)
        gamma=float(rng.uniform(.9,1.1) if mild else rng.uniform(.75,1.3))
        sigma=float(rng.uniform(1,3) if mild else rng.uniform(3,7))
        quality=int(rng.integers(70,91) if mild else rng.integers(40,66))
        a=np.clip(np.asarray(im,dtype=np.float32)/255*gains,0,1)**gamma
        a=np.clip(a*255+rng.normal(0,sigma,a.shape),0,255)
        buf=io.BytesIO();Image.fromarray(a.astype(np.uint8)).save(buf,format='JPEG',quality=quality)
        buf.seek(0)
        with Image.open(buf) as result: im=result.convert('RGB')
        p=dict(channel_gains=gains.tolist(),gamma=gamma,noise_sigma_255=sigma,jpeg_quality=quality)
    elif family == 'compound':
        for f in FAMILIES[:-1]:
            im,q=transform(im,f,level,rng); p[f]=q
    else: raise ValueError(family)
    return im,p

def inventory():
    paths=[]
    for c in CLASSES:
        files=sorted(p for p in (DATA/'my_dataset_test_filtered'/c).glob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp'])
        assert len(files)<=1
        paths.extend(files)
    assert len(paths)==201 and len(CLASSES)==203
    return paths,np.array([CLASSES.index(p.parent.name) for p in paths])

def generate(paths,family,level):
    images=[];rows=[]
    for q,path in enumerate(paths):
        with Image.open(path) as src: base=resize(src.convert('RGB'))
        for repeat in range(REPEATS):
            key=f'20261027-v1|{path.parent.name}|{family}|{level}|{repeat}'
            seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
            im,params=transform(base.copy(),family,level,np.random.default_rng(seed))
            images.append(im)
            rows.append(dict(query_index=q,class_id=path.parent.name,query_path=str(path.relative_to(ROOT)).replace('\\','/'),source_sha256=sha(path.read_bytes()),family=family,severity=level,repeat=repeat,seed=seed,pixel_sha256=sha(im.tobytes()),parameters_json=json.dumps(params,sort_keys=True,separators=(',',':'))))
    return images,rows

def protocol(paths):
    record=dict(version='20261027-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='locked before new feature extraction; follow-up sensitivity, not preregistration',
        reference_images_per_class=1,reference_classes=203,original_query_images=201,
        families=FAMILIES,severities=LEVELS,synthetic_variants_per_condition=REPEATS,
        total_synthetic_images=201*len(FAMILIES)*len(LEVELS)*REPEATS,
        independent_query_count_unchanged=201,unit='one original query identity; all descendants kept together',
        input='same original Q images; deterministic Resize256/CenterCrop224 before synthetic operations',
        inference='one transformed image per retrieval; no query-view averaging, tuning, training, reference addition or threshold fitting',
        reference_comparison='frozen single-reference and existing A30 pools seeds42,43,44; A30 is 30 views of the same single reference',
        bootstrap='2000 resamples of 201 original-query clusters, seed20261027; descriptive conditional intervals; no p-value fishing',
        unsupported=['new physical specimens','new capture sessions','real devices','unseen surface regions','batch equivalence','calibrated physical lighting or camera angles'],
        script_sha256=sha(Path(__file__).read_bytes()),queries=[dict(class_id=p.parent.name,sha256=sha(p.read_bytes())) for p in paths])
    fp=D/'structured_protocol.json'
    if not fp.exists():fp.write_text(json.dumps(record,indent=2),encoding='utf8')
    else:
        old=json.loads(fp.read_text());assert old['script_sha256']==record['script_sha256']

def run():
    paths,y=inventory();protocol(paths)
    for model in ['resnet50','dinov2_s14']:
        missing=[(f,l) for f in FAMILIES for l in LEVELS if not (D/f'{model}_{f}_{l}.npz').exists()]
        if not missing:continue
        m=load(model)
        for family,level in missing:
            start=time.time();images,rows=generate(paths,family,level)
            manifest=D/f'inputs_{family}_{level}.csv'
            if manifest.exists():
                with manifest.open(encoding='utf-8-sig') as f: previous=list(csv.DictReader(f))
                assert [r['pixel_sha256'] for r in previous]==[r['pixel_sha256'] for r in rows]
            else: write(manifest,rows)
            features=encode(m,images);assert features.shape[0]==603
            np.savez_compressed(D/f'{model}_{family}_{level}.npz',features=features,labels=np.repeat(y,REPEATS),input_hashes=np.array([r['pixel_sha256'] for r in rows]))
            print(f'{model} {family} {level}: 603 encodings, {time.time()-start:.1f}s',flush=True)
        del m
    summarize()

def summarize():
    paths,y=inventory();rows=[];predictions=[];per_query=[];paired=[]
    boot=np.random.default_rng(20261027).integers(0,201,(2000,201))
    for model in ['resnet50','dinov2_s14']:
        base=np.load(OLD/f'{model}_rgb_features.npz');pool=[]
        for seed in [42,43,44]:
            f=np.load(PREV/'new_experiments'/f'controlled_{model}_seed{seed}_N50_embeddings.npz')['features'][:,:30]
            p=f.mean(1);p/=np.linalg.norm(p,axis=1,keepdims=True);pool.append(p)
        all_results={}
        for variant,refs in [('frozen',[base['dataset_train']]),('A30',pool)]:
            clean=np.mean([(base['my_dataset_test_filtered']@p.T).argmax(1)==y for p in refs],axis=0)
            cells={}
            for family in FAMILIES:
                for level in LEVELS:
                    d=np.load(D/f'{model}_{family}_{level}.npz');f=d['features']
                    assert np.array_equal(d['labels'],np.repeat(y,REPEATS))
                    assert np.max(np.abs(np.linalg.norm(f,axis=1)-1))<1e-5
                    correctness=[]
                    for s,p in enumerate(refs):
                        score=f@p.T;order=np.argsort(-score,axis=1,kind='stable')[:,:3]
                        hit=order[:,0]==np.repeat(y,REPEATS);correctness.append(hit.reshape(201,REPEATS))
                        for i,o in enumerate(order):
                            predictions.append(dict(model=model,variant=variant,reference_seed='fixed' if variant=='frozen' else [42,43,44][s],family=family,severity=level,class_id=paths[i//REPEATS].parent.name,repeat=i%REPEATS,top1_class=CLASSES[o[0]],top1_correct=int(hit[i]),top3_correct=int(y[i//REPEATS] in o)))
                    a=np.mean(correctness,axis=0);cluster=a.mean(1);cells[family,level]=cluster
                    lo,hi=np.quantile(cluster[boot].mean(1),[.025,.975])
                    rows.append(dict(model=model,variant=variant,family=family,severity=level,original_queries=201,synthetic_images=603,reference_draws=len(refs),clean_top1=100*clean.mean(),top1=100*cluster.mean(),ci_low=100*lo,ci_high=100*hi,drop_pp=100*(clean-cluster).mean(),all3_correct=100*np.mean(np.mean([z.all(1) for z in correctness],axis=0))))
                    for i,c in enumerate(cluster):per_query.append(dict(model=model,variant=variant,family=family,severity=level,class_id=paths[i].parent.name,correct_fraction=float(c),clean_correct_fraction=float(clean[i])))
            for level in LEVELS:
                cluster=np.mean([cells[f,level] for f in FAMILIES],axis=0);lo,hi=np.quantile(cluster[boot].mean(1),[.025,.975])
                rows.append(dict(model=model,variant=variant,family='mean_6',severity=level,original_queries=201,synthetic_images=3618,reference_draws=len(refs),clean_top1=100*clean.mean(),top1=100*cluster.mean(),ci_low=100*lo,ci_high=100*hi,drop_pp=100*(clean-cluster).mean(),all3_correct='not_pooled'))
                all_results[variant,level]=cluster
        for level in LEVELS:
            delta=all_results['A30',level]-all_results['frozen',level]
            lo,hi=np.quantile(delta[boot].mean(1),[.025,.975]);paired.append(dict(model=model,severity=level,comparison='A30_minus_frozen_mean6',difference_pp=100*delta.mean(),ci_low=100*lo,ci_high=100*hi))
    write(D/'structured_results.csv',rows);write(D/'structured_predictions.csv',predictions);write(D/'structured_per_query.csv',per_query);write(D/'structured_paired.csv',paired)
    print('COMPLETE structured query sensitivity',len(predictions),'predictions',flush=True)

if __name__=='__main__':
    run()
