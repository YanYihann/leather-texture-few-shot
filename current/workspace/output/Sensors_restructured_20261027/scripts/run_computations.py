"""C1-C5 locked follow-up. Prior result directories are read-only inputs."""
from pathlib import Path
import sys,json,csv,hashlib,io,time,os,gc,platform,ctypes
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
O=Path(__file__).resolve().parents[1];ROOT=O.parents[1]
P=ROOT/'output/Sensors_revision_20261011'; Q=ROOT/'output/Sensors_revision_20261027'; OLD=ROOT/'output/Sensors_revision_20261010/new_experiments'
sys.path.insert(0,str(P/'scripts'));sys.path.insert(0,str(Q/'scripts'))
from additional_experiments import load,encode
from recompute_diagnostics import DATA,CLASSES,resize,norm
from structured_query_augmentation import generate,inventory,FAMILIES,LEVELS
from evaluate_augmented_robustness import apply_perturbation,PERTURBATIONS,SEVERITY_LEVELS
_legacy_load=load
def load(name):
 """The integrated package can load its supplied weights without a download."""
 weight_root=os.environ.get('IC_WEIGHT_ROOT')
 if not weight_root:return _legacy_load(name)
 wr=Path(weight_root)
 if name=='resnet50':
  from torchvision import models
  model=models.resnet50(weights=None);model.load_state_dict(torch.load(wr/'resnet50-11ad3fa6.pth',map_location='cpu'));model.fc=torch.nn.Identity()
 else:
  model=torch.hub.load(str(wr/'dinov2_code'),'dinov2_vits14',source='local',pretrained=False);model.load_state_dict(torch.load(wr/'dinov2_vits14_pretrain.pth',map_location='cpu'))
 return model.eval()
torch.set_num_threads(3)
D=O/'results';C=O/'cache';D.mkdir(exist_ok=True);C.mkdir(exist_ok=True)
LOCK=json.loads((O/'audit/locked_protocol.json').read_text(encoding='utf8'))
assert hashlib.sha256((O/'audit/locked_protocol.json').read_bytes()).hexdigest()==(O/'audit/locked_protocol.sha256').read_text().split()[0]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows):
 if not rows:return
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def unit(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
def hist(ims,kind):
 out=[]
 for im in ims:
  if kind=='rgb_histogram':h=np.histogramdd(np.asarray(im).reshape(-1,3),bins=(8,)*3,range=((0,256),)*3)[0].ravel()
  else:h=np.histogram(np.asarray(im.convert('L')),bins=32,range=(0,256))[0]
  out.append(np.sqrt(h/h.sum()))
 return np.array(out,dtype=np.float32)
paths,y=inventory();rp=[next((DATA/'dataset_train'/c).glob('*')) for c in CLASSES]
def imread(p,pre=True):
 with Image.open(p) as im:return resize(im.convert('RGB')) if pre else im.convert('RGB')
def refs(method):
 if method in ['rgb_histogram','gray_histogram']:return hist([imread(p) for p in rp],method)
 return np.load(P/'new_experiments'/f'{method}_rgb_features.npz')['dataset_train']
boot=np.random.default_rng(20261027).integers(0,201,(2000,201))
def ci(a):return np.quantile(np.asarray(a)[boot].mean(1),[.025,.975])*100
def metric(score,labels):
 order=np.argsort(-score,axis=1,kind='stable');rank=(order==labels[:,None]).argmax(1)+1
 return dict(recall1=100*np.mean(rank==1),recall3=100*np.mean(rank<=3),recall5=100*np.mean(rank<=5),mrr=np.mean(1/rank)),order,rank
def newpred(score,labels,meta,method,protocol,condition,variant='frozen',seed='fixed'):
 met,order,rank=metric(score,labels);rr=[]
 for i,(o,t,m) in enumerate(zip(order,labels,meta)):
  wrong=o[o!=t][0];rr.append(dict(method=method,variant=variant,reference_seed=seed,protocol=protocol,condition=condition,class_id=CLASSES[int(t)],repeat=m.get('repeat',0),rank=int(rank[i]),top1=CLASSES[o[0]],top3='|'.join(CLASSES[k] for k in o[:3]),top5='|'.join(CLASSES[k] for k in o[:5]),true_score=float(score[i,t]),wrong_class=CLASSES[wrong],wrong_score=float(score[i,wrong]),true_margin=float(score[i,t]-score[i,wrong])))
 return met,rr
def color():
 allrows=[];pred=[];paired=[];verified=[];reference={m:refs(m) for m in ['rgb_histogram','gray_histogram']}
 for protocol in ['A','B']:
  cells=[(f,l) for f in (PERTURBATIONS if protocol=='A' else FAMILIES) for l in (SEVERITY_LEVELS if protocol=='A' else LEVELS)]
  for family,level in cells:
   if protocol=='B':
    ims,meta=generate(paths,family,level);prev=read(Q/'new_experiments'/f'inputs_{family}_{level}.csv');assert [r['pixel_sha256'] for r in meta]==[r['pixel_sha256'] for r in prev];labels=np.repeat(y,3)
   else:
    ims=[];meta=[];labels=y
    prev={r['class_id']:r for r in read(P/'new_experiments/robust_resnet50_input_hashes.csv') if r['perturbation']==family and r['severity']==level}
    for p in paths:
     seed=int.from_bytes(hashlib.sha256(f'20261011|{p.parent.name}|{family}|{level}'.encode()).digest()[:8],'little')
     im=resize(apply_perturbation(imread(p,False),family,level,np.random.default_rng(seed)));h=hashlib.sha256(im.tobytes()).hexdigest();assert h==prev[p.parent.name]['pixel_sha256'];ims.append(im);meta.append(dict(pixel_sha256=h,repeat=0))
   verified.append(dict(protocol=protocol,condition=family+'_'+level,inputs=len(ims),all_pixel_hashes_equal=True))
   scores={}
   for m in ['rgb_histogram','gray_histogram','resnet50','dinov2_s14']:
    if m in reference:
     ff=hist(ims,m);score=ff@reference[m].T;np.savez_compressed(C/f'{protocol}_{m}_{family}_{level}.npz',features=ff,labels=labels)
    else:
     fn=(P/'new_experiments'/f'robust_{m}_{family}_{level}.npz') if protocol=='A' else Q/'new_experiments'/f'{m}_{family}_{level}.npz'
     a=np.load(fn);assert np.array_equal(a['labels'],labels);score=a['features']@refs(m).T
     if protocol=='B':assert np.array_equal(a['input_hashes'],np.array([r['pixel_sha256'] for r in meta]))
    met,rr=newpred(score,labels,meta,m,protocol,family+'_'+level);pred.extend(rr);correct=(np.argsort(-score,axis=1,kind='stable')[:,0]==labels).reshape(201,-1).mean(1);lo,hi=ci(correct)
    allrows.append(dict(protocol=protocol,method=m,family=family,severity=level,original_queries=201,synthetic_inputs=len(labels),**met,ci_low=lo,ci_high=hi));scores[m]=correct
   for m in ['gray_histogram','resnet50','dinov2_s14']:
    delta=scores[m]-scores['rgb_histogram'];lo,hi=ci(delta);paired.append(dict(protocol=protocol,method=m,comparison='minus_RGB',family=family,severity=level,difference_pp=100*delta.mean(),ci_low=lo,ci_high=hi))
   print('C1 verified/scored',protocol,family,level,flush=True)
 write(D/'c1_results.csv',allrows);write(D/'c1_predictions.csv',pred);write(D/'c1_paired.csv',paired);write(O/'audit/c1_pixel_verification.csv',verified)
 print('COMPLETE C1',flush=True)
def c2():
 rows=[];pr=[];sub=[];stability=[];cur={};pairs=read(P/'new_experiments/reference_challenge_pairs.csv')
 for m in ['rgb_histogram','gray_histogram','resnet50','dinov2_s14']:
  if m.endswith('histogram'):ff=hist([imread(p) for p in paths],m)
  else:ff=np.load(P/'new_experiments'/f'{m}_rgb_features.npz')['my_dataset_test_filtered']
  score=ff@refs(m).T;cur[m]=score;met,rr=newpred(score,y,[{}]*201,m,'clean','clean');rows.append(dict(method=m,variant='frozen',seed='fixed',**met));pr.extend(rr)
  for g in sorted({r['group'] for r in pairs}):
   ids={r[k] for r in pairs if r['group']==g for k in ['class_a','class_b']};ix=[i for i,p in enumerate(paths) if p.parent.name in ids];met,_,_=metric(score[ix],y[ix]);sub.append(dict(group=g,method=m,n_queries=len(ix),library=203,**met))
  variants=[('frozen','fixed',refs(m))]
  if not m.endswith('histogram'):
   for seed in [42,43,44]:
    f=np.load(P/'new_experiments'/f'controlled_{m}_seed{seed}_N50_embeddings.npz')['features'][:,:30];proto=unit(f.mean(1));met,rr=newpred(ff@proto.T,y,[{}]*201,m,'clean','clean','A30',seed);rows.append(dict(method=m,variant='A30',seed=seed,**met));pr.extend(rr);variants.append(('A30',seed,proto))
  for variant,seed,proto in variants:
   cells=[]
   for fam in FAMILIES:
    for lev in LEVELS:
     fp=C/f'B_{m}_{fam}_{lev}.npz' if m.endswith('histogram') else Q/'new_experiments'/f'{m}_{fam}_{lev}.npz'
     f=np.load(fp)['features'];correct=(f@proto.T).argmax(1)==np.repeat(y,3);cells.append(correct.reshape(201,3))
   hits=np.concatenate(cells,axis=1);assert hits.shape==(201,36)
   for i,p in enumerate(paths):stability.append(dict(method=m,variant=variant,seed=seed,class_id=p.parent.name,correct_fraction=float(hits[i].mean()),all36=int(hits[i].all())))
 np.savez_compressed(C/'current_clean_scores.npz',**cur,labels=y,classes=np.array(CLASSES))
 write(D/'clean_metrics.csv',rows);write(D/'clean_predictions.csv',pr);write(D/'full_library_subgroups.csv',sub);write(D/'query_stability.csv',stability)
 # Recompute all A30 B retrieval ranks, retaining full-order metrics.
 out=[]
 for m in ['resnet50','dinov2_s14']:
  for seed in [42,43,44]:
   proto=unit(np.load(P/'new_experiments'/f'controlled_{m}_seed{seed}_N50_embeddings.npz')['features'][:,:30].mean(1))
   for fam in FAMILIES:
    for lev in LEVELS:
     f=np.load(Q/'new_experiments'/f'{m}_{fam}_{lev}.npz')['features'];met,_,_=metric(f@proto.T,np.repeat(y,3));out.append(dict(method=m,seed=seed,family=fam,severity=lev,**met))
 write(D/'b_a30_ranking_metrics.csv',out);print('COMPLETE C2',flush=True)
def dtd():
 root=P/'external/dtd';files=sorted((root/'images').glob('*/*.jpg'));assert len(files)==5640
 names=[p.relative_to(root/'images').as_posix() for p in files];ix={n:i for i,n in enumerate(names)};cats=sorted({p.parent.name for p in files});labels=np.array([cats.index(p.parent.name) for p in files]);plans=json.loads((P/'new_experiments/dtd_protocol.json').read_text())['splits']
 write(O/'audit/dtd_image_manifest.csv',[dict(path=n,sha256=sha(p)) for p,n in zip(files,names)])
 aug=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.4,1.)),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ColorJitter(.2,.2,.2,0)])
 view_manifest=[];results=[];pred=[];validation=[]
 for method in ['rgb_histogram','gray_histogram','resnet50','dinov2_s14']:
  m=None
  if method.endswith('histogram'):
   fp=C/f'dtd_{method}_features.npz'
   if not fp.exists():np.savez_compressed(fp,features=hist([imread(p) for p in files],method),names=np.array(names),labels=labels)
   f=np.load(fp)['features']
  else:
   m=load(method);fp=P/'new_experiments'/f'dtd_{method}_features.npz'
   if not fp.exists():
    fp=C/f'dtd_{method}_features.npz';shards=[]
    for start in range(0,len(files),160):
     shard=C/f'dtd_{method}_{start:04d}.npz'
     if not shard.exists():np.savez_compressed(shard,features=encode(m,[imread(p) for p in files[start:start+160]]))
     shards.append(np.load(shard)['features']);print('C3 encode',method,start,'/5640',flush=True)
    np.savez_compressed(fp,features=np.concatenate(shards),names=np.array(names),labels=labels)
   a=np.load(fp);assert np.array_equal(a['names'],names) and np.array_equal(a['labels'],labels);f=a['features'];assert np.max(np.abs(np.linalg.norm(f,axis=1)-1))<1e-5
   fixed=[0,2819,5639];re=encode(m,[imread(files[i]) for i in fixed]);err=float(np.max(np.abs(re-f[fixed])));assert err<2e-5,err
   validation.append(dict(method=method,cache=fp.name,check='fixed feature replay [0,2819,5639]',max_abs_error=err,sha256=sha(fp)))
  for plan in plans:
   split=plan['split'];ri=np.array([ix[n] for n in plan['refs']]);qi=np.array([ix[n] for n in plan['queries']]);assert len(ri)==47 and len(qi)==1880 and not set(ri)&set(qi)
   proto=f[ri];variants=[(method,proto)]
   if m is not None:
    af=P/'new_experiments'/f'dtd_{method}_split{split}_A5.npz';check=[];ims=[]
    for k,n in enumerate(plan['refs']):
     for v in range(5):
      sd=int.from_bytes(hashlib.sha256(f'dtd|{split}|{n}|{v}'.encode()).digest()[:4],'little');torch.manual_seed(sd);im=aug(imread(root/'images'/n,False));ims.append(im);view_manifest.append(dict(method=method,split=split,path=n,view=v,seed=sd,pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest()))
    if af.exists():
     pp=np.load(af)['prototypes'];re=unit(encode(m,ims[:5]).mean(0));err=float(np.max(np.abs(re-pp[0])));assert err<2e-5,err;validation.append(dict(method=method,cache=af.name,check='first reference A5 replay',max_abs_error=err,sha256=sha(af)))
    else:
     af=C/f'dtd_{method}_split{split}_A5.npz'
     if af.exists():pp=np.load(af)['prototypes']
     else:pp=unit(encode(m,ims).reshape(47,5,-1).mean(1));np.savez_compressed(af,prototypes=pp)
    variants.append((method+'_A5',pp))
   for name,p in variants:
    met,order,ranks=metric(f[qi]@p.T,labels[qi]);results.append(dict(method=name,split=split,n_queries=len(qi),**met))
    pred.extend(dict(method=name,split=split,path=names[i],true_class=cats[labels[i]],predicted_class=cats[o[0]],rank=int(r)) for i,o,r in zip(qi,order,ranks))
   write(D/'dtd_results.csv',results);write(D/'dtd_predictions.csv',pred);write(O/'audit/dtd_cache_validation.csv',validation);write(O/'audit/dtd_view_manifest.csv',view_manifest);print('C3 scored',method,split,flush=True)
  del m;gc.collect()
 # paired view verification across backbone methods
 a=[r['pixel_sha256'] for r in view_manifest if r['method']=='resnet50'];b=[r['pixel_sha256'] for r in view_manifest if r['method']=='dinov2_s14'];assert a==b and len(a)==2350
 assert len(results)==60
 print('COMPLETE C3',flush=True)
def factor_inputs(factor,level):
 ims=[];meta=[]
 if factor in ['BOX','LANCZOS']:
  source=read(Q/'new_experiments'/f'inputs_resolution_{level}.csv')
  for r in source:
   if int(r['repeat'])!=0:continue
   base=imread(paths[int(r['query_index'])]);k=json.loads(r['parameters_json'])['intermediate_side'];im=base.resize((k,k),getattr(Image.Resampling,factor)).resize(base.size,Image.Resampling.BICUBIC);ims.append(im);meta.append(dict(r,pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest(),factor=factor))
 else:
  source=read(Q/'new_experiments'/f'inputs_device_pipeline_{level}.csv')
  for r in source:
   base=imread(paths[int(r['query_index'])]);a=np.asarray(base,dtype=np.float32)/255;rng=np.random.default_rng(int(r['seed']));mild=level=='mild'
   gains=rng.uniform(.9,1.1,3) if mild else rng.uniform(.75,1.25,3);gamma=float(rng.uniform(.9,1.1) if mild else rng.uniform(.75,1.3));sigma=float(rng.uniform(1,3) if mild else rng.uniform(3,7));quality=int(rng.integers(70,91) if mild else rng.integers(40,66));p=json.loads(r['parameters_json']);assert np.allclose(gains,p['channel_gains']) and gamma==p['gamma'] and quality==p['jpeg_quality'] and sigma==p['noise_sigma_255']
   noise=rng.normal(0,sigma,a.shape)
   if factor=='channel_gains':im=Image.fromarray(np.uint8(np.clip(a*gains,0,1)*255))
   elif factor=='gamma':im=Image.fromarray(np.uint8(a**gamma*255))
   elif factor=='noise':im=Image.fromarray(np.uint8(np.clip(a*255+noise,0,255)))
   elif factor=='jpeg':
    buf=io.BytesIO();base.save(buf,format='JPEG',quality=quality);buf.seek(0);im=Image.open(buf).convert('RGB')
   else:raise ValueError(factor)
   ims.append(im);meta.append(dict(r,pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest(),factor=factor))
 return ims,meta
def factors():
 manifest=[];results=[];pred=[];paired=[]
 methods=['rgb_histogram','gray_histogram','resnet50','dinov2_s14'];base={m:refs(m) for m in methods}
 clean={m:(hist([imread(p) for p in paths],m) if m.endswith('histogram') else np.load(P/'new_experiments'/f'{m}_rgb_features.npz')['my_dataset_test_filtered'])@base[m].T for m in methods}
 for method in methods:
  model=load(method) if not method.endswith('histogram') else None
  for factor in ['channel_gains','gamma','noise','jpeg','BOX','LANCZOS']:
   for lev in LEVELS:
    ims,meta=factor_inputs(factor,lev);labels=np.repeat(y,3) if factor not in ['BOX','LANCZOS'] else y
    if method==methods[0]:manifest.extend(meta)
    fp=C/f'c5_{method}_{factor}_{lev}.npz'
    if fp.exists():f=np.load(fp)['features']
    else:
     f=hist(ims,method) if model is None else encode(model,ims);np.savez_compressed(fp,features=f,labels=labels,input_hashes=np.array([r['pixel_sha256'] for r in meta]))
    score=f@base[method].T;met,rr=newpred(score,labels,meta,method,'C5',factor+'_'+lev);pred.extend(rr);hit=(score.argmax(1)==labels).reshape(201,-1).mean(1);lo,hi=ci(hit)
    results.append(dict(method=method,factor=factor,severity=lev,inputs=len(labels),**met,ci_low=lo,ci_high=hi))
    cleanhit=clean[method].argmax(1)==y
    full=np.load(Q/'new_experiments'/f'{method}_{"resolution" if factor in ["BOX","LANCZOS"] else "device_pipeline"}_{lev}.npz')['features'] if model is not None else np.load(C/f'B_{method}_{"resolution" if factor in ["BOX","LANCZOS"] else "device_pipeline"}_{lev}.npz')['features']
    fullhit=((full@base[method].T).argmax(1)==np.repeat(y,3)).reshape(201,3)
    if factor in ['BOX','LANCZOS']:fullhit=fullhit[:,0]
    else:fullhit=fullhit.mean(1)
    for name,other in [('minus_clean',cleanhit),('minus_B_resolution_repeat0' if factor in ['BOX','LANCZOS'] else 'minus_B_full_device_pipeline',fullhit)]:
     delta=hit-other;low,high=ci(delta);paired.append(dict(method=method,factor=factor,severity=lev,comparison=name,difference_pp=delta.mean()*100,ci_low=low,ci_high=high))
    write(D/'c5_results.csv',results);write(D/'c5_predictions.csv',pred);write(D/'c5_paired.csv',paired);write(O/'audit/c5_input_manifest.csv',manifest);print('C5',method,factor,lev,flush=True)
  del model;gc.collect()
 assert len(manifest)==5628;print('COMPLETE C5',flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['color','c2','dtd','factors','all']);args=p.parse_args()
 if args.mode=='all':color();c2();dtd();factors()
 else:globals()[args.mode]()
