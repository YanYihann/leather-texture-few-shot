"""Cached ranking replay, independent arithmetic checks and source-manifest verification."""
from pathlib import Path
import sys,csv,json,hashlib,datetime,math
from collections import defaultdict
import numpy as np
O=Path(__file__).resolve().parents[1];ROOT=O.parents[1];P=ROOT/'output/Sensors_revision_20261011';Q=ROOT/'output/Sensors_revision_20261027';D=O/'results';C=O/'cache';R=C/'complete_rankings';R.mkdir(exist_ok=True)
from threadpoolctl import threadpool_limits
limits=threadpool_limits(3)
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def unit(z):return z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-12)
METHODS=['rgb_histogram','gray_histogram','resnet50','dinov2_s14'];classes=np.load(C/'current_clean_scores.npz')['classes'];y=np.load(C/'current_clean_scores.npz')['labels'];ids=[str(classes[i]) for i in y];boot=np.random.default_rng(20261027).integers(0,201,(2000,201))
def interval(v):return np.quantile(v[boot].mean(1),[.025,.975])*100
def clean_reference(m):
 if not m.endswith('histogram'):return np.load(P/'new_experiments'/f'{m}_rgb_features.npz')['dataset_train']
 from PIL import Image
 from torchvision import transforms
 resize=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224)]);out=[];data=ROOT/'IC_project/01_核心代码与数据集/dataset_train'
 for c in classes:
  with Image.open(next((data/str(c)).glob('*'))) as src:im=resize(src.convert('RGB'))
  if m=='rgb_histogram':h=np.histogramdd(np.asarray(im).reshape(-1,3),bins=(8,)*3,range=((0,256),)*3)[0].ravel()
  else:h=np.histogram(np.asarray(im.convert('L')),bins=32,range=(0,256))[0]
  out.append(np.sqrt(h/h.sum()))
 return np.array(out,dtype=np.float32)
def main():
 summary=[];checks=[];pred=read(D/'c1_predictions.csv');clean=read(D/'clean_metrics.csv');rows=read(D/'c1_results.csv')
 clean1={m:float(next(r for r in clean if r['method']==m and r['variant']=='frozen')['recall1']) for m in METHODS}
 for r in rows:r['clean_recall1']=clean1[r['method']];r['drop_pp']=clean1[r['method']]-float(r['recall1'])
 write(D/'c1_results.csv',rows)
 by=defaultdict(list)
 for r in pred:by[r['protocol'],r['method'],r['condition']].append(r)
 failure=[]
 for m in METHODS:
  ref=clean_reference(m)
  for protocol,method,condition in sorted(by):
   if method!=m:continue
   fam,sev=condition.rsplit('_',1);fn=C/f'{protocol}_{m}_{fam}_{sev}.npz' if m.endswith('histogram') else (P/'new_experiments'/f'robust_{m}_{fam}_{sev}.npz' if protocol=='A' else Q/'new_experiments'/f'{m}_{fam}_{sev}.npz');cache=np.load(fn);f=cache['features'];labels=cache['labels'];score=f@ref.T;order=np.argsort(-score,axis=1,kind='stable');rank=(order==labels[:,None]).argmax(1)+1;rr=by[protocol,m,condition]
   assert np.array_equal(rank,np.array([int(r['rank']) for r in rr]));assert np.array_equal(labels,np.array([np.where(classes==r['class_id'])[0][0] for r in rr]));assert np.max(np.abs(np.linalg.norm(f,axis=1)-1))<1e-5
   # Rankings reconstruct every candidate identity, not only Top-5.
   np.savez_compressed(R/f'{protocol}_{m}_{condition}.npz',order=order.astype(np.uint16),true_label=labels,classes=classes)
   cc=(rank==1).reshape(201,-1).mean(1);lo,hi=interval(cc);failure.append(dict(protocol=protocol,method=m,variant='frozen',family=fam,severity=sev,failure_rate=100*(1-cc.mean()),ci_low=100-hi,ci_high=100-lo))
   checks.append(dict(check='C1 full ranking replay',method=m,condition=protocol+'_'+condition,n=len(rank),pass_check=True))
  if (D/'c5_results.csv').exists() and len(read(D/'c5_results.csv'))==48:
   c5=read(D/'c5_predictions.csv');groups=defaultdict(list)
   for r in c5:
    if r['method']==m:groups[r['condition']].append(r)
   for condition,rr in groups.items():
    factor,sev=condition.rsplit('_',1);cache=np.load(C/f'c5_{m}_{factor}_{sev}.npz');histcache=np.load(C/f'c5_rgb_histogram_{factor}_{sev}.npz');assert np.array_equal(cache['input_hashes'],histcache['input_hashes']);score=cache['features']@ref.T;order=np.argsort(-score,axis=1,kind='stable');labels=cache['labels'];rank=(order==labels[:,None]).argmax(1)+1;assert np.array_equal(rank,np.array([int(r['rank']) for r in rr]));np.savez_compressed(R/f'C5_{m}_{condition}.npz',order=order.astype(np.uint16),true_label=labels,classes=classes);checks.append(dict(check='C5 full ranking / shared pixel replay',method=m,condition=condition,n=len(rank),pass_check=True))
 for m in ['resnet50','dinov2_s14']:
  # The former duplicate cache copies are identical, permitting a single current source.
  for split in ['dataset_train','my_dataset_test_filtered']:
   x=np.load(P/'new_experiments'/f'{m}_rgb_features.npz')[split];z=np.load(ROOT/'output/Sensors_revision_20261010/new_experiments'/f'{m}_rgb_features.npz')[split];assert np.array_equal(x,z)
  for seed in [42,43,44]:
   pp=unit(np.load(P/'new_experiments'/f'controlled_{m}_seed{seed}_N50_embeddings.npz')['features'][:,:30].mean(1))
   for fam in ['regional_crop','perspective','resolution','illumination','device_pipeline','compound']:
    for sev in ['mild','moderate']:
     f=np.load(Q/'new_experiments'/f'{m}_{fam}_{sev}.npz')['features'];labels=np.repeat(y,3);score=f@pp.T;order=np.argsort(-score,axis=1,kind='stable');np.savez_compressed(R/f'B_{m}_A30_seed{seed}_{fam}_{sev}.npz',order=order.astype(np.uint16),true_label=labels,classes=classes)
     cc=(order[:,0]==labels).reshape(201,3).mean(1);lo,hi=interval(cc);failure.append(dict(protocol='B',method=m,variant='A30_seed'+str(seed),family=fam,severity=sev,failure_rate=100*(1-cc.mean()),ci_low=100-hi,ci_high=100-lo))
 write(D/'family_failure_rates.csv',failure)
 stability=read(D/'query_stability.csv')
 for m,v in dict.fromkeys((r['method'],r['variant']) for r in stability):
  rr=[r for r in stability if r['method']==m and r['variant']==v];frac=np.array([np.mean([float(r['correct_fraction']) for r in rr if r['class_id']==c]) for c in ids]);all36=np.array([np.mean([int(r['all36']) for r in rr if r['class_id']==c]) for c in ids]);lo,hi=interval(all36);summary.append(dict(method=m,variant=v,original_queries=201,mean_fraction=frac.mean(),median_fraction=np.median(frac),q10=np.quantile(frac,.1),q90=np.quantile(frac,.9),all36=all36.mean()*100,all36_ci_low=lo,all36_ci_high=hi))
 write(D/'stability_summary.csv',summary)
 # DTD independent prediction-count verification, all methods and all ten splits.
 dt=read(D/'dtd_results.csv');dp=read(D/'dtd_predictions.csv');assert len(dt)==60 and len(dp)==112800
 plans=json.loads((P/'new_experiments/dtd_protocol.json').read_text())['splits'];dtdroot=P/'external/dtd';planchecks=[]
 for plan in plans:
  split=plan['split'];train=set((dtdroot/f'labels/train{split}.txt').read_text().splitlines());test=(dtdroot/f'labels/test{split}.txt').read_text().splitlines();assert plan['queries']==test and len(test)==1880;assert len(plan['refs'])==47 and all(n in train for n in plan['refs']);assert len({n.split('/')[0] for n in plan['refs']})==47;assert not set(plan['refs'])&set(test);planchecks.append(dict(split=split,train_references=47,official_test_queries=1880,exact_official_test_order=True,all_references_from_train=True))
 write(O/'audit/dtd_official_plan_verification.csv',planchecks)
 for r in dt:
  ss=[z for z in dp if z['method']==r['method'] and z['split']==r['split']];rank=np.array([int(z['rank']) for z in ss]);assert len(ss)==1880;assert abs((rank==1).mean()*100-float(r['recall1']))<1e-10;assert abs((rank<=3).mean()*100-float(r['recall3']))<1e-10;assert abs((1/rank).mean()-float(r['mrr']))<1e-12
 checks.append(dict(check='DTD all 60 prediction tables',method='all',condition='all10splits',n=len(dp),pass_check=True))
 # Source and old deliverables remain preserved; original figures are byte-identical.
 assert sha(Q/'package/main.pdf')=='3c2ec1326e507463d78f78e96cdc3e594c8608d9625d0544275a04de76feedeb'
 for i in range(1,14):assert sha(O/'preserved_original_figures'/f'Fig{i}.pdf')==sha(ROOT/'Sensors_submission_20260824/package'/f'Fig{i}.pdf')
 images=read(P/'audit/image_manifest.csv');assert len(images)==810
 for r in images:assert sha(ROOT/r['path'])==r['file_sha256']
 cleanmat=np.load(C/'current_clean_scores.npz');hr=cleanmat['resnet50'].argmax(1)==y;hd=cleanmat['dinov2_s14'].argmax(1)==y;only_r=int((hr&~hd).sum());only_d=int((hd&~hr).sum());n=only_r+only_d;pv=2*sum(math.comb(n,k) for k in range(min(only_r,only_d)+1))/2**n;assert (only_r,only_d)==(10,3) and abs(pv-.09228515625)<1e-12
 for m in METHODS:
  if (O/'audit'/f'cost_{m}.json').exists():
   rr=read(D/f'cost_query_{m}.csv');reg=read(D/f'cost_registration_{m}.csv');assert len(rr)==100 and len(reg)==9;assert all(float(r[k])>0 for r in rr for k in ['preprocess_ms','encode_ms','match_ms','total_ms']);assert {int(r['encodings']) for r in reg}=={203,1015,6090}
 write(O/'audit/numerical_checks.csv',checks)
 meta=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_pass=True,C1_conditions=len(rows),C1_predictions=len(pred),C5_conditions=len(read(D/'c5_results.csv')) if (D/'c5_results.csv').exists() else 0,DTD_method_splits=len(dt),DTD_predictions=len(dp),rank_cache_files=len(list(R.glob('*.npz'))),query_identities=201,reference_identities=203,original_figures=13,original_image_hashes_checked=810,McNemar=dict(resnet_only=only_r,dino_only=only_d,exact_two_sided_p=pv),base_pdf_unchanged=True,scope='cached ranking and arithmetic replay; not independent acquisition validation or full training reproduction')
 (O/'audit/numerical_verification.json').write_text(json.dumps(meta,indent=2),encoding='utf8');print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
