"""Deterministic colour controls and full-matrix class-holdout diagnostics (CPU)."""
from pathlib import Path
import csv,json,hashlib,time,platform
import numpy as np
from PIL import Image,ImageOps
import torch
from torchvision import transforms,models
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[3]; OUT=Path(__file__).resolve().parents[1]
DATA=ROOT/'IC_project/01_核心代码与数据集'; DEST=OUT/'new_experiments'; DEST.mkdir(exist_ok=True)
torch.set_num_threads(4); torch.manual_seed(42)
CLASSES=sorted(p.name for p in (DATA/'dataset_train').iterdir() if p.is_dir())
SPLITS=['dataset_train','dataset_val','my_dataset_test_filtered']
resize=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224)])
norm=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])

def write_csv(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def prepare():
    images={};labels={};paths={}
    for split in SPLITS:
        images[split]=[];labels[split]=[];paths[split]=[]
        for c in CLASSES:
            pp=sorted(p for p in (DATA/split/c).glob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp'])
            if not pp: continue
            assert len(pp)==1
            with Image.open(pp[0]) as im: images[split].append(resize(im.convert('RGB')))
            labels[split].append(CLASSES.index(c));paths[split].append(str(pp[0].relative_to(ROOT)))
        print('prepared',split,len(images[split]),flush=True)
    return images,labels,paths

def transform_control(im,mode):
    if mode=='rgb': return im
    if mode=='gray': return ImageOps.grayscale(im).convert('RGB')
    a=np.asarray(im,dtype=np.float32)/255
    # Per-image per-channel moment normalization; this is a colour control, not learned adaptation.
    a=(a-a.mean((0,1),keepdims=True))/np.maximum(a.std((0,1),keepdims=True),1/255)*.2+.5
    return Image.fromarray(np.uint8(np.clip(a,0,1)*255))

def evaluate(scores,labels,paths,name,split,rows):
    order=np.argsort(-scores,axis=1,kind='stable')[:,:3];labels=np.array(labels)
    k1=int((order[:,0]==labels).sum());k3=int((order==labels[:,None]).any(1).sum())
    pred=[dict(image_path=p,true_class=CLASSES[y],top1_class=CLASSES[o[0]],top1_score=float(s[o[0]]),top2_class=CLASSES[o[1]],top2_score=float(s[o[1]]),top3_class=CLASSES[o[2]],top3_score=float(s[o[2]]),top1_correct=bool(o[0]==y),top3_correct=bool(y in o)) for p,y,o,s in zip(paths,labels,order,scores)]
    write_csv(DEST/f'{name}_{split}_predictions.csv',pred)
    row=dict(method=name,split=split,n=len(labels),top1_correct=k1,top3_correct=k3,top1=100*k1/len(labels),top3=100*k3/len(labels))
    rows.append(row);print(row,flush=True)

def auc(unknown_scores,known_scores):
    # Larger scores indicate known; rank-sum AUROC.
    vals=np.r_[known_scores,unknown_scores];r=rankdata(vals)
    n=len(known_scores);m=len(unknown_scores)
    return float((r[:n].sum()-n*(n+1)/2)/(n*m))

def open_set(matrices,labels):
    rng=np.random.default_rng(20261010); labels=np.array(labels);splits=[]
    for k in range(10):
        order=rng.permutation(len(labels));splits.append((order[:20],order[20:110],order[110:]))
    split_manifest=[dict(split=k,role=role,class_id=CLASSES[labels[i]]) for k,groups in enumerate(splits) for role,ix in zip(['unknown_evaluation','known_calibration','known_evaluation'],groups) for i in ix]
    write_csv(DEST/'open_set_partitions.csv',split_manifest)
    result=[]
    for method,scores in matrices.items():
        for k,(unknown,cal,known) in enumerate(splits):
            keep=np.ones(len(CLASSES),bool);keep[labels[unknown]]=False
            col=np.where(keep)[0];sub=scores[:,keep];order=np.argsort(-sub,axis=1,kind='stable')
            top=np.take_along_axis(sub,order[:,:2],axis=1);predict=col[order[:,0]]
            for rule,value in [('score',top[:,0]),('margin',top[:,0]-top[:,1])]:
                threshold=float(np.quantile(value[cal],.05));accept=value>=threshold
                result.append(dict(method=method,split=k,rule=rule,calibration_n=len(cal),known_evaluation_n=len(known),unknown_n=len(unknown),library_n=int(keep.sum()),threshold=threshold,
                    calibration_acceptance=float(accept[cal].mean()),known_acceptance=float(accept[known].mean()),unknown_rejection=float((~accept[unknown]).mean()),
                    known_correct_and_accepted=float(((predict[known]==labels[known])&accept[known]).mean()),known_auroc=auc(value[unknown],value[known])))
    write_csv(DEST/'open_set_disjoint_results.csv',result)
    (DEST/'open_set_protocol.json').write_text(json.dumps(dict(seed=20261010,partitions=10,unknown_classes=20,calibration_classes=90,evaluation_known_classes=91,library_classes=183,notes='All 20 unknown identities removed for every query. Class-disjoint calibration/evaluation within each partition; overlapping partitions are not independent datasets. Acquisition provenance remains unverified.'),indent=2))

def main():
    start=time.time();images,labels,paths=prepare();rows=[]; matrices={}
    # Joint 8x8x8 RGB histogram, square-rooted L1 histogram (Hellinger embedding).
    for mode in ['rgb','gray']:
        fs={}
        for split,ims in images.items():
            feats=[]
            for im in ims:
                a=np.asarray(im)
                if mode=='rgb': hist=np.histogramdd(a.reshape(-1,3),bins=(8,8,8),range=((0,256),)*3)[0].ravel()
                else: hist=np.histogram(np.asarray(im.convert('L')),bins=32,range=(0,256))[0]
                feats.append(np.sqrt(hist/hist.sum()))
            fs[split]=np.stack(feats)
        for split in SPLITS[1:]:evaluate(fs[split]@fs[SPLITS[0]].T,labels[split],paths[split],mode+'_histogram',split,rows)
    for model_name in ['resnet50','dinov2_s14']:
        if model_name=='resnet50':
            model=models.resnet50(weights=None);model.load_state_dict(torch.load(Path.home()/'.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth',map_location='cpu'));model.fc=torch.nn.Identity()
        else:
            model=torch.hub.load(str(Path.home()/'.cache/torch/hub/facebookresearch_dinov2_main'),'dinov2_vits14',source='local',pretrained=True)
        model.eval()
        for mode in ['rgb','gray','color_normalized']:
            cache=DEST/f'{model_name}_{mode}_features.npz'
            fs={}
            if cache.exists(): fs=dict(np.load(cache))
            else:
                for split in SPLITS:
                    feat=[]
                    with torch.inference_mode():
                        for pos in range(0,len(images[split]),16):
                            batch=torch.stack([norm(transform_control(im,mode)) for im in images[split][pos:pos+16]])
                            f=model(batch);feat.append(torch.nn.functional.normalize(f,dim=1).numpy())
                    fs[split]=np.concatenate(feat);print('encoded',model_name,mode,split,flush=True)
                np.savez_compressed(cache,**fs)
            for split in SPLITS[1:]:
                score=fs[split]@fs[SPLITS[0]].T
                evaluate(score,labels[split],paths[split],model_name+'_'+mode,split,rows)
                if mode=='rgb' and split==SPLITS[-1]: matrices[model_name]=score
            write_csv(DEST/'color_controls_summary.csv',rows)
    np.savez_compressed(DEST/'rgb_full_similarity_matrices.npz',**matrices,query_labels=np.array(labels[SPLITS[-1]]),class_names=np.array(CLASSES))
    open_set(matrices,labels[SPLITS[-1]])
    env=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,threads=4,device='cpu',elapsed_seconds=time.time()-start)
    (DEST/'runtime.json').write_text(json.dumps(env,indent=2));print('COMPLETE',env,flush=True)

if __name__=='__main__':main()
