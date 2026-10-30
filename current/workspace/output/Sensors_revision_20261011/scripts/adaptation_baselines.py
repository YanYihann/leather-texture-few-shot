"""ResNet50 linear/partial/full adaptation with matched image exposure and S-only tuning."""
from pathlib import Path
import csv,json,time,hashlib,copy
import torch,numpy as np
from PIL import Image
from torchvision import models
from recompute_diagnostics import ROOT,OUT,DATA,DEST,CLASSES,norm,resize,write_csv
torch.set_num_threads(3)
OLD=ROOT/'output/Sensors_revision_20261010/new_experiments'
SAVE=DEST/'adaptation';SAVE.mkdir(exist_ok=True)
F=np.load(OLD/'resnet50_rgb_features.npz');initial=torch.from_numpy(F['dataset_train']).float()
paths={s:[next((DATA/s/c).glob('*')) for c in CLASSES if (DATA/s/c).exists()] for s in ['dataset_val','my_dataset_test_filtered']}
labels={s:torch.tensor([CLASSES.index(p.parent.name) for p in ps]) for s,ps in paths.items()}
images={s:torch.stack([norm(resize(Image.open(p).convert('RGB'))) for p in ps]) for s,ps in paths.items()}
EPOCHS=10;rows=[];history=[];selection=[]
def infer(net,head,x,linear=False):
    vals=[]
    with torch.inference_mode():
        for i in range(0,len(x),16):
            f=x[i:i+16] if linear else net(x[i:i+16])
            vals.append(20*torch.nn.functional.normalize(f,dim=1)@torch.nn.functional.normalize(head,dim=1).T)
    return torch.cat(vals)
def metrics(scores,y):
    order=scores.argsort(descending=True,stable=True);return int((order[:,0]==y).sum()),int((order[:,:3]==y[:,None]).any(1).sum())
def run(mode,seed,lr):
    torch.manual_seed(seed);started=time.time();linear=mode=='linear'
    checkpoint=SAVE/f'{mode}_{seed}_{lr}.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu')
        hh=SAVE/f'{mode}_{seed}_{lr}_history.json'
        if hh.exists():history.extend(json.loads(hh.read_text()))
        else:
            import re
            lines=(OUT/'audit/adaptation.log').read_text(encoding='utf8')
            for m in re.finditer(r'(linear|partial|full) (\d+) ([\de.+-]+) (\d+) dev (\d+)',lines):
                if m[1]==mode and int(m[2])==seed and float(m[3])==lr:
                    history.append(dict(mode=mode,seed=seed,lr=lr,epoch=int(m[4]),dev_correct=int(m[5]),updates=int(m[4])*13,loss_last=''))
        print('reused completed checkpoint',mode,seed,lr,flush=True)
        return dict(mode=mode,seed=seed,lr=lr,dev_correct=saved['development_correct'],epoch=saved['selected_epoch'],seconds='',checkpoint=checkpoint.name)
    net=models.resnet50(weights=None);net.load_state_dict(torch.load(Path.home()/'.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth',map_location='cpu'));net.fc=torch.nn.Identity();net.eval()
    for name,p in net.named_parameters():p.requires_grad_(mode=='full' or (mode=='partial' and name.startswith('layer4.')))
    head=torch.nn.Parameter(initial.clone())
    opt=torch.optim.Adam([{'params':[p for p in net.parameters() if p.requires_grad],'lr':lr},{'params':[head],'lr':lr if linear else lr*10}],weight_decay=1e-4)
    feats=np.load(OLD/f'controlled_resnet50_seed{seed}_N10_embeddings.npz')['features']
    best=-1;beststate=None;besthead=None;bestepoch=0
    resume=SAVE/f'{mode}_{seed}_{lr}_resume.pt';begin=0
    if resume.exists():
        d=torch.load(resume,map_location='cpu');net.load_state_dict(d['net']);head.data.copy_(d['head']);opt.load_state_dict(d['optimizer']);torch.set_rng_state(d['rng']);begin=d['next_epoch'];best=d['best'];beststate=d['beststate'];besthead=d['besthead'];bestepoch=d['bestepoch'];history.extend(d['history']);del d
    for epoch in range(begin,EPOCHS):
        order=torch.randperm(len(CLASSES))
        for pos in range(0,len(order),16):
            ix=order[pos:pos+16]
            if linear:f=torch.from_numpy(feats[ix.numpy(),epoch]).float()
            else:
                batch=torch.stack([norm(Image.open(OLD/'controlled_views_hue0'/f'{seed}_{CLASSES[i]}_{epoch:02d}.png').convert('RGB')) for i in ix.tolist()]);f=net(batch)
            logits=20*torch.nn.functional.normalize(f,dim=1)@torch.nn.functional.normalize(head,dim=1).T
            loss=torch.nn.functional.cross_entropy(logits,ix);opt.zero_grad();loss.backward();opt.step()
        dev=infer(net,head,torch.from_numpy(F['dataset_val']).float() if linear else images['dataset_val'],linear)
        correct,_=metrics(dev,labels['dataset_val']);history.append(dict(mode=mode,seed=seed,lr=lr,epoch=epoch+1,dev_correct=correct,updates=(epoch+1)*13,loss_last=float(loss.detach())))
        if correct>best:
            best=correct;bestepoch=epoch+1;besthead=head.detach().clone();beststate={k:v.detach().clone() for k,v in net.state_dict().items()} if not linear else None
        print(mode,seed,lr,epoch+1,'dev',correct,flush=True)
        torch.save(dict(net=net.state_dict(),head=head.detach(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),next_epoch=epoch+1,best=best,beststate=beststate,besthead=besthead,bestepoch=bestepoch,history=[r for r in history if r['mode']==mode and r['seed']==seed and r['lr']==lr]),resume)
    checkpoint=SAVE/f'{mode}_{seed}_{lr}.pt';torch.save(dict(backbone=beststate,head=besthead,mode=mode,seed=seed,lr=lr,selected_epoch=bestepoch,development_correct=best),checkpoint)
    (SAVE/f'{mode}_{seed}_{lr}_history.json').write_text(json.dumps([r for r in history if r['mode']==mode and r['seed']==seed and r['lr']==lr]))
    return dict(mode=mode,seed=seed,lr=lr,dev_correct=best,epoch=bestepoch,seconds=time.time()-started,checkpoint=checkpoint.name)
for mode in ['linear','partial','full']:
    rates=[.001,.01] if mode=='linear' else [.00001,.0001]
    candidates=[run(mode,42,lr) for lr in rates]
    chosen=max(candidates,key=lambda r:r['dev_correct']);selected=[chosen]
    for seed in [43,44]:selected.append(run(mode,seed,chosen['lr']))
    selection.extend(candidates+selected[1:]);write_csv(SAVE/'selection.csv',selection);write_csv(SAVE/'training_history.csv',history)
    # Q is read only after rate selection for this mode. No Q feedback selects settings.
    for record in selected:
        d=torch.load(SAVE/record['checkpoint'],map_location='cpu');linear=mode=='linear'
        net=models.resnet50(weights=None);net.fc=torch.nn.Identity()
        if not linear:net.load_state_dict(d['backbone'])
        net.eval()
        for split in paths:
            scores=infer(net,d['head'],torch.from_numpy(F[split]).float() if linear else images[split],linear);order=scores.argsort(descending=True,stable=True);k1,k3=metrics(scores,labels[split])
            write_csv(SAVE/f'{mode}_{record["seed"]}_{split}_predictions.csv',[dict(class_id=CLASSES[y],top1=CLASSES[o[0]],top2=CLASSES[o[1]],top3=CLASSES[o[2]],score=float(scores[i,o[0]])) for i,(y,o) in enumerate(zip(labels[split],order))])
            rows.append(dict(method=mode,seed=record['seed'],lr=record['lr'],epoch=record['epoch'],split=split,n=len(order),top1=100*k1/len(order),top3=100*k3/len(order),top1_correct=k1,top3_correct=k3,checkpoint_sha256=hashlib.sha256((SAVE/record['checkpoint']).read_bytes()).hexdigest()))
    write_csv(SAVE/'results.csv',rows)
(SAVE/'protocol.json').write_text(json.dumps(dict(backbone='ImageNet ResNet50',epochs=10,batch=16,updates_per_run=130,images_per_run=2030,learning_rates={'linear':[.001,.01],'partial_full':[1e-5,1e-4]},tune_seed=42,repeat_seeds=[42,43,44],selection='highest S Top-1; first epoch/rate on ties; Q excluded',head='cosine linear, scale20, initialized from frozen reference features',partial='layer4 + head',batchnorm='fixed eval running statistics for all modes',augmentation='identical cached 10 views per specimen/seed; epoch indexes view',budget='2 rate candidates on seed42, then chosen rate at seeds43,44; equal epoch/exposure budget across modes'),indent=2))
print('COMPLETE adaptation',flush=True)
