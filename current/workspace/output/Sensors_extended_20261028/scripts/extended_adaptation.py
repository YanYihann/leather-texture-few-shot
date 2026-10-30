"""Exploratory 50-epoch adaptation; S-only model selection, untouched previous release."""
from pathlib import Path
import torch,numpy as np,csv,json,time,math,hashlib,sys
from PIL import Image
from torchvision import models,transforms
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
DATA=ROOT/'IC_project/01_核心代码与数据集';OLD=ROOT/'output/Sensors_revision_20261010/new_experiments';SAVE=OUT/'new_experiments/adaptation50';SAVE.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(3);torch.set_num_interop_threads(1)
CLASSES=sorted(p.name for p in (DATA/'dataset_train').iterdir() if p.is_dir())
F=np.load(OLD/'resnet50_rgb_features.npz');initial=torch.from_numpy(F['dataset_train']).float()
norm=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))]);transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),norm])
def csvout(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def loadsplit(s):
    paths=[next((DATA/s/c).glob('*')) for c in CLASSES if (DATA/s/c).is_dir()]
    return torch.stack([transform(Image.open(p).convert('RGB')) for p in paths]),torch.tensor([CLASSES.index(p.parent.name) for p in paths])
dev,dy=loadsplit('dataset_val')
def infer(net,head,x,linear=False):
    with torch.inference_mode():
        return torch.cat([20*torch.nn.functional.normalize(x[i:i+16] if linear else net(x[i:i+16]),dim=1)@torch.nn.functional.normalize(head,dim=1).T for i in range(0,len(x),16)])
def run(mode,seed,lr):
    dst=SAVE/f'{mode}_{seed}_{lr}.pt';hist=SAVE/f'{mode}_{seed}_{lr}.json';resume=SAVE/f'{mode}_{seed}_{lr}_resume.pt'
    if dst.exists():return json.loads(hist.read_text())['selection']
    torch.manual_seed(seed);started=time.time();linear=mode=='linear';net=models.resnet50(weights=None)
    net.load_state_dict(torch.load(Path.home()/'.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth',map_location='cpu'));net.fc=torch.nn.Identity();net.eval()
    for name,p in net.named_parameters():p.requires_grad_(mode=='full' or (mode=='partial' and name.startswith('layer4.')))
    head=torch.nn.Parameter(initial.clone());opt=torch.optim.Adam([{'params':[p for p in net.parameters() if p.requires_grad],'lr':lr},{'params':[head],'lr':lr if linear else 10*lr}],weight_decay=1e-4)
    feats=np.load(OLD/f'controlled_resnet50_seed{seed}_N10_embeddings.npz')['features'];best=-1;bestepoch=0;history=[];begin=0;beststate=None;besthead=None
    # Decode uint8 views once per run: 305 MB, preserving original lossless cached pixels.
    views=None
    if not linear:
        views=np.stack([np.stack([np.asarray(Image.open(OLD/'controlled_views_hue0'/f'{seed}_{c}_{v:02d}.png').convert('RGB')) for v in range(10)]) for c in CLASSES]);print('decoded',mode,seed,lr,flush=True)
    if resume.exists():
        d=torch.load(resume,map_location='cpu');net.load_state_dict(d['net']);head.data.copy_(d['head']);opt.load_state_dict(d['optimizer']);torch.set_rng_state(d['rng']);begin=d['next_epoch'];best=d['best'];bestepoch=d['bestepoch'];beststate=d['beststate'];besthead=d['besthead'];history=d['history'];started-=float(history[-1]['elapsed_s']) if history else 0.;del d
        print('resumed saved epoch',mode,seed,lr,begin,flush=True)
    for epoch in range(begin,50):
        factor=.01+.99*(1+math.cos(math.pi*epoch/49))/2
        for group,base in zip(opt.param_groups,[lr,lr if linear else 10*lr]):group['lr']=base*factor
        order=torch.randperm(203);loss_sum=0.
        for pos in range(0,203,16):
            ix=order[pos:pos+16];view=epoch%10
            if linear:f=torch.from_numpy(feats[ix.numpy(),view]).float()
            else:
                x=torch.from_numpy(views[ix.numpy(),view].copy()).permute(0,3,1,2).float()/255
                x=(x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None];f=net(x)
            logits=20*torch.nn.functional.normalize(f,dim=1)@torch.nn.functional.normalize(head,dim=1).T
            loss=torch.nn.functional.cross_entropy(logits,ix);opt.zero_grad(set_to_none=True);loss.backward();opt.step();loss_sum+=float(loss.detach())*len(ix)
        scores=infer(net,head,torch.from_numpy(F['dataset_val']).float() if linear else dev,linear);correct=int((scores.argmax(1)==dy).sum())
        history.append(dict(epoch=epoch+1,dev_correct=correct,updates=(epoch+1)*13,presentations=(epoch+1)*203,mean_loss=loss_sum/203,lr=lr*factor,elapsed_s=time.time()-started))
        if correct>best:best=correct;bestepoch=epoch+1;besthead=head.detach().clone();beststate={k:v.detach().clone() for k,v in net.state_dict().items()} if not linear else None
        if epoch==0 or (epoch+1)%5==0:print(mode,seed,lr,'epoch',epoch+1,'S',correct,'best',best,'seconds',round(time.time()-started),flush=True)
        if (epoch+1)%5==0:
            torch.save(dict(net=net.state_dict(),head=head.detach(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),next_epoch=epoch+1,best=best,beststate=beststate,besthead=besthead,bestepoch=bestepoch,history=history),resume)
            hist.write_text(json.dumps(dict(history=history),indent=2),encoding='utf8')
    torch.save(dict(backbone=beststate,head=besthead,mode=mode,seed=seed,lr=lr,selected_epoch=bestepoch,development_correct=best),dst)
    record=dict(mode=mode,seed=seed,lr=lr,dev_correct=best,epoch=bestepoch,epochs_executed=50,updates_executed=650,presentations=10150,unique_views=2030,seconds=history[-1]['elapsed_s'],checkpoint=dst.name,checkpoint_sha256=hashlib.sha256(dst.read_bytes()).hexdigest());hist.write_text(json.dumps(dict(selection=record,history=history),indent=2),encoding='utf8');return record
protocol=dict(exploratory=True,prior_Q_results_already_seen=True,Q_not_used_for_selection=True,epochs=50,batch=16,updates_per_run=650,presentations_per_run=10150,unique_views_per_run=2030,views='same 10 cached views as original budget, cyclically reused five times',rates={'linear':[.001,.01,.1],'partial_full':[1e-5,3e-5,1e-4]},optimizer='Adam, weight_decay=1e-4',schedule='cosine decay, factor 1 to .01',batchnorm='fixed pretrained running statistics; affine parameters follow adaptation scope',selection='S Top-1; earliest epoch; first rate on ties; seed42 rate selection then chosen rate seeds43/44',backbone='ImageNet1K V2 ResNet50, fixed original checkpoint',head='reference initialized cosine head, scale20, partial/full head rate 10x backbone',retained_old_budget='10 epochs, 130 updates, constant lr, two rate candidates; separate historical comparator')
(SAVE/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
selections=[];rows=[]
for mode in ['linear','partial','full']:
    rates=protocol['rates']['linear' if mode=='linear' else 'partial_full'];candidates=[run(mode,42,lr) for lr in rates];chosen=max(candidates,key=lambda z:z['dev_correct']);selected=[chosen]+[run(mode,s,chosen['lr']) for s in [43,44]];selections.extend(candidates+selected[1:]);csvout(SAVE/'selection.csv',selections)
    # No Q images/features enter training or rate/epoch selection. Evaluate after all runs of this scope.
    q,qy=loadsplit('my_dataset_test_filtered')
    for r in selected:
        d=torch.load(SAVE/r['checkpoint'],map_location='cpu');net=models.resnet50(weights=None);net.fc=torch.nn.Identity()
        if mode!='linear':net.load_state_dict(d['backbone'])
        net.eval()
        for split,x,y in [('dataset_val',dev,dy),('my_dataset_test_filtered',q,qy)]:
            scores=infer(net,d['head'],torch.from_numpy(F[split]).float() if mode=='linear' else x,mode=='linear');order=scores.argsort(descending=True,stable=True);rank=(order==y[:,None]).int().argmax(1)+1
            np.savez_compressed(SAVE/f'{mode}_{r["seed"]}_{split}_scores.npz',scores=scores.numpy(),labels=y.numpy(),classes=np.array(CLASSES))
            csvout(SAVE/f'{mode}_{r["seed"]}_{split}_predictions.csv',[dict(class_id=CLASSES[int(yi)],rank=int(ri),top1=CLASSES[int(oi[0])],top2=CLASSES[int(oi[1])],top3=CLASSES[int(oi[2])]) for yi,ri,oi in zip(y,rank,order)])
            rows.append(dict(method=mode,seed=r['seed'],lr=r['lr'],epoch=r['epoch'],split=split,n=len(y),top1_correct=int((rank==1).sum()),top3_correct=int((rank<=3).sum()),top1=float((rank==1).float().mean())*100,top3=float((rank<=3).float().mean())*100,checkpoint_sha256=r['checkpoint_sha256']))
    csvout(SAVE/'results.csv',rows)
print('COMPLETE adaptation50',flush=True)
