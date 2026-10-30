"""Matched-budget cosine linear probe, last-stage and full ResNet50 adaptation."""
from common import *
import argparse
import copy
import math


def checkpoint_path(mode,seed,lr):
    return OUT/'checkpoints'/f'{mode}_seed{seed}_lr{lr:g}.pt'


def save_torch(path,value):
    temporary=Path(str(path)+'.tmp');torch.save(value,temporary);temporary.replace(path)


def state_of(net,mode):
    if mode=='linear':return None
    module=net.layer4 if mode=='partial' else net
    return {key:value.detach().clone() for key,value in module.state_dict().items()}


def apply_state(net,state,mode):
    if mode=='partial':net.layer4.load_state_dict(state)
    elif mode=='full':net.load_state_dict(state)


def logits(net,head,mode,data):
    predictions=[]
    with torch.inference_mode():
        for pos in range(0,len(data),16):
            if mode=='linear':f=torch.from_numpy(np.array(data[pos:pos+16])).float()
            elif mode=='partial':
                x=torch.from_numpy(np.array(data[pos:pos+16])).contiguous(memory_format=torch.channels_last)
                f=net.avgpool(net.layer4(x)).flatten(1)
            else:f=net(tensor(data[pos:pos+16]))
            predictions.append(20*torch.nn.functional.normalize(f,dim=1)@torch.nn.functional.normalize(head,dim=1).T)
    return torch.cat(predictions)


def run(mode,seed,lr,ix,labels,max_seconds):
    checkpoint=checkpoint_path(mode,seed,lr)
    signature=filehash(OUT/'protocol/experiment_protocol.json')
    if checkpoint.exists():
        value=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert value['protocol_sha256']==signature
        print('reuse completed training',checkpoint.name,flush=True)
        record=dict(value['record'])
        record['checkpoint_sha256']=filehash(checkpoint)
        return record
    started=time.time();si=SEEDS.index(seed)
    net,weight=load_model('resnet50')
    for name,parameter in net.named_parameters():
        parameter.requires_grad_(mode=='full' or (mode=='partial' and name.startswith('layer4.')))
    net.eval()
    base=np.load(OUT/'cache/resnet50_base.npy',mmap_mode='r')
    head=torch.nn.Parameter(torch.from_numpy(base[ix['reference']].copy()))
    trainable=[parameter for parameter in net.parameters() if parameter.requires_grad]
    optimizer=torch.optim.Adam([{'params':trainable,'lr':lr},
                                {'params':[head],'lr':lr if mode=='linear' else 10*lr}],weight_decay=1e-4)
    if mode=='linear':
        train=np.load(OUT/'cache/resnet50_aug.npy',mmap_mode='r')[si,:,:10]
        validation=base[ix['validation']]
    elif mode=='partial':
        train=np.load(OUT/'cache/resnet50_aug_layer3.npy',mmap_mode='r')[si]
        validation=np.load(OUT/'cache/resnet50_validation_layer3.npy',mmap_mode='r')
    else:
        train=np.load(OUT/'cache/aug_pixels.npy',mmap_mode='r')[si,:,:10]
        validation=np.load(OUT/'cache/center_pixels.npy',mmap_mode='r')[ix['validation']]
    torch.manual_seed(seed)
    best=-1;best_epoch=0;besthead=None;bestnet=None;history=[];begin=0
    resume=Path(str(checkpoint)+'.resume')
    if resume.exists():
        value=torch.load(resume,map_location='cpu',weights_only=False)
        assert value['protocol_sha256']==signature
        apply_state(net,value['backbone'],mode);head.data.copy_(value['head']);optimizer.load_state_dict(value['optimizer'])
        torch.set_rng_state(value['rng']);begin=value['next_epoch'];best=value['best'];best_epoch=value['best_epoch']
        besthead=value['besthead'];bestnet=value['bestnet'];history=value['history'];started-=value['elapsed_seconds']
        print('resumed',mode,seed,lr,'epoch',begin,flush=True)
    for epoch in range(begin,10):
        if time.time()-started>max_seconds:raise TimeoutError('Configured per-run hard timeout reached')
        order=torch.randperm(355);total_loss=0.;correct_train=0
        for pos in range(0,355,16):
            indices=order[pos:pos+16]
            pixels=np.array(train[indices.numpy(),epoch])
            if mode=='linear':f=torch.from_numpy(pixels).float()
            elif mode=='partial':
                x=torch.from_numpy(pixels).contiguous(memory_format=torch.channels_last)
                f=net.avgpool(net.layer4(x)).flatten(1)
            else:f=net(tensor(pixels))
            score=20*torch.nn.functional.normalize(f,dim=1)@torch.nn.functional.normalize(head,dim=1).T
            loss=torch.nn.functional.cross_entropy(score,indices)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            total_loss+=float(loss.detach())*len(indices);correct_train+=int((score.argmax(1)==indices).sum())
        validation_score=logits(net,head,mode,validation)
        correct=int((validation_score.argmax(1)==torch.from_numpy(labels[ix['validation']])).sum())
        loss_dev=float(torch.nn.functional.cross_entropy(validation_score,torch.from_numpy(labels[ix['validation']])))
        history.append(dict(mode=mode,seed=seed,lr=lr,epoch=epoch+1,mean_train_loss=total_loss/355,
                            train_correct=correct_train,validation_correct=correct,validation_loss=loss_dev,
                            updates=(epoch+1)*23,presentations=(epoch+1)*355,elapsed_seconds=time.time()-started))
        if correct>best:
            best=correct;best_epoch=epoch+1;besthead=head.detach().clone();bestnet=state_of(net,mode)
        print('train',mode,'seed',seed,'lr',lr,'epoch',epoch+1,'val',correct,'/355',
              'best',best,'loss',round(total_loss/355,5),'seconds',round(time.time()-started),flush=True)
        save_torch(resume,dict(protocol_sha256=signature,backbone=state_of(net,mode),head=head.detach(),
                              optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),next_epoch=epoch+1,
                              best=best,best_epoch=best_epoch,besthead=besthead,bestnet=bestnet,
                              history=history,elapsed_seconds=time.time()-started))
        csvout(Path(str(checkpoint)+'.history.csv'),history)
    record=dict(mode=mode,seed=seed,lr=lr,selected_epoch=best_epoch,validation_correct=best,
                epochs_executed=10,presentations=3550,updates=230,seconds=time.time()-started,
                checkpoint=checkpoint.name)
    save_torch(checkpoint,dict(protocol_sha256=signature,backbone=bestnet,head=besthead,
                               record=record,history=history))
    record['checkpoint_sha256']=filehash(checkpoint)
    jsonout(Path(str(checkpoint)+'.record.json'),record)
    print('COMPLETE RUN',json.dumps(record),flush=True)
    return record


def final_scores(selected,ix):
    base=np.load(OUT/'cache/resnet50_base.npy',mmap_mode='r')
    pixels=np.load(OUT/'cache/center_pixels.npy',mmap_mode='r')
    scorefile=OUT/'results/adaptation_scores.npz'
    scores=dict(np.load(scorefile)) if scorefile.exists() else {}
    for record in selected:
        mode,seed=record['mode'],record['seed']
        keys=[f'{mode}_{seed}_{role}' for role in ['validation','test']]
        if all(key in scores for key in keys):continue
        checkpoint=OUT/'checkpoints'/record['checkpoint'];saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        net,weight=load_model('resnet50');apply_state(net,saved['backbone'],mode);head=saved['head']
        for role in ['validation','test']:
            if mode=='linear':score=logits(net,head,'linear',base[ix[role]])
            else:
                # Complete network inference reconstructs the selected model for both partial and full.
                score=logits(net,head,'full',pixels[ix[role]])
            scores[f'{mode}_{seed}_{role}']=score.numpy()
            print('encoded selected checkpoint',mode,seed,role,score.shape,flush=True)
        temporary=Path(str(scorefile)+'.tmp')
        with temporary.open('wb') as handle:np.savez_compressed(handle,**scores)
        temporary.replace(scorefile)
    print('COMPLETE SELECTED CHECKPOINT INFERENCE',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-features',action='store_true')
    parser.add_argument('--max-run-seconds',type=int,default=7200)
    args=parser.parse_args();lock_protocol();rows,classes,ix,labels=splits()
    marker=OUT/'cache/resnet50.done.json';waiting=time.time()
    while not marker.exists():
        if not args.wait_for_features:raise RuntimeError('ResNet50 feature cache is not complete')
        if time.time()-waiting>7200:raise TimeoutError('Feature wait exceeded configured two-hour timeout')
        print('waiting for complete ResNet50/shared-view cache',flush=True);time.sleep(30)
    all_candidates=[];selected=[]
    for mode in ['linear','partial','full']:
        rates=[.001,.01] if mode=='linear' else [1e-5,1e-4]
        candidates=[run(mode,42,lr,ix,labels,args.max_run_seconds) for lr in rates]
        chosen=max(candidates,key=lambda row:row['validation_correct'])
        runs=[chosen]+[run(mode,seed,chosen['lr'],ix,labels,args.max_run_seconds) for seed in [43,44]]
        all_candidates.extend(candidates+runs[1:]);selected.extend(runs)
        jsonout(OUT/'results/adaptation_candidates.json',all_candidates)
        jsonout(OUT/'results/adaptation_selection.json',selected)
        csvout(OUT/'results/adaptation_candidates.csv',all_candidates)
    # All learning-rate and checkpoint choices are now fixed. Only now produce test scores.
    final_scores(selected,ix)
    jsonout(OUT/'results/adaptation.done.json',dict(status='completed',runs=12,selected=9,
        protocol_sha256=filehash(OUT/'protocol/experiment_protocol.json')))
    print('COMPLETE ADAPTATION',flush=True)


if __name__=='__main__':main()
