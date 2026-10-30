"""Official CLIP ViT-B/32 image-to-image cosine, no anonymous text prompts."""
from pathlib import Path
import torch,numpy as np,json,csv,urllib.request,hashlib,time,importlib.util
from PIL import Image
from torchvision import transforms
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1];DATA=ROOT/'IC_project/01_核心代码与数据集';SAVE=OUT/'new_experiments/clip';SAVE.mkdir(parents=True,exist_ok=True);VENDOR=OUT/'vendor/clip';VENDOR.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(3);torch.set_num_interop_threads(1)
SHA='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af';url=f'https://openaipublic.azureedge.net/clip/models/{SHA}/ViT-B-32.pt';weight=OUT/'weights/ViT-B-32.pt';weight.parent.mkdir(exist_ok=True)
def fetch(url,dst):
    if dst.exists():return
    print('download',url,flush=True)
    with urllib.request.urlopen(url,timeout=60) as response,dst.with_suffix(dst.suffix+'.partial').open('wb') as f:
        while True:
            chunk=response.read(1024*1024)
            if not chunk:break
            f.write(chunk)
    dst.with_suffix(dst.suffix+'.partial').replace(dst)
fetch(url,weight);assert hashlib.sha256(weight.read_bytes()).hexdigest()==SHA
# Pin the retrieved official source by its content hash in protocol.json.
fetch('https://raw.githubusercontent.com/openai/CLIP/main/clip/model.py',VENDOR/'model.py');fetch('https://raw.githubusercontent.com/openai/CLIP/main/LICENSE',VENDOR/'LICENSE')
spec=importlib.util.spec_from_file_location('official_clip_model',VENDOR/'model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with weight.open('rb') as weight_stream:jit=torch.jit.load(weight_stream,map_location='cpu').eval()
model=module.build_model(jit.state_dict()).float().eval();del jit
classes=sorted(p.name for p in (DATA/'dataset_train').iterdir() if p.is_dir());norm=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
spatial={'shared':transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224)]),'native':transforms.Compose([transforms.Resize(224,interpolation=transforms.InterpolationMode.BICUBIC),transforms.CenterCrop(224)])}
def encode(images,recipe):
    parts=[]
    with torch.inference_mode():
        for i in range(0,len(images),16):
            x=torch.stack([norm(spatial[recipe](im)) for im in images[i:i+16]]);f=model.encode_image(x);parts.append(torch.nn.functional.normalize(f,dim=1).numpy())
    return np.concatenate(parts)
def csvout(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=[];pred=[]
for recipe in spatial:
    fs={};ys={}
    for split in ['dataset_train','dataset_val','my_dataset_test_filtered']:
        paths=[next((DATA/split/c).glob('*')) for c in classes if (DATA/split/c).is_dir()];ys[split]=np.array([classes.index(p.parent.name) for p in paths]);cache=SAVE/f'{recipe}_{split}.npz'
        if cache.exists():fs[split]=np.load(cache)['features']
        else:
            ims=[Image.open(p).convert('RGB') for p in paths];t=time.time();fs[split]=encode(ims,recipe);np.savez_compressed(cache,features=fs[split],labels=ys[split],classes=np.array(classes));print('encoded',recipe,split,'seconds',round(time.time()-t),flush=True)
    for split in ['dataset_val','my_dataset_test_filtered']:
        score=fs[split]@fs['dataset_train'].T;order=np.argsort(-score,axis=1,kind='stable');y=ys[split];rank=(order==y[:,None]).argmax(1)+1
        np.savez_compressed(SAVE/f'{recipe}_{split}_scores.npz',scores=score,labels=y,classes=np.array(classes))
        rows.append(dict(method='CLIP ViT-B/32',recipe=recipe,split=split,n=len(y),top1_correct=int((rank==1).sum()),top3_correct=int((rank<=3).sum()),recall1=100*np.mean(rank==1),recall3=100*np.mean(rank<=3),recall5=100*np.mean(rank<=5),mrr=float(np.mean(1/rank))))
        pred.extend(dict(method='CLIP ViT-B/32',recipe=recipe,split=split,class_id=classes[yi],rank=int(ri),top1=classes[oi[0]],top2=classes[oi[1]],top3=classes[oi[2]]) for yi,ri,oi in zip(y,rank,order))
    csvout(SAVE/'results.csv',rows);csvout(SAVE/'predictions.csv',pred);print(rows[-2:],flush=True)
protocol=dict(model='OpenAI CLIP ViT-B/32',matching='normalized image embeddings; cosine nearest reference; full 203 library; no text or prompts',pretrained_weight_url=url,weight_sha256=SHA,official_source='https://github.com/openai/CLIP',source_sha256=hashlib.sha256((VENDOR/'model.py').read_bytes()).hexdigest(),recipes={'shared':'Resize(short=256,bilinear), CenterCrop224; CLIP normalization','native':'Resize(short=224,bicubic), CenterCrop224; CLIP normalization'},selection='both recipes prespecified and reported, no Q tuning',scope='clean S and Q sensitivity only; CLIP perturbation-grid and external experiments not executed in this script',device='cpu',precision='float32',torch=torch.__version__)
(SAVE/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8');print('COMPLETE CLIP',flush=True)
