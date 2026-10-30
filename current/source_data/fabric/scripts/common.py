from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import time
import numpy as np
import torch
from torchvision import models

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'T1k_dataset/fabric_images'
SEEDS = [42,43,44]
POOL_N = 30
torch.set_num_threads(6)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def jsonout(path, data):
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def csvout(path, rows):
    rows=list(rows)
    if not rows:
        return
    with path.open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def filehash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for value in iter(lambda:file.read(1048576),b''):
            h.update(value)
    return h.hexdigest()


def splits():
    rows=json.loads((OUT/'protocol/photo_splits.json').read_text(encoding='utf-8'))
    classes=sorted({row['class_name'] for row in rows})
    ix={role:np.array([row['index'] for row in rows if row['role']==role],dtype=np.int64)
        for role in ['reference','validation','test']}
    labels=np.array([row['label'] for row in rows],dtype=np.int64)
    assert all(labels[ix[role]].tolist()==list(range(355)) for role in ['reference','validation'])
    return rows,classes,ix,labels


def load_model(name):
    if name=='resnet50':
        weight=Path.home()/'.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth'
        net=models.resnet50(weights=None)
        net.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True))
        net.fc=torch.nn.Identity()
        net.to(memory_format=torch.channels_last)
    elif name=='dinov2':
        weight=Path.home()/'.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth'
        net=torch.hub.load(str(Path.home()/'.cache/torch/hub/facebookresearch_dinov2_main'),
                           'dinov2_vits14',source='local',pretrained=False)
        net.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True))
    elif name=='clip':
        weight=ROOT/'output/Sensors_extended_20261028/weights/ViT-B-32.pt'
        source=ROOT/'output/Sensors_extended_20261028/vendor/clip/model.py'
        spec=importlib.util.spec_from_file_location('t1k_official_clip_model',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with weight.open('rb') as file:
            jit=torch.jit.load(file,map_location='cpu').eval()
        net=module.build_model(jit.state_dict()).float()
        del jit
    else:
        raise ValueError(name)
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net,weight


def tensor(pixels,name='resnet50'):
    x=torch.from_numpy(np.asarray(pixels).copy()).permute(0,3,1,2).float().div_(255)
    mean=(.48145466,.4578275,.40821073) if name=='clip' else (.485,.456,.406)
    std=(.26862954,.26130258,.27577711) if name=='clip' else (.229,.224,.225)
    x.sub_(torch.tensor(mean)[None,:,None,None]).div_(torch.tensor(std)[None,:,None,None])
    return x.contiguous(memory_format=torch.channels_last) if name=='resnet50' else x.contiguous()


def resnet_layer3(net,x):
    return net.layer3(net.layer2(net.layer1(net.maxpool(net.relu(net.bn1(net.conv1(x)))))))


def normalize(x):
    x=np.asarray(x,dtype=np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def extract(net,x,name):
    f=net.encode_image(x) if name=='clip' else net(x)
    return torch.nn.functional.normalize(f.float(),dim=1)


def lock_protocol():
    rows,classes,ix,labels=splits()
    protocol=dict(
        version='T1K355_photo_disjoint_v1',date='2026-10-29',raw_photos=1792,classes=355,
        unique_file_photos=1787,duplicate_label_pair=['a452','a456'],
        duplicate_handling='Retain all official 355 labels for strict identity scoring; synchronize roles by decoded-photo hash; separately report clean-353 sensitivity. No byte/pixel duplicate crosses roles.',
        split={'reference':355,'validation':355,'test':1082},
        split_seed=20261029,split_unit='whole original acquisition; EXIF-transposed pixel hash; no patch split',
        spatial='EXIF transpose, RGB, Resize shorter side 256 bilinear, CenterCrop224; exact cached pixels shared by all primary descriptors',
        frozen=['RGB joint 8x8x8 sqrt histogram','gray 32-bin sqrt histogram','LBP 256-bin normalized','ImageNet1K V2 ResNet50','DINOv2 ViT-S/14','OpenAI CLIP ViT-B/32'],
        augmentation={'pool_seeds':SEEDS,'N':[1,5,10,20,30],'scale':[.4,1.0],'ratio':[.75,4/3],
                      'horizontal_vertical_flip':.5,'brightness_contrast_saturation':.2,'hue':0,
                      'seed':'SHA256(dataset|pool|source-decoded-pixel-hash|view); shared saved uint8 pixels',
                      'N1':'random augmented view, distinct from deterministic center baseline',
                      'prototype':'Normalize(mean of individually normalized views)',
                      'models':['resnet50','dinov2'],'query':'deterministic center pixels'},
        adaptation={'backbone':'ResNet50 ImageNet1K V2','modes':['linear','partial','full'],
                    'epochs':10,'batch':16,'presentations':3550,'updates':230,'seeds':SEEDS,
                    'rates':{'linear':[.001,.01],'partial':[1e-5,1e-4],'full':[1e-5,1e-4]},
                    'selection':'seed42 chooses rate and earliest best epoch by validation Top1 only; first rate on ties; seeds43,44 independently train chosen rate and select epoch using validation',
                    'optimizer':'Adam, weight_decay=1e-4; head LR10x backbone for partial/full; linear head LR as listed; constant schedule',
                    'head':'cosine scale20, initialized by frozen center reference embeddings',
                    'views':'one unique cached support augmentation per class per epoch (first10 of each seed pool)',
                    'BN':'fixed pretrained running statistics; affine parameters follow scope',
                    'partial':'layer4 + cosine head; exact frozen layer3 activations cached in float32 for speed',
                    'test_excluded_from_selection':True,'scope':'bounded budget, not fully tuned optimum'},
        additional={'fusion':'DINO+RGB and DINO+LBP, alpha grid [0,.1,.25,.5,.75,1] chosen only on validation',
                    'two_shot':'same 727 test queries for 1 vs 2 real references; move first primary-test photo to extra reference before comparing',
                    'reference_choice':'five predeclared alternative galleries for frozen center descriptors; overlapping descriptive sensitivity only',
                    'open_set':'282 known /35 calibration unknown /36 test unknown; exclude both duplicate-label classes; primitive frozen methods only; no adapted methods trained on unknown IDs'},
        metrics=['micro Recall@1/3/5','macro Recall@1','MRR (single relevant ID)','validation-selected checkpoint','per-query predictions'],
        uncertainty='2000 source-identity-cluster bootstrap draws; exact-copy IDs a452/a456 share one cluster (354 clusters for official355;353 for clean353); three fixed pool/training runs averaged within query before resampling; run SD separate; paired differences conditional on fixed models and photographs',
        scoring='all355 candidates; stable ascending-class tie break; no query-aware pruning',
        boundaries=['Fabric external instance retrieval, not leather replication','Repeated source photographs, not independently certified sessions/devices','Exact duplicate screening is not near-duplicate or capture-independence certification','No manuscript changes, uploads or submission'],
        source_threads=['01a18297-fbcb-75d1-950b-47ae4268a0e8','01a18298-1803-7e81-97ce-9383ad8dbc54'],
        photo_split_sha256=filehash(OUT/'protocol/photo_splits.json'))
    path=OUT/'protocol/experiment_protocol.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if old!=protocol:
            raise RuntimeError('Protocol already locked with different settings')
    else:
        jsonout(path,protocol)
    return protocol
