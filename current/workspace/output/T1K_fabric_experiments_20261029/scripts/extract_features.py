"""Shared-pixel feature caches, reference pools and exact frozen-stage activations."""
from common import *
import argparse
import math
from PIL import Image,ImageOps,ImageEnhance
from torchvision.transforms import functional as TF, InterpolationMode


def array(path,shape,dtype):
    if path.exists():
        value=np.load(path,mmap_mode='r+')
        assert value.shape==shape and value.dtype==dtype,(path,value.shape,shape)
        return value
    return np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)


def random_view(image,seed):
    rng=np.random.default_rng(seed)
    w,h=image.size;area=w*h
    for _ in range(10):
        target=area*rng.uniform(.4,1.)
        ratio=math.exp(rng.uniform(math.log(.75),math.log(4/3)))
        cw,ch=round(math.sqrt(target*ratio)),round(math.sqrt(target/ratio))
        if 0<cw<=w and 0<ch<=h:
            top,left=int(rng.integers(0,h-ch+1)),int(rng.integers(0,w-cw+1))
            break
    else:
        ratio=w/h
        if ratio<.75:cw,ch=w,round(w/.75)
        elif ratio>4/3:cw,ch=round(h*4/3),h
        else:cw,ch=w,h
        top,left=(h-ch)//2,(w-cw)//2
    image=TF.resized_crop(image,top,left,ch,cw,[224,224],InterpolationMode.BILINEAR)
    if rng.random()<.5:image=TF.hflip(image)
    if rng.random()<.5:image=TF.vflip(image)
    factors=rng.uniform(.8,1.2,3)
    operations=[TF.adjust_brightness,TF.adjust_contrast,TF.adjust_saturation]
    for i in rng.permutation(3):image=operations[i](image,float(factors[i]))
    return np.asarray(image).copy()


def generate_views(rows,ix):
    shape=(3,355,30,224,224,3)
    path=OUT/'cache/aug_pixels.npy'
    done=OUT/'cache/aug_pixels.done.json'
    if done.exists():
        print('reuse augmented pixels',flush=True);return
    status=OUT/'cache/aug_pixels.progress.json'
    begin=json.loads(status.read_text()).get('next_class',0) if status.exists() else 0
    values=array(path,shape,np.dtype('uint8'))
    hashes=[]
    hashpath=OUT/'audit/augmented_pixel_hashes.csv'
    if hashpath.exists() and begin:
        with hashpath.open(encoding='utf-8-sig') as file:
            hashes=list(csv.DictReader(file))
    start=time.time()
    for c in range(begin,355):
        row=rows[int(ix['reference'][c])]
        with Image.open(DATA/row['relative_path']) as source:
            image=ImageOps.exif_transpose(source).convert('RGB');image.load()
        for si,seed in enumerate(SEEDS):
            for view in range(30):
                key=f'T1K355|{seed}|{row["decoded_pixel_sha256"]}|{view}'
                draw=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
                pixels=random_view(image,draw)
                values[si,c,view]=pixels
                hashes.append(dict(seed=seed,class_name=row['class_name'],view=view,
                                   source_pixel_sha256=row['decoded_pixel_sha256'],draw=draw,
                                   sha256=hashlib.sha256(pixels.tobytes()).hexdigest()))
        if (c+1)%10==0 or c==354:
            values.flush();csvout(hashpath,hashes);jsonout(status,dict(next_class=c+1))
            print('views classes',c+1,'/355 seconds',round(time.time()-start),flush=True)
    jsonout(done,dict(shape=shape,dtype='uint8',pool_seeds=SEEDS,source_split_sha256=filehash(OUT/'protocol/photo_splits.json')))


def handcrafted():
    dest=OUT/'cache/handcrafted.npz'
    if dest.exists():return
    pixels=np.load(OUT/'cache/center_pixels.npy',mmap_mode='r')
    rgb=[];gray=[];lbp=[]
    for image in pixels:
        a=image.astype(np.int32)//32
        h=np.bincount((a[:,:,0]*64+a[:,:,1]*8+a[:,:,2]).ravel(),minlength=512).astype(np.float32)
        rgb.append(np.sqrt(h/h.sum()))
        g=np.asarray(Image.fromarray(image).convert('L'))
        h=np.bincount((g.astype(np.int32)//8).ravel(),minlength=32).astype(np.float32)
        gray.append(np.sqrt(h/h.sum()))
        center=g[1:-1,1:-1];code=np.zeros_like(center,dtype=np.uint8)
        neighbors=[g[:-2,:-2],g[:-2,1:-1],g[:-2,2:],g[1:-1,2:],g[2:,2:],g[2:,1:-1],g[2:,:-2],g[1:-1,:-2]]
        for bit,neighbor in enumerate(neighbors):code|=(neighbor>=center).astype(np.uint8)<<(7-bit)
        h=np.bincount(code.ravel(),minlength=256).astype(np.float32)
        lbp.append(h/max(np.linalg.norm(h),1e-12))
    np.savez_compressed(dest,rgb=np.array(rgb),gray=np.array(gray),lbp=np.array(lbp))
    print('handcrafted complete',flush=True)


def encode_model(name,rows,ix,batch):
    base_path=OUT/f'cache/{name}_base.npy'
    aug_path=OUT/f'cache/{name}_aug.npy'
    marker=OUT/f'cache/{name}.done.json'
    if marker.exists():print('reuse completed',name,flush=True);return
    dimension={'resnet50':2048,'dinov2':384,'clip':512}[name]
    net,weight=load_model(name)
    info=dict(model=name,weight=str(weight),weight_sha256=filehash(weight),precision='float32',threads=6)
    base_pixels=np.load(OUT/'cache/center_pixels.npy',mmap_mode='r')
    base=array(base_path,(1792,dimension),np.dtype('float32'))
    progress_path=OUT/f'cache/{name}.progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'base_next':0,'aug_next':0}
    start=time.time()
    if name=='resnet50':
        dev3=array(OUT/'cache/resnet50_validation_layer3.npy',(355,1024,14,14),np.dtype('float32'))
        dev_lookup={int(index):slot for slot,index in enumerate(ix['validation'])}
    with torch.inference_mode():
        for pos in range(progress['base_next'],1792,batch):
            x=tensor(base_pixels[pos:pos+batch],name)
            if name=='resnet50':
                l3=resnet_layer3(net,x)
                # Prove cache factorization equals the ordinary encoder before using it for adaptation.
                f=net.avgpool(net.layer4(l3)).flatten(1)
                if pos==0:
                    direct=net(x);difference=float((direct-f).abs().max())
                    assert difference<1e-5,difference
                    info['layer3_factorization_max_abs_difference']=difference
                for i in range(len(x)):
                    if pos+i in dev_lookup:dev3[dev_lookup[pos+i]]=l3[i].numpy()
                f=torch.nn.functional.normalize(f,dim=1)
            else:f=extract(net,x,name)
            base[pos:pos+len(x)]=f.numpy()
            progress['base_next']=pos+len(x)
            if (pos//batch)%10==0 or progress['base_next']==1792:
                base.flush()
                if name=='resnet50':dev3.flush()
                jsonout(progress_path,progress)
                print(name,'base',progress['base_next'],'/1792 seconds',round(time.time()-start),flush=True)
        info['base_elapsed_seconds']=time.time()-start
        if name!='clip':
            augment=np.load(OUT/'cache/aug_pixels.npy',mmap_mode='r').reshape(-1,224,224,3)
            output=array(aug_path,(3,355,30,dimension),np.dtype('float32'))
            flattened=output.reshape(-1,dimension)
            if name=='resnet50':
                aug3=array(OUT/'cache/resnet50_aug_layer3.npy',(3,355,10,1024,14,14),np.dtype('float32'))
            start_aug=time.time()
            for pos in range(progress['aug_next'],len(augment),batch):
                x=tensor(augment[pos:pos+batch],name)
                if name=='resnet50':
                    l3=resnet_layer3(net,x);f=torch.nn.functional.normalize(net.avgpool(net.layer4(l3)).flatten(1),dim=1)
                    for i in range(len(x)):
                        si,cl,v=np.unravel_index(pos+i,(3,355,30))
                        if v<10:aug3[si,cl,v]=l3[i].numpy()
                else:f=extract(net,x,name)
                flattened[pos:pos+len(x)]=f.numpy()
                progress['aug_next']=pos+len(x)
                if (pos//batch)%10==0 or progress['aug_next']==len(augment):
                    output.flush()
                    if name=='resnet50':aug3.flush()
                    jsonout(progress_path,progress)
                    print(name,'aug',progress['aug_next'],'/',len(augment),'seconds',round(time.time()-start_aug),flush=True)
            info['aug_elapsed_seconds']=time.time()-start_aug
            info['aug_pool_count']=3
            info['aug_views_per_reference']=30
    # Warm loaded encoder timings use already decoded/spatially cached pixels, not full disk-to-query latency.
    x=tensor(base_pixels[:1],name)
    times=[]
    with torch.inference_mode():
        for i in range(35):
            start_call=time.perf_counter();feature=extract(net,x,name);feature@torch.from_numpy(base[ix['reference']].copy()).T
            if i>=10:times.append(1000*(time.perf_counter()-start_call))
    info.update(cached_pixels_to_cosine_median_ms=float(np.median(times)),cached_pixels_to_cosine_p95_ms=float(np.percentile(times,95)),
                timing_exclusions='file decoding, EXIF transform, spatial resize, weight loading and input tensor conversion',
                stored_single_prototype_float32_bytes=355*dimension*4)
    jsonout(marker,info)
    print('COMPLETE MODEL',name,json.dumps(info),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',default=['resnet50','dinov2','clip']);parser.add_argument('--batch',type=int,default=32)
    args=parser.parse_args();lock_protocol();rows,classes,ix,labels=splits()
    handcrafted()
    generate_views(rows,ix)
    for name in args.models:encode_model(name,rows,ix,args.batch)
    print('COMPLETE FEATURES',flush=True)


if __name__=='__main__':main()
