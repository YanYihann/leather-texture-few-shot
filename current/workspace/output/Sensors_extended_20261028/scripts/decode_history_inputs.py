"""Run in isolated libjpeg-turbo Pillow runtime; save decoded/cropped PNG controls."""
from pathlib import Path
from PIL import Image,features
import PIL,json,hashlib
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1];DATA=ROOT/'IC_project/01_核心代码与数据集';DEST=OUT/'cache/jpeg_turbo_inputs';DEST.mkdir(parents=True,exist_ok=True)
rows=[]
for split in ['dataset_train','dataset_val','my_dataset_test_filtered']:
    for p in sorted((DATA/split).glob('*/*')):
        if p.suffix.lower() not in ['.jpg','.jpeg']:continue
        im=Image.open(p).convert('RGB');w,h=im.size
        size=(int(256*w/h),256) if w>h else (256,int(256*h/w))
        im=im.resize(size,Image.Resampling.BILINEAR);w,h=im.size;left=round((w-224)/2);top=round((h-224)/2);im=im.crop((left,top,left+224,top+224))
        dst=DEST/split/(p.parent.name+'.png');dst.parent.mkdir(parents=True,exist_ok=True);im.save(dst)
        rows.append(dict(split=split,class_id=p.parent.name,input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest()))
(OUT/'audit/jpeg_turbo_runtime.json').write_text(json.dumps(dict(pillow=PIL.__version__,jpeg=features.version_codec('jpg'),turbo=features.version_feature('libjpeg_turbo'),inputs=rows),indent=2),encoding='utf8');print('decoded',len(rows),flush=True)
