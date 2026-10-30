"""Figure 5: exact query-level ECDFs and the complete real-photo error plate.
Local code only. No model generation, inference, training, or API calls.
"""
from pathlib import Path
import csv,json,hashlib,shutil,statistics as st,math,sys,base64,io
import numpy as np
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle
from matplotlib.lines import Line2D
import xml.etree.ElementTree as ET

ROOT=Path(r'D:/科研/IC_project');REV=ROOT/'output/JDIS_revision_20261029'
PACK=ROOT/'output/JDIS_main8_PaperBanana_prompts_20261029';ASSETS=PACK/'assets'
DATA=REV/'source_data/leather/results';OUT=ROOT/'output/JDIS_F5_local_20261029';OUT.mkdir(exist_ok=True)
def read(n):
    with (DATA/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def writecsv(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
qrows=read('query_stability.csv');pred=read('clean_predictions.csv')
fail=sorted([r for r in pred if r['method']=='resnet50' and r['variant']=='frozen' and int(r['rank'])>1],key=lambda r:int(r['class_id'].split('_')[1]))
assert [r['class_id'] for r in fail]==['Leather_26','Leather_33','Leather_48','Leather_182']
assert [r['wrong_class'] for r in fail]==['Leather_165','Leather_102','Leather_29','Leather_18']
assert len([r for r in pred if r['method']=='resnet50' and r['variant']=='frozen'])==201
for r in fail:
    assert float(r['true_margin'])<0 and math.isclose(float(r['true_score'])-float(r['wrong_score']),float(r['true_margin']),abs_tol=1e-12)
prompt=(PACK/'F5/content.txt').read_text(encoding='utf-8')
prompt_ecdf=json.loads(prompt.split('```json\n',1)[1].split('\n```',1)[0])

methods=['resnet50','dinov2_s14','rgb_histogram','gray_histogram']
labels=['ResNet50','DINOv2','RGB','Gray'];colors=['#3677A8','#C67B39','#568C76','#696E76']
linestyles=['-','--','-.',':'];ecdfs={};units=[];step_records=[];float_reconciliation=[]
for m in methods:
    for variant in (['frozen','A30'] if m in methods[:2] else ['frozen']):
        rr=[r for r in qrows if r['method']==m and r['variant']==variant]
        ids=sorted({r['class_id'] for r in rr},key=lambda c:int(c.split('_')[1]));assert len(ids)==201
        values=[]
        for c in ids:
            cr=[r for r in rr if r['class_id']==c]
            assert len(cr)==(3 if variant=='A30' else 1)
            assert {r['seed'] for r in cr}==({'42','43','44'} if variant=='A30' else {'fixed'})
            # Every within-pool fraction is an integer count out of 36.
            # Recover this known count to merge mathematically tied ECDF steps;
            # do not allow floating-point addition order to split equal fractions.
            counts=[round(float(r['correct_fraction'])*36) for r in cr]
            assert all(math.isclose(float(r['correct_fraction']),n/36,abs_tol=1e-14) for r,n in zip(cr,counts))
            v=sum(counts)/(36*len(counts));legacy=st.mean(float(r['correct_fraction']) for r in cr)
            assert math.isclose(v,legacy,abs_tol=2e-16)
            values.append(v)
            units.append(dict(method=m,variant=variant,class_id=c,reference_pools=len(cr),correct_fraction=v,correct_descendant_sum=sum(counts),descendant_denominator_for_fraction=36*len(cr),independent_query_unit=1))
            float_reconciliation.append(abs(v-legacy))
        unique,counts=np.unique(values,return_counts=True);cum=np.cumsum(counts)
        ecdfs[(m,variant)]=(unique,cum/201)
        for x,n in zip(unique,cum):step_records.append(dict(method=m,variant=variant,correct_fraction=float(x),cumulative_identities=int(n),ECDF=float(n/201)))
        assert cum[-1]==201 and np.all(np.diff(cum)>0)
        # Existing prompt ECDF and the new exact-tie representation agree away
        # from machine-epsilon neighborhoods of the same theoretical jumps.
        old=prompt_ecdf[m+'/'+variant]
        for x,y in zip(unique,cum/201):
            matches=[p for p in old if abs(p['x']-x)<1e-14]
            assert matches and math.isclose(max(p['ecdf'] for p in matches),float(y),abs_tol=1e-14)
writecsv('F5_query_identity_units_plotted.csv',units)
writecsv('F5_ECDF_exact_steps.csv',step_records)
writecsv('F5_complete_clean_failures.csv',fail)

# Construct the same PIL spatial input transform documented for the leather
# figure: RGB conversion, shorter-side Resize256 bilinear, CenterCrop224.
# The original files and their backgrounds are retained separately.
manifest=json.loads((ASSETS/'manifest.json').read_text(encoding='utf-8'))
photo_audit=[];images={};cropdir=OUT/'model_input_crops';cropdir.mkdir(exist_ok=True)
originaldir=OUT/'original_assets';originaldir.mkdir(exist_ok=True)
def make_input(p):
    im=Image.open(p).convert('RGB');w,h=im.size
    if w<=h:nw,nh=256,int(256*h/w)
    else:nw,nh=int(256*w/h),256
    resized=im.resize((nw,nh),resample=Image.Resampling.BILINEAR)
    left=int(round((nw-224)/2));top=int(round((nh-224)/2))
    return resized.crop((left,top,left+224,top+224)),{'original_size':[w,h],'resized_size':[nw,nh],'crop_rectangle':[left,top,left+224,top+224],'interpolation':'PIL bilinear','input_mode':'RGB'}
for row in fail:
    c=row['class_id'];w=row['wrong_class']
    requests=[('query',c,'F5_query'),('true',c,'F5_true'),('wrong',w,'F5_wrong_for_'+c)]
    for col,identity,role in requests:
        found=[r for r in manifest if r['role']==role and r['ID']==identity];assert len(found)==1
        rec=found[0];src=PACK/rec['packaged'];assert sha(src)==rec['sha256']
        assert sha(Path(rec['source']))==rec['sha256']
        shutil.copy2(src,originaldir/src.name)
        crop,info=make_input(src);outfile=cropdir/f'{c}_{col}_{identity}_224.png';crop.save(outfile)
        assert crop.size==(224,224);images[(c,col)]=np.array(crop)
        photo_audit.append(dict(query_id=c,column=col,displayed_id=identity,original=str(src),original_sha256=sha(src),original_source=rec['source'],crop_file=str(outfile),crop_sha256=sha(outfile),**info))
assert len(photo_audit)==12
(OUT/'photo_preprocessing_audit.json').write_text(json.dumps(photo_audit,ensure_ascii=False,indent=2),encoding='utf-8')

W,H=180,158;INK='#202020'
style={'font.family':'Arial','font.size':8,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
       'svg.fonttype':'none','svg.hashsalt':'JDIS-F5-20261029','pdf.fonttype':42,'ps.fonttype':42,
       'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,'axes.spines.top':False,'axes.spines.right':False,
       'figure.facecolor':'white','savefig.facecolor':'white'}
with mpl.rc_context(style):
    fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=150)
    can=fig.add_axes([0,0,1,1]);can.set(xlim=(0,W),ylim=(0,H));can.axis('off')
    def tx(x,y,s,size=8,weight='normal',**kw):return can.text(x,y,s,fontsize=size,fontweight=weight,va='center',**kw)
    def axis(x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
    tx(4,151,'(a)  Query consistency',10.8,'bold')
    tx(101,151,'(b)  Every clean ResNet50 error',10.8,'bold')
    tx(4,144.5,'201 original Q identities per curve',8.5)
    tx(101,144.5,'All four errors; fixed numeric ID order',8.5)
    ax1=axis(15,91,32,40);ax2=axis(61,91,32,40)
    for ax,variant in [(ax1,'frozen'),(ax2,'A30')]:
        for i,m in enumerate(methods if variant=='frozen' else methods[:2]):
            x,y=ecdfs[(m,variant)]
            ax.step(np.r_[0.,x,1.],np.r_[0.,y,1.],where='post',c=colors[i],ls=linestyles[i],lw=1.15,label=labels[i],clip_on=False)
        ax.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.25,.5,.75,1],xlabel='Correct fraction')
        ax.tick_params(labelsize=8,length=2.5,pad=2);ax.grid(c='#E6E9EC',lw=.5)
        ax.xaxis.label.set_size(8)
    ax1.set_ylabel('ECDF of Q identities',fontsize=8);ax2.tick_params(labelleft=False)
    tx(31,135.5,'Center',9,'bold',ha='center')
    tx(77,135.5,'A30: pool mean',9,'bold',ha='center')
    handles=[Line2D([],[],c=colors[i],ls=linestyles[i],lw=1.3,label=labels[i]) for i in range(4)]
    can.legend(handles=handles,loc='center',bbox_to_anchor=(48/W,73/H),ncol=2,frameon=False,fontsize=8.5,
               handlelength=2.2,columnspacing=1.7,labelspacing=.65,handletextpad=.5)
    tx(5,59,'Per identity: 6 B families × 2 levels × 3 repeats',8)
    tx(5,53.8,'A30: average three pool fractions within identity',8)
    can.add_patch(FancyBboxPatch((5,19),88,28,boxstyle='round,pad=0,rounding_size=1.3',facecolor='#EAF2F8',edgecolor='#A9B7C2',linewidth=.65))
    tx(8,41.5,'Reading the ECDF',9,'bold')
    tx(8,35.5,'At the same correctness fraction x,',8.3)
    tx(8,30,'a lower curve means more identities exceed x.',8.3)
    tx(8,23.8,'Digital descendants are not new captures.',8.3)
    tx(5,12,'Leather only; no Fabric failure inference.',8)

    xpos=[102,127,152];size=21
    headers=['Query','True reference','Strongest wrong\nreference']
    for x,label in zip(xpos,headers):tx(x+size/2,138,label,8.3,'bold',ha='center')
    imaxes=[];margin_labels=[]
    for i,r in enumerate(fail):
        c=r['class_id'];cid=int(c.split('_')[1]);wid=int(r['wrong_class'].split('_')[1]);y=111-i*31
        for j,col in enumerate(['query','true','wrong']):
            ax=axis(xpos[j],y,size,size);ax.imshow(images[(c,col)],interpolation='none',aspect='equal',rasterized=False)
            ax.set(xticks=[],yticks=[])
            for sp in ax.spines.values():sp.set_visible(True);sp.set_linewidth(.7);sp.set_edgecolor(['#696E76','#568C76','#AD5E57'][j])
            imaxes.append(ax)
        tx(xpos[0]+size/2,y-3.2,f'ID {cid}',8,ha='center')
        tx(xpos[1]+size/2,y-3.2,f"ID {cid} · {float(r['true_score']):.3f}",8,ha='center')
        tx(xpos[2]+size/2,y-3.2,f"ID {wid} · {float(r['wrong_score']):.3f}",8,ha='center')
        margin_labels.append(tx(102,y-8.1,f"Δ = {float(r['true_margin']):+.3f}; true rank = {r['rank']}",8,color=INK))
    tx(4,2.4,'Scores and margins are rounded independently. Photos use the common input crop without appearance edits.',8)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();overflow=[]
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5:overflow.append({'text':t.get_text(),'bbox':list(b.bounds)})
    assert not overflow,overflow
    for i in range(3):
        assert margin_labels[i].get_window_extent(renderer).y0>imaxes[(i+1)*3].get_window_extent(renderer).y1+.5
    # Every photo frame is square; all 12 crops are exactly 224×224 RGB.
    assert len(imaxes)==12
    assert all(abs(a.get_window_extent(renderer).width/a.get_window_extent(renderer).height-1)<1e-10 for a in imaxes)
    stem=OUT/'F5_query_consistency_and_complete_failures'
    for ext in ['svg','pdf','png']:fig.savefig(stem.with_suffix('.'+ext),dpi=600,facecolor='white',bbox_inches=None)
    fig.savefig(OUT/'F5_preview.png',dpi=185,facecolor='white',bbox_inches=None)
    Image.open(OUT/'F5_preview.png').convert('L').save(OUT/'F5_grayscale_QA.png')
    plt.close(fig)

ns={'s':'http://www.w3.org/2000/svg'};svg=ET.parse(stem.with_suffix('.svg'))
assert len(svg.findall('.//s:image',ns))==12
embedded=svg.findall('.//s:image',ns)
expected=[images[(r['class_id'],col)] for r in fail for col in ['query','true','wrong']]
for element,pixels in zip(embedded,expected):
    uri=element.attrib['{http://www.w3.org/1999/xlink}href']
    decoded=Image.open(io.BytesIO(base64.b64decode(uri.split(',',1)[1]))).convert('RGB')
    assert decoded.size==(224,224) and np.array_equal(np.asarray(decoded),pixels)
assert len(svg.findall('.//s:text',ns))>=50
png=Image.open(stem.with_suffix('.png'))
source_paths=[DATA/'query_stability.csv',DATA/'clean_predictions.csv',ASSETS/'manifest.json']
for n in ['query_stability.csv','clean_predictions.csv']:
    (OUT/'source_data').mkdir(exist_ok=True);shutil.copy2(DATA/n,OUT/'source_data'/n)
report={'date':'2026-10-29','figure':'F5','physical_mm':[W,H],'PNG_pixels':list(png.size),'PNG_dpi':list(png.info.get('dpi',[])),
        'ECDF_series':6,'original_query_units_per_series':201,'digital_descendants_per_identity_per_pool':36,
        'A30_pools':[42,43,44],'A30_aggregation':'per-identity average of three within-pool correct fractions, then ECDF',
        'exact_integer_count_ties_preserved':True,'max_fraction_float_reconciliation':max(float_reconciliation),
        'ECDF_steps_cross_checked_against_prompt':True,'complete_failure_IDs':[26,33,48,182],'wrong_IDs':[165,102,29,18],
        'genuine_original_photo_inputs':12,'crop_size':[224,224],'crop_normalization':'RGB; shorter-side256 PIL bilinear; CenterCrop224; no channel normalization for display',
        'photo_decoder_scope':'Original assets verified by SHA256; spatial crops reconstructed using documented PIL processing. No bytewise equality claim against an unavailable historical decoded-tensor cache.',
        'SVG_photo_images':12,'SVG_photo_pixel_identity_verified':True,'SVG_photo_native_pixels':[224,224],'SVG_text_elements':len(svg.findall('.//s:text',ns)),'SVG_editability':'ECDF, axes and text are vector; twelve actual photos are embedded rasters.',
        'text_overflow':overflow,'model_API_calls':0,'API_keys_used':False,'new_training_or_inference':False,'source_files':[{'path':str(p),'sha256':sha(p)} for p in source_paths],
        'python':sys.executable,'matplotlib':mpl.__version__,'Pillow':Image.__version__,'manuscript_overwritten':False}
(OUT/'validation_and_provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
caption='Query consistency and the complete current clean ResNet50 failure set on leather. (a) Center and A30 ECDFs use all 201 original Q identities, with correctness assessed over 36 Protocol B descendants per identity. For A30, the three reference-pool fractions are averaged within identity before forming the ECDF. No smoothing is applied; mathematically equal count-based fractions share a step. At any fixed x, a lower ECDF means more identities exceed that correctness fraction. The descendant inputs are not new acquisitions. (b) Every deterministic frozen ResNet50 clean-Q Top-1 error, sorted by numeric identity: 26, 33, 48 and 182, against the complete 203-reference gallery. Each row shows the query, assigned true reference and strongest wrong reference, with cosine scores, true-minus-wrong margin Δ and true-reference rank. Scores and margins are rounded independently. Images are reconstructed from verified original assets using the common RGB Resize256-bilinear/CenterCrop224 spatial preprocessing, without appearance edits. The twelve originals, display crops and preprocessing records are retained. Failure appearance does not establish a causal effect of color, grain, background or acquisition and is not evidence of Fabric failure mechanisms.'
(OUT/'F5_caption.txt').write_text(caption,encoding='utf-8')
(OUT/'F5_alt_text.txt').write_text('Two ECDF panels compare center references and A30 over the same 201 leather query identities. Lower ECDF at a given correctness fraction indicates more identities exceed it. A30 uses the within-identity mean of three pools. A twelve-photo plate lists all four frozen ResNet50 clean errors: ID26 retrieved165, ID33 retrieved102, ID48 retrieved29 and ID182 retrieved18. True margins are −0.017, −0.119, −0.045 and −0.004 respectively; the true ranks are 2,11,2,2. All photographs are real study assets.',encoding='utf-8')
shutil.copy2(Path(__file__),OUT/Path(__file__).name)
print(json.dumps({k:v for k,v in report.items() if k not in ['source_files']},ensure_ascii=False,indent=2))
