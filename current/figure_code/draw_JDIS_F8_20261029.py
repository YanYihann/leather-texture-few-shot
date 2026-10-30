"""Exact local F8 from archived timing records, after design-package study.
No model API/key, new measurement, training, inference or bootstrap.
"""
from pathlib import Path
import csv,json,hashlib,shutil,sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator,FuncFormatter,NullFormatter
from PIL import Image
import xml.etree.ElementTree as ET

ROOT=Path(r'D:/科研/IC_project');OUT=Path(__file__).resolve().parent
PACK=ROOT/'output/JDIS_main8_PaperBanana_prompts_20261029';L=PACK/'data/leather'
REV=ROOT/'output/JDIS_revision_20261029/source_data/leather/results'
ASTRA=ROOT/'output/Sensors_figures_Astra_20261027/data'
LEARN=ROOT/'output/journal_figure_design_20261027/顶刊图表设计学习包'
sources=[]
def preserve(p,rel):
    dest=OUT/'source_data'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if dest.exists():assert dest.read_bytes()==p.read_bytes(),str(p)+' changed after initial preservation'
        else:shutil.copy2(p,dest)
    assert dest.exists(),dest
    sources.append(dict(original_path=str(p),bundled_path=str(dest.relative_to(OUT)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    return dest
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(n,rows):
    with (OUT/(n+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def num(r,k):return float(r[k])
def pick(rows,**kw):
    rr=[r for r in rows if all(str(r[k])==str(v) for k,v in kw.items())];assert len(rr)==1,(kw,len(rr));return rr[0]
methods=['resnet50','dinov2_s14','rgb_histogram','gray_histogram']
labels=['ResNet50','DINOv2','RGB','Gray'];colors=['#3677A8','#C67B39','#568C76','#696E76'];marks=['o','s','^','D']
for m in methods:
    for n in [f'cost_query_{m}.csv',f'cost_registration_{m}.csv',f'cost_{m}.json']:
        p=preserve(L/n,Path('leather')/n)
        assert p.read_bytes()==(ASTRA/n).read_bytes()
        if (REV/n).exists():assert p.read_bytes()==(REV/n).read_bytes()
for n in ['clean_metrics.csv','clean_predictions.csv']:
    p=preserve(L/n,Path('leather')/n)
    assert p.read_bytes()==(REV/n).read_bytes()
preserve(ROOT/'output/Sensors_restructured_20261027/scripts/measure_cost.py',Path('owning_scripts/measure_cost.py'))
preserve(ROOT/'output/Sensors_restructured_20261027/scripts/run_computations.py',Path('owning_scripts/run_computations.py'))
preserve(ROOT/'output/Sensors_figures_Astra_20261027/scripts/draw_figures.py',Path('owning_scripts/previous_draw_figures.py'))
for n in ['content.txt','caption.txt','checklist.md']:
    preserve(PACK/'F8'/n,Path('input_contract')/n)
for n in ['00_使用说明.md','04_GPT学习资料/中文图表设计方法指南.md','04_GPT学习资料/给GPT的完整学习提示词.md','04_GPT学习资料/图表索引.csv']:
    preserve(LEARN/n,Path('design_guidance')/Path(n).name)
reference_paths=['02_高清图表/01_高水平期刊/J01_RADAM/J01_RADAM_Fig6_p10.png','02_高清图表/02_高度相关期刊补充/B01_VORTEX/B01_VORTEX_Fig8_p15.png','02_高清图表/01_高水平期刊/J02_DTPNet/J02_DTPNet_Table2_p13.png']
refs=[]
for rel in reference_paths:
    p=LEARN/rel
    if p.exists():refs.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),use='Viewed before F8 design; layout/encoding only, not experimental evidence'))
(OUT/'design_reference_manifest.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2),encoding='utf-8')
clean=read(OUT/'source_data/leather/clean_metrics.csv');pred=read(OUT/'source_data/leather/clean_predictions.csv')
prompt=(OUT/'source_data/input_contract/content.txt').read_text(encoding='utf-8')
online=[];offline=[];query_raw=[];rebuild_raw=[];metadata={};time_ids=[]
for m in methods:
    q=read(OUT/f'source_data/leather/cost_query_{m}.csv');r=read(OUT/f'source_data/leather/cost_registration_{m}.csv')
    meta=json.loads((OUT/f'source_data/leather/cost_{m}.json').read_text());metadata[m]=meta
    assert meta['cpu']=='Intel(R) Core(TM) Ultra 7 155H'
    assert (meta['threads'],meta['batch_query'],meta['batch_registration'],meta['warmup'],meta['repeats'],meta['registration_repeats'])==(3,1,16,10,100,3)
    assert 'decoded before timing' in meta['scope'] and 'no disk IO or weight load' in meta['scope']
    assert len(q)==100 and [int(x['repeat']) for x in q]==list(range(100)) and len({x['query'] for x in q})==100
    assert all(x['method']==m for x in q+r)
    time_ids.append([x['query'] for x in q]);t=np.array([num(x,'total_ms') for x in q])
    assert np.all(t>0)
    # Independently timed total, never reconstructed by adding component timings/quantiles.
    med=float(np.median(t));p95=float(np.percentile(t,95,method='linear'))
    cp=[x for x in pred if x['method']==m and x['variant']=='frozen' and x['protocol']=='clean' and x['condition']=='clean']
    assert len(cp)==201 and len({x['class_id'] for x in cp})==201
    hits=sum(x['top1']==x['class_id'] for x in cp)
    acc=num(pick(clean,method=m,variant='frozen'),'recall1');assert np.isclose(acc,100*hits/201,atol=1e-12)
    assert set(x['query'] for x in q)<=set(x['class_id'] for x in cp)
    out=dict(method=m,clean_recall1_pct=acc,clean_correct=hits,clean_queries=201,gallery_references=203,total_query_median_ms=med,total_query_p95_ms=p95,recorded_queries=100,warmups=10,batch_query=1,percentile_method='numpy linear',time_scope='decoded image, independently timed complete query; no loading/disk IO')
    line=next(s for s in prompt.splitlines() if s.startswith(f'| {m} |') and len(s.split('|'))==7)
    vals=[x.strip() for x in line.split('|')[1:-1]]
    assert np.allclose([acc,med,p95,100],[float(x) for x in vals[1:]],rtol=0,atol=6e-8)
    online.append(out);query_raw.extend(q)
    assert len(r)==9
    for v,setting in [(1,'Center'),(5,'A5'),(30,'A30')]:
        rr=[x for x in r if int(x['views'])==v]
        assert len(rr)==3 and {int(x['repeat']) for x in rr}=={0,1,2}
        assert all(int(x['encodings'])==203*v for x in rr)
        assert {x['reference_mode'] for x in rr}==({'deterministic_center'} if v==1 else {'seed42_random_nested'})
        seconds=np.array([num(x,'seconds') for x in rr]);assert np.all(seconds>0)
        out=dict(method=m,reference_setting=setting,encodings=203*v,rebuilds=3,batch_registration=16,seconds_median=float(np.median(seconds)),seconds_min=float(seconds.min()),seconds_max=float(seconds.max()),repeat0_seconds=float(seconds[0]),repeat1_seconds=float(seconds[1]),repeat2_seconds=float(seconds[2]),prototype_bytes=int(rr[0]['prototype_bytes']),reference_mode=rr[0]['reference_mode'])
        assert len({x['prototype_bytes'] for x in rr})==1
        line=next(s for s in prompt.splitlines() if s.startswith(f'| {m} | {setting} |'))
        vals=[x.strip() for x in line.split('|')[1:-1]]
        assert np.allclose([out[k] for k in ['seconds_median','seconds_min','seconds_max']],[float(x) for x in vals[2:5]],rtol=0,atol=6e-8)
        assert np.allclose(seconds,[float(x) for x in vals[5].split(',')],rtol=0,atol=6e-8)
        offline.append(out)
    assert len({x['prototype_bytes'] for x in r})==1
    rebuild_raw.extend(r)
assert all(x==time_ids[0] for x in time_ids)
assert [r['clean_correct'] for r in online]==[197,190,197,161]
assert np.isclose(pick(offline,method='dinov2_s14',reference_setting='A30')['seconds_max'],781.2903776,atol=1e-8)
write('F8_online4_median_p95_accuracy_plotted',online);write('F8_offline12_median_observed_ranges_plotted',offline)
write('F8_online400_independent_query_timings',query_raw);write('F8_offline36_actual_rebuilds',rebuild_raw)

W,H=180,150;INK='#202020'
style={'font.family':'Arial','font.size':8,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'svg.fonttype':'none','svg.hashsalt':'JDIS-F8-20261029','pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'}
with mpl.rc_context(style):
    fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=170)
    can=fig.add_axes([0,0,1,1]);can.set(xlim=(0,W),ylim=(0,H));can.axis('off')
    def tx(x,y,s,size=8,weight='normal',**kw):return can.text(x,y,s,fontsize=size,fontweight=weight,va='center',**kw)
    def axis(x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
    def basic(ax):
        ax.tick_params(labelsize=8,length=2.5,pad=2);ax.tick_params(which='minor',length=1.5)
        ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8)
        ax.grid(c='#E6E9EC',lw=.5,which='major',zorder=0)
    tx(4,144,'Online latency and offline construction have different costs',10.6,'bold')
    tx(4,138,'Intel Core Ultra 7 155H · CPU · 3 inference threads · already-decoded inputs',8)
    tx(4,133,'203-reference leather library · model loading and disk I/O excluded',8)
    tx(4,128,'Online: batch 1, 100 timings after 10 warmups · offline: batch 16',8)
    can.plot([4,176],[123.5,123.5],c='#B8C2CA',lw=.7)
    tx(4,118,'(a)  Online: accuracy–latency',10.2,'bold')
    tx(100,118,'(b)  Offline: full-library rebuild',10.2,'bold')
    tx(16,112,'Deterministic center · 201 clean Q',8)
    tx(112,112,'3 rebuilds: median and observed range',8)
    a=axis(16,48,68,58);b=axis(112,48,61,58)
    # Points encode median; horizontal one-sided segment ends at the empirical p95.
    offsets=[(-3,7),(0,-9),(3,7),(3,7)]
    for i,(m,label,co,mark) in enumerate(zip(methods,labels,colors,marks)):
        r=pick(online,method=m);x,y=r['total_query_median_ms'],r['clean_recall1_pct'];hi=r['total_query_p95_ms']
        a.errorbar(x,y,xerr=[[0],[hi-x]],fmt=mark,c=co,mfc=co,mec=INK,mew=.45,ms=4.7,lw=1.05,capsize=2.5,zorder=3)
        a.annotate(label+f'\n{x:.1f} ms',(x,y),xytext=offsets[i],textcoords='offset points',fontsize=8.2,ha='right' if i<2 else 'left',va='top' if i==1 else 'bottom',linespacing=1.15)
    a.set(xscale='log',xlim=(4,140),ylim=(76,103),yticks=[80,90,100],xlabel='Total-query latency (ms; log scale)',ylabel='Clean Q Recall@1 (%)')
    a.xaxis.set_major_locator(FixedLocator([5,10,20,50,100]));a.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'));a.xaxis.set_minor_formatter(NullFormatter());basic(a)
    for m,co,mark in zip(methods,colors,marks):
        rr=[pick(offline,method=m,reference_setting=s) for s in ['Center','A5','A30']]
        y=np.array([r['seconds_median'] for r in rr]);lo=np.array([r['seconds_min'] for r in rr]);hi=np.array([r['seconds_max'] for r in rr])
        b.errorbar([0,1,2],y,yerr=[y-lo,hi-y],fmt=mark+'-',c=co,mfc=co,mec=INK,mew=.4,ms=4,lw=1,capsize=2.5,zorder=3)
    b.set(yscale='log',xlim=(-.18,2.18),ylim=(.4,1200),xticks=[0,1,2],xticklabels=['Center','A5','A30'],xlabel='Reference construction (categorical)',ylabel='Rebuild time (s; log scale)')
    b.yaxis.set_major_locator(FixedLocator([.5,1,10,100,1000]));b.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'));b.yaxis.set_minor_formatter(NullFormatter());basic(b)
    slow=pick(offline,method='dinov2_s14',reference_setting='A30')
    b.annotate(f'max {slow["seconds_max"]:.1f} s',(2,slow['seconds_max']),xytext=(-5,4),textcoords='offset points',ha='right',va='bottom',fontsize=8)
    can.plot([17,25],[31,31],color='#41484D',lw=1)
    can.plot([25,25],[30.2,31.8],color='#41484D',lw=.7)
    can.plot(17,31,'o',c='#41484D',ms=4)
    tx(29,31,'Median point → p95 cap',8)
    handles=[Line2D([],[],c=c,marker=m,ls='-',lw=1,ms=3.6,label=l) for c,m,l in zip(colors,marks,labels)]
    can.legend(handles=handles,loc='center',bbox_to_anchor=(140/W,30/H),frameon=False,fontsize=8,ncol=2,columnspacing=1,handletextpad=.5,handlelength=1.5,labelspacing=.5,borderaxespad=0)
    tx(4,21,'Center: 203 inputs; A5: 1,015; A30: 6,090. One float32 prototype per identity in every setting.',8)
    tx(4,15,'Histogram A5/A30 measure construction cost only; no corresponding new accuracy baseline.',8)
    tx(4,9,'Query totals are independently timed. Median→p95 and observed ranges are not confidence intervals.',8)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();overflow=[]
    texts=[t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text()]
    for t in texts:
        z=t.get_window_extent(renderer)
        if z.x0<-.5 or z.y0<-.5 or z.x1>fig.bbox.width+.5 or z.y1>fig.bbox.height+.5:overflow.append(dict(text=t.get_text(),bbox=list(z.bounds)))
    assert not overflow,overflow
    assert min(t.get_fontsize() for t in texts)>=8
    assert a.get_xscale()=='log' and b.get_yscale()=='log' and b.get_xscale()=='linear'
    assert np.allclose(np.diff(b.transData.transform([(i,1) for i in range(3)])[:,0]),np.diff(b.transData.transform([(i,1) for i in range(3)])[:,0])[0])
    assert all(a.get_xlim()[0]<r['total_query_median_ms']<=r['total_query_p95_ms']<a.get_xlim()[1] and a.get_ylim()[0]<r['clean_recall1_pct']<a.get_ylim()[1] for r in online)
    assert all(b.get_ylim()[0]<r['seconds_min']<=r['seconds_median']<=r['seconds_max']<b.get_ylim()[1] for r in offline)
    stem=OUT/'F8_measured_online_and_offline_cost'
    for ext in ['svg','pdf','png']:fig.savefig(stem.with_suffix('.'+ext),dpi=600,bbox_inches=None,facecolor='white')
    fig.savefig(OUT/'F8_preview.png',dpi=180,bbox_inches=None,facecolor='white')
    Image.open(OUT/'F8_preview.png').convert('L').save(OUT/'F8_grayscale_QA.png')
    plt.close(fig)
svg=ET.parse(stem.with_suffix('.svg'));ns={'s':'http://www.w3.org/2000/svg'}
assert not svg.findall('.//s:image',ns)
im=Image.open(stem.with_suffix('.png'))
manifest=dict(date='2026-10-29',figure='F8',physical_mm=[W,H],PNG_pixels=list(im.size),PNG_dpi=list(im.info['dpi']),SVG_images=0,SVG_text_elements=len(svg.findall('.//s:text',ns)),source_files=sources,learning_package_studied_before_drawing=True,new_reference_figures_viewed=len(refs),online_method_points=4,online_raw_timings=400,online_summary='median to empirical p95; numpy linear interpolation; no CI or SD',timed_query_subset=100,clean_accuracy_queries=201,gallery_references=203,warmups=10,query_batch=1,registration_batch=16,offline_method_setting_points=12,offline_observations=36,offline_summary='median and observed min/max of 3 actual rebuilds',offline_settings=['deterministic Center','A5','A30'],offline_x_categorical_evenly_spaced=True,online_x_log=True,offline_y_log=True,cpu='Intel Core Ultra 7 155H',inference_threads=3,decoded_inputs_only=True,model_load_disk_io_excluded=True,independently_measured_total=True,DINOv2_A30_max_seconds=slow['seconds_max'],minimum_font_pt=8,text_overflow=overflow,python=sys.executable,matplotlib=mpl.__version__,numpy=np.__version__,model_API_calls=0,API_keys_used=False,new_measurement_training_inference_bootstrap=False,manuscript_overwritten=False)
(OUT/'validation_and_provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
caption='Measured online and offline cost for the original 203-reference leather library on an Intel Core Ultra 7 155H CPU with three inference threads. Images were decoded before timing; disk I/O and model loading are excluded. (a) Current deterministic-center clean Recall@1 on the same 201 Q identities and 203 candidates versus independently measured complete query latency. Marker positions are medians of 100 recorded batch-one query times after ten warmups; one-sided horizontal segments extend from median to the empirical 95th percentile, using NumPy linear interpolation. They are not confidence intervals or standard deviations. Complete totals include preprocessing, feature extraction and full-library matching/stable Top-5 ranking, timed independently rather than obtained by summing component medians or p95s. The 100 timed queries are a fixed subset, while accuracy uses all 201 Q identities. RGB and ResNet50 each identify 197/201; equal hit counts do not establish equivalence. (b) Three categorical, evenly spaced reference-construction settings: deterministic Center, A5 and A30. Markers are medians and whiskers span the observed minimum–maximum of three actual full-library rebuilds per method/setting; all 36 measurements accompany the figure, including the slower DINOv2 A30 rebuild (781.3 s). Registration includes view generation, normalization, batch-16 encoding, aggregation and final prototype formation. Center processes 203 inputs, A5 1015 and A30 6090; each stores one float32 prototype per identity. Histogram A5/A30 measure construction cost only and do not supply new accuracy baselines. The latency x-axis and rebuild-time y-axis are logarithmic; the construction x-axis is categorical. First post-load calls, weight-file size, prototype payload and whole-process peak working set are separate archived measurements, not alternative totals in these panels. These shared-desktop observations do not measure Fabric, CLIP, fusion or adapted-model cost, production throughput, hardware-independent latency or human labor savings.'
(OUT/'F8_caption.txt').write_text(caption,encoding='utf-8')
(OUT/'F8_alt_text.txt').write_text('Two panels separate online query cost from offline full-library construction. For 201 clean leather queries and 203 references, ResNet50 and RGB each reach 98.01% Recall@1, DINOv2 94.53% and Gray 80.10%. Independently timed median query latencies are 57.8, 6.7, 99.4 and 5.1 ms respectively; each point extends to empirical p95 on a logarithmic latency axis. Offline logarithmic seconds show three actual rebuilds summarized by median and observed range at equally spaced Center/A5/A30 settings. The DINOv2 A30 maximum of 781.3 seconds is retained. All timings use the recorded Intel Core Ultra 7 155H CPU with three threads and decoded inputs; model loading and disk I/O are excluded.',encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k!='source_files'},ensure_ascii=False,indent=2))
