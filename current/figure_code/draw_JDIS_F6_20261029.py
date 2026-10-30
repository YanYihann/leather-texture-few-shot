"""Exact local Figure 6. No model API, training, inference or bootstrap.
Rebuild from bundled source_data with the local PaperBanana Python environment.
"""
from pathlib import Path
import csv, json, hashlib, shutil, sys, statistics as st
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

ROOT=Path(r'D:/科研/IC_project')
OUT=Path(__file__).resolve().parent
PACK=ROOT/'output/JDIS_main8_PaperBanana_prompts_20261029'
FAB=ROOT/'output/JDIS_revision_20261029/source_data/fabric'
OLD=ROOT/'output/Sensors_figures_Astra_20261027'
SOURCES=[]
def preserve(p, rel):
    dest=OUT/'source_data'/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if dest.exists(): assert p.read_bytes()==dest.read_bytes(),str(p)+' changed since delivery'
        else: shutil.copy2(p,dest)
    assert dest.exists(),dest
    SOURCES.append(dict(original_path=str(p),bundled_path=str(dest.relative_to(OUT)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    return dest
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(n, rows):
    with (OUT/(n+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def pick(rows,**kw):
    rr=[r for r in rows if all(str(r[k])==str(v) for k,v in kw.items())]
    assert len(rr)==1,(kw,len(rr))
    return rr[0]
def num(r,k):return float(r[k])
for n in ['main_metrics.csv','per_run_metrics.csv','paired_recall1_differences.csv','additional_descriptive_paired_contrasts.csv','fusion_selection.json','adaptation_selection.json','adaptation_candidates.csv','evaluation_complete.json']:
    preserve(FAB/'results'/n,Path('fabric/results')/n)
for n in ['common.py','evaluate_results.py','build_final_report.py']:
    preserve(FAB/'scripts'/n,Path('fabric/owning_scripts')/n)
for n in ['matsim_results.csv','dtd_results.csv','external_paired_plot_data.csv']:
    preserve(OLD/'data'/n,Path('external')/n)
for n in ['draw_figures.py','assemble_delivery.py']:
    preserve(OLD/'scripts'/n,Path('external/owning_scripts')/n)
for n in ['matsim_evaluation.py','additional_experiments.py']:
    preserve(ROOT/'output/Sensors_revision_20261011/scripts'/n,Path('external/owning_scripts')/n)
for n in ['content.txt','caption.txt','checklist.md','data_payload.md']:
    preserve(PACK/'F6'/n,Path('input_contract')/n)
fm=read(OUT/'source_data/fabric/results/main_metrics.csv')
fp=read(OUT/'source_data/fabric/results/paired_recall1_differences.csv')
extra=read(OUT/'source_data/fabric/results/additional_descriptive_paired_contrasts.csv')
ma=read(OUT/'source_data/external/matsim_results.csv');dt=read(OUT/'source_data/external/dtd_results.csv')
oldpairs=read(OUT/'source_data/external/external_paired_plot_data.csv')
prompt=(OUT/'source_data/input_contract/content.txt').read_text(encoding='utf-8')
methods=['rgb_center','gray_center','lbp_center','resnet50_center','dinov2_center','clip_center','dinov2_A30','dinov2_rgb_fusion','dinov2_lbp_fusion','resnet50_linear','resnet50_partial','resnet50_full']
labels=['RGB','Gray','LBP','ResNet50','DINOv2','CLIP','DINOv2 A30','DINOv2 + RGB (α=.25)','DINOv2 + LBP (α=.10)','ResNet50 linear','ResNet50 last-stage','ResNet50 full']
colors=['#568C76','#696E76','#8C6D46','#3677A8','#C67B39','#887CA7','#C67B39','#C67B39','#C67B39','#3677A8','#3677A8','#3677A8']
markers=['^','D','v','o','s','P','s','^','v','D','s','o']
assert len(fm)==42 and len({r['method'] for r in fm})==21
selected=[]
for study,nq,nc,nu in [('primary355',1082,354,1079),('clean353',1076,353,1076)]:
    for m in methods:
        r=pick(fm,study=study,method=m)
        assert (int(r['n_queries']),int(r['bootstrap_clusters']),int(r['unique_query_source_photos']))==(nq,nc,nu)
        selected.append(r)
        line=next(s for s in prompt.splitlines() if s.startswith(f'| {study} | {m} |'))
        q=[v.strip() for v in line.split('|')[1:-1]]
        assert np.allclose([float(x) for x in q[4:8]],[num(r,k) for k in ['recall1','recall1_ci_low','recall1_ci_high','recall1_run_sd']],atol=6e-8,rtol=0)
assert round(num(pick(fm,study='primary355',method='clip_center'),'recall1'),2)==92.70
fs=json.loads((OUT/'source_data/fabric/results/fusion_selection.json').read_text())
assert {r['component']:r['alpha'] for r in fs}=={'rgb':.25,'lbp':.1}
contrast_spec=[('dinov2_center','rgb_center','DINOv2 − RGB'),('dinov2_center','resnet50_center','DINOv2 − ResNet50'),('dinov2_rgb_fusion','dinov2_center','DINOv2 + RGB − DINOv2'),('resnet50_full','resnet50_partial','ResNet50 full − last-stage'),('resnet50_linear','resnet50_center','ResNet50 linear − frozen'),('resnet50_partial','resnet50_center','ResNet50 last-stage − frozen'),('resnet50_full','resnet50_center','ResNet50 full − frozen')]
contrasts=[]
for t,b,lab in contrast_spec:
    rr=[r for r in fp if r['study']=='primary355' and r['treatment']==t and r['baseline']==b]
    r=rr[0] if rr else pick(extra,treatment=t,baseline=b)
    delta=num(r,'recall1_difference_pp' if 'recall1_difference_pp' in r else 'difference_pp')
    assert np.isclose(delta,num(pick(fm,study='primary355',method=t),'recall1')-num(pick(fm,study='primary355',method=b),'recall1'),atol=1e-10)
    contrasts.append(dict(treatment=t,baseline=b,label=lab,difference_pp=delta,ci_low_pp=num(r,'ci_low_pp'),ci_high_pp=num(r,'ci_high_pp'),n_queries=1082,source_clusters=354))
for i in [1,3]:assert contrasts[i]['ci_low_pp']<0<contrasts[i]['ci_high_pp']
extmethods=['rgb_histogram','gray_histogram','resnet50','resnet50_A5','dinov2_s14','dinov2_s14_A5']
extlabels=['RGB','Gray','ResNet50 center','ResNet50 A5','DINOv2 center','DINOv2 A5']
extcolors=['#568C76','#696E76','#3677A8','#3677A8','#C67B39','#C67B39']
extmarks=['^','D','o','o','s','s']
observations=[];summaries=[];paired=[];psummary=[];values={}
for task,rows,key,unit in [('MatSim',ma,'macro_top1','seed'),('DTD',dt,'recall1','split')]:
    for m in extmethods:
        rr=sorted([r for r in rows if r['method']==m and (task=='DTD' or r['protocol']=='one_reference')],key=lambda r:int(r[unit]))
        if task=='MatSim' and m=='gray_histogram':
            assert not rr
            summaries.append(dict(task=task,method=m,n=0,mean='',sample_sd='',status='Not evaluated'))
            continue
        assert len(rr)==10 and [int(r[unit]) for r in rr]==list(range(0 if task=='MatSim' else 1,10 if task=='MatSim' else 11))
        assert all(int(r['n_queries'])==(63 if task=='MatSim' else 1880) for r in rr)
        v=np.array([num(r,key) for r in rr]);values[task,m]=v
        observations.extend(dict(task=task,method=m,unit=int(r[unit]),n_queries=int(r['n_queries']),score=num(r,key)) for r in rr)
        summaries.append(dict(task=task,method=m,n=10,mean=float(v.mean()),sample_sd=float(v.std(ddof=1)),status='evaluated'))
        line=next(s for s in prompt.splitlines() if s.startswith(f'| {task} | {m} |'))
        q=[x.strip() for x in line.split('|')[1:-1]]
        assert np.allclose([float(q[2]),float(q[3])],[v.mean(),v.std(ddof=1)],atol=6e-8,rtol=0)
        assert np.allclose([float(x) for x in q[4].split(',')],v,atol=6e-8,rtol=0)
    for m in ['resnet50','dinov2_s14']:
        v=values[task,m+'_A5']-values[task,m]
        for i,d in enumerate(v):
            u=i if task=='MatSim' else i+1
            old=pick(oldpairs,task=task.lower(),model=m,paired_unit=u)
            assert np.isclose(d,num(old,'difference_pp'),atol=1e-10)
            paired.append(dict(task=task,method=m,unit=u,difference_pp=float(d)))
        psummary.append(dict(task=task,method=m,n=10,mean=float(v.mean()),sample_sd=float(v.std(ddof=1)),negative_units=int((v < -1e-10).sum())))
assert [r['negative_units'] for r in psummary]==[2,3,4,1]
write('F6_Fabric12_plotted',selected);write('F6_Fabric_all21_complete',fm);write('F6_Fabric_paired_plotted',contrasts)
write('F6_MatSim_DTD_all110_observations',observations);write('F6_MatSim_DTD_summary',summaries)
write('F6_MatSim_DTD_all40_paired_effects',paired);write('F6_MatSim_DTD_paired_summary',psummary)

W,H=180,275;INK='#202020'
style={'font.family':'Arial','font.size':8,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'svg.fonttype':'none','svg.hashsalt':'JDIS-F6-20261029','pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'}
with mpl.rc_context(style):
    fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=170)
    can=fig.add_axes([0,0,1,1]);can.set(xlim=(0,W),ylim=(0,H));can.axis('off')
    def tx(x,y,s,size=8,weight='normal',**kw):return can.text(x,y+5,s,fontsize=size,fontweight=weight,va='center',**kw)
    def axmm(x,y,w,h):return fig.add_axes([x/W,(y+5)/H,w/W,h/H])
    def basic(ax):
        ax.tick_params(labelsize=8,length=2,pad=2)
        ax.grid(axis='x',color='#E6E9EC',lw=.5,zorder=0)
        ax.spines['left'].set_visible(False)
        ax.xaxis.label.set_size(8)
    tx(4,264,'External evidence at instance, material and attribute levels',10.4,'bold')
    can.plot([4,176],[263,263],c='#B8C2CA',lw=.7)
    tx(4,253,'(a)  T1K+ Fabric · surface-instance IDs',10.3,'bold')
    tx(4,247,'Strict: 1,082 records / 1,079 distinct test photos / 354 source clusters',8)
    tx(4,242,'Sensitivity: 1,076 queries / 353 clusters; evaluation only, no retraining',8)
    tx(76,235,'355-label strict',8.5,'bold',ha='center')
    tx(148,235,'353-label sensitivity',8.5,'bold',ha='center')
    a1=axmm(53,181,46,49);a2=axmm(126,181,46,49)
    for j,(ax,study) in enumerate([(a1,'primary355'),(a2,'clean353')]):
        for i,(m,co,mark) in enumerate(zip(methods,colors,markers)):
            r=pick(fm,study=study,method=m);v=num(r,'recall1');lo=num(r,'recall1_ci_low');hi=num(r,'recall1_ci_high')
            e=ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt=mark,c=co,mfc=co if j==0 else 'white',mec=INK if j==0 else co,mew=.45,ms=3.6,lw=.85,capsize=1.8,zorder=3)
            if j==1:
                for lc in e[2]:lc.set_linestyle('--')
            if j==0:
                ax.text(-.03,i,labels[i],transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=8)
                ax.text(1.07,i,f'{v:.2f}',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=8)
        for yy in [5.5,6.5,8.5]:ax.axhline(yy,c='#D5DCE1',lw=.55)
        ax.set(xlim=(55,100),ylim=(11.6,-.6),yticks=[],xticks=[60,80,100],xlabel='Recall@1 (%)');basic(ax)
    tx(4,168,'Fabric whiskers: conditional 95% source-cluster intervals; 12 of 21 methods',8)
    tx(4,158,'(b)  Fabric · paired differences on the strict 355-label test',10.1,'bold')
    tx(4,152,'95% source-cluster intervals · 1,082 records · 354 clusters',8)
    b=axmm(83,114,89,32)
    for i,r in enumerate(contrasts):
        v=r['difference_pp'];co='#C67B39' if r['treatment'].startswith('dino') else '#3677A8';mark='s' if co=='#C67B39' else 'o'
        b.errorbar(v,i,xerr=[[v-r['ci_low_pp']],[r['ci_high_pp']-v]],fmt=mark,c=co,ms=3.8,mec=INK,mew=.4,lw=.95,capsize=2)
        b.text(-.04,i,r['label'],transform=b.get_yaxis_transform(),ha='right',va='center',fontsize=8)
    b.axvline(0,c='#555555',lw=.8)
    b.set(xlim=(-1.5,10.6),ylim=(6.6,-.6),yticks=[],xticks=[0,2,4,6,8,10],xlabel='Paired difference (pp)');basic(b)
    tx(4,102,'Full − last-stage: +0.03 [−0.70, +0.75] pp; exploratory, unadjusted intervals',8)
    can.plot([4,176],[101,101],c='#B8C2CA',lw=.7)
    tx(4,92,'MatSim · 16 material classes',9.6,'bold')
    tx(94,92,'DTD · 47 texture attributes',9.6,'bold')
    tx(4,86,'79 valid-mask photos; 63 queries / draw',8)
    tx(94,86,'1,880 test queries / official split',8)
    tx(4,81,'10 fixed reference selections',8)
    tx(94,81,'10 overlapping official splits',8)
    tx(4,74,'(c)  Macro Recall@1 · mean ± draw SD',8.5,'bold')
    tx(94,74,'(e)  Recall@1 · mean ± split SD',8.5,'bold')
    c=axmm(34,39,50,30);e=axmm(124,39,50,30)
    # Fixed y offsets make even coincident values individually countable; x never jittered.
    offsets=np.linspace(-.24,.24,10)
    point_counts={}
    for task,ax in [('MatSim',c),('DTD',e)]:
        for i,(m,co,mk) in enumerate(zip(extmethods,extcolors,extmarks)):
            if (task,m) not in values:
                ax.text(3,i,'Not evaluated',fontsize=8,va='center',color='#696E76');continue
            v=values[task,m];point_counts[task+'|'+m]=len(v)
            ax.plot(v,i+offsets,mk,c=co,mfc='white',mec=co,ms=2.5,mew=.5,linestyle='none',alpha=.8,zorder=3)
            ax.errorbar(v.mean(),i,xerr=v.std(ddof=1),fmt=mk,c=co,mfc=co if m.endswith('A5') else 'white',mec=INK,mew=.5,ms=4.5,capsize=2.5,lw=1.3,zorder=4)
        ax.set(xlim=(0,85) if task=='MatSim' else (0,55),ylim=(5.6,-.6),yticks=range(6),yticklabels=extlabels,xticks=[0,20,40,60,80] if task=='MatSim' else [0,20,40],xlabel='Macro Recall@1 (%)' if task=='MatSim' else 'Recall@1 (%)');basic(ax)
    tx(4,23,'(d)  A5 − center · mean ± draw SD',8.5,'bold')
    tx(94,23,'(f)  A5 − center · mean ± split SD',8.5,'bold')
    d=axmm(34,4,50,14);f=axmm(124,4,50,14)
    for task,ax in [('MatSim',d),('DTD',f)]:
        for i,(m,co,mk) in enumerate(zip(['resnet50','dinov2_s14'],['#3677A8','#C67B39'],['o','s'])):
            v=values[task,m+'_A5']-values[task,m]
            ax.plot(v,i+offsets,mk,c=co,mfc='white',mec=co,ms=2.5,mew=.5,linestyle='none',alpha=.8,zorder=3)
            ax.errorbar(v.mean(),i,xerr=v.std(ddof=1),fmt=mk,c=co,ms=4.5,mec=INK,mew=.5,capsize=2.5,lw=1.3,zorder=4)
        ax.axvline(0,c='#555555',lw=.8);ax.set(ylim=(1.55,-.55),yticks=[0,1],yticklabels=['ResNet50','DINOv2'],xlim=(-7,13) if task=='MatSim' else (-1.7,2.4),xticks=[-5,0,5,10] if task=='MatSim' else [-1,0,1,2],xlabel='Paired difference (pp)');basic(ax)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();overflow=[]
    texts=[t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text()]
    for t in texts:
        box=t.get_window_extent(renderer)
        if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5:overflow.append(dict(text=t.get_text(),bbox=list(box.bounds)))
    assert not overflow,overflow
    assert min(t.get_fontsize() for t in texts)>=8
    assert np.allclose([a1.transData.transform((60,i))[1] for i in range(12)],[a2.transData.transform((60,i))[1] for i in range(12)])
    stem=OUT/'F6_external_tasks_distinct_relevance'
    for ext in ['svg','pdf','png']:fig.savefig(stem.with_suffix('.'+ext),dpi=600,bbox_inches=None,facecolor='white')
    fig.savefig(OUT/'F6_preview.png',dpi=170,bbox_inches=None,facecolor='white')
    Image.open(OUT/'F6_preview.png').convert('L').save(OUT/'F6_grayscale_QA.png')
    plt.close(fig)
svg=ET.parse(stem.with_suffix('.svg'));ns={'s':'http://www.w3.org/2000/svg'}
assert not svg.findall('.//s:image',ns)
png=Image.open(stem.with_suffix('.png'))
manifest=dict(date='2026-10-29',figure='F6',physical_mm=[W,H],pixels=list(png.size),dpi=list(png.info['dpi']),SVG_text_elements=len(svg.findall('.//s:text',ns)),SVG_raster_images=0,source_files=SOURCES,Fabric_selected_methods=12,Fabric_complete_methods=21,Fabric_plotted_intervals=24,Fabric_paired_intervals=7,MatSim_DTD_score_points=110,MatSim_DTD_paired_points=40,score_point_counts=point_counts,paired_summary=psummary,SD_ddof=1,MatSim_gray='Not evaluated',saved_CI_reused_without_bootstrap=True,Fabric_sensitivity_evaluation_only=True,minimum_font_pt=8,text_overflow=overflow,python=sys.executable,matplotlib=mpl.__version__,numpy=np.__version__,model_API_calls=0,API_keys_used=False,training_inference_bootstrap=False,manuscript_overwritten=False)
(OUT/'validation_and_provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
caption='External evidence at distinct relevance levels. (a) T1K+ Fabric surface-instance retrieval: strict 355-label evaluation (1082 label records, 1079 distinct test photos, 354 source clusters; filled markers, solid intervals) and 353-label sensitivity (1076 records, 353 clusters; open markers, dashed intervals). The sensitivity removes ambiguous exact-copy labels at evaluation and re-scores the same fitted models; it is not 353-label retraining. Twelve representative methods are shown in prespecified rule order; all 21 methods and both evaluations accompany the figure. DINOv2 A30 averages digital views of one source photo. Fusion weights were selected only on validation (RGB α=.25; LBP α=.10), with all candidate results retained. Whiskers in (a,b) are saved conditional 95% source-cluster bootstrap intervals: the exact-copy pair a452/a456 shares one source cluster; fixed pool/training runs were averaged within query before resampling. Run SD is separate and retained in the accompanying CSV: strict A30 pool SD .23 pp; linear, last-stage and full training-seed SD .51, .53 and .42 pp, respectively. Fixed-run CSV zeros do not mean zero population uncertainty. (b) Strict-test paired contrasts retain the zero-crossing DINOv2-minus-ResNet50 and full-minus-last-stage intervals; these do not establish equivalence or definitive ordering. Comparisons are descriptive and not multiplicity-adjusted. (c,d) MatSim: only the 79-photo valid-mask subset, 16 material classes, one reference per class and 63 queries in each of ten fixed selections; macro Recall@1 weights classes equally. Gray was not evaluated. (e,f) DTD: 47 texture attributes, one seeded official-train reference per attribute and all 1880 official-test queries in each of ten overlapping official splits. (c,e) Every small open marker is one selection/split; larger markers and capped whiskers are the mean ± sample SD (ddof=1). Larger filled markers denote A5 and open markers center descriptors. (d,f) Every within-selection/split A5-minus-center effect is retained, including negative effects; capped whiskers show the mean ± sample SD. Small points are offset only vertically to separate coincident values. Draw/split SD is not a query confidence interval and differs from Fabric source-cluster uncertainty. Repeated selections and overlapping splits are not independent datasets. The three relevance tasks are not pooled; none verifies session/device independence of the original leather acquisition.'
(OUT/'F6_caption.txt').write_text(caption,encoding='utf-8')
(OUT/'F6_alt_text.txt').write_text('Fabric instance retrieval is separated from MatSim material classes and DTD texture attributes. Fabric shows 12 methods under strict355 and sensitivity353, with source-cluster intervals, and seven paired contrasts. CLIP strict Recall@1 is 92.70%. DINOv2 versus ResNet50 and full versus last-stage intervals cross zero. MatSim and DTD show all 110 recorded selection/split scores and all 40 neural A5-minus-center effects, with means and sample SD. MatSim Gray is marked Not evaluated. Both encoders have negative A5 effects on both tasks. No task average is calculated.',encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k not in ['source_files','score_point_counts']},ensure_ascii=False,indent=2))
