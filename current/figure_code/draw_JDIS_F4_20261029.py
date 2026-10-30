"""Data-bound Figure 4: digital construction and original-photo budgets.
Run in the installed PaperBanana .venv for local matplotlib only; no API calls.
"""
from pathlib import Path
import csv,json,hashlib,shutil,statistics as st,sys,math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from PIL import Image
import xml.etree.ElementTree as ET

ROOT=Path(r'D:/科研/IC_project');REV=ROOT/'output/JDIS_revision_20261029'
PACK=ROOT/'output/JDIS_main8_PaperBanana_prompts_20261029'
L=PACK/'data/leather';F=REV/'source_data/fabric/results'
OUT=ROOT/'output/JDIS_F4_local_20261029';OUT.mkdir(exist_ok=True)
SOURCES=[]
def read(folder,n):
    p=folder/(n+'.csv');SOURCES.append(p)
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(n,rows):
    with (OUT/(n+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def pick(rows,**kw):
    r=[r for r in rows if all(str(r[k])==str(v) for k,v in kw.items())];assert len(r)==1,(kw,len(r));return r[0]
def number(r,k):return float(r[k])

ab=read(L,'controlled_ablation_results');angles=read(L,'prototype_angular_variation')
margin=read(L,'prototype_query_mechanism');clean=read(L,'clean_metrics');bpair=read(L,'structured_paired')
fm=read(F,'main_metrics');fa=read(F,'reference_pool_stability');fp=read(F,'paired_recall1_differences')
real=read(F,'real_reference_budget');realp=read(F,'real_reference_budget_paired')
methods=['resnet50','dinov2_s14'];fabric=['resnet50','dinov2']
ns=[1,5,10,20,30,50];nf=[1,5,10,20,30]
colors=['#3677A8','#C67B39'];markers=['o','s'];INK='#202020'
summary=[]
for m in methods:
    for n in ns:
        ar=[r for r in ab if r['model']==m and int(r['n_views'])==n and r['metric']=='cosine' and r['split']=='my_dataset_test_filtered']
        ag=[r for r in angles if r['model']==m and int(r['n_views'])==n]
        mr=[r for r in margin if r['model']==m and int(r['views'])==n]
        assert len(ar)==len(ag)==len(mr)==3
        assert {r['seed'] for r in ar}=={'42','43','44'}
        assert all(int(r['n_queries'])==201 for r in ar)
        assert {(r['seed1'],r['seed2']) for r in ag}=={('42','43'),('42','44'),('43','44')}
        summary.append(dict(method=m,N=n,recall1_mean=st.mean(number(r,'top1') for r in ar),
                            recall1_pool_sd=st.stdev(number(r,'top1') for r in ar),
                            between_pool_angle_mean=st.mean(number(r,'mean_angle_deg') for r in ag),
                            true_margin_mean=st.mean(number(r,'mean_true_margin') for r in mr),
                            true_margin_pool_sd=st.stdev(number(r,'mean_true_margin') for r in mr)))
fs=[]
for m in fabric:
    for n in nf:
        r=pick(fm,study='primary355',method=m+'_A'+str(n));a=pick(fa,method=m+'_A'+str(n))
        assert int(r['n_queries'])==1082 and int(r['bootstrap_clusters'])==354 and int(r['fixed_runs'])==3
        fs.append(dict(method=m,N=n,recall1_mean=number(r,'recall1'),recall1_pool_sd=number(r,'recall1_run_sd'),
                       between_pool_angle_mean=number(a,'mean_between_pool_angle_deg')))
fabric_delta=pick(fp,study='primary355',treatment='dinov2_A30',baseline='dinov2_center')
assert number(fabric_delta,'ci_low_pp')<0<number(fabric_delta,'ci_high_pp')
assert int(fabric_delta['n_queries'])==1082
real_methods=['rgb','gray','lbp','resnet50','dinov2','clip'];rsummary=[]
for m in real_methods:
    a=pick(real,method=m+'_R1');b=pick(real,method=m+'_R2');p=pick(realp,method=m)
    assert all(int(r['n_queries'])==727 and int(r['bootstrap_clusters'])==354 for r in [a,b,p])
    assert all(int(r['unique_query_source_photos'])==725 for r in [a,b])
    assert math.isclose(number(b,'recall1')-number(a,'recall1'),number(p,'recall1_two_minus_one_pp'),abs_tol=1e-10)
    rsummary.append(dict(method=m,R1_recall1=number(a,'recall1'),R2_recall1=number(b,'recall1'),
                         difference_pp=number(p,'recall1_two_minus_one_pp'),ci_low_pp=number(p,'ci_low_pp'),ci_high_pp=number(p,'ci_high_pp')))
assert pick(summary,method='resnet50',N=30)['recall1_mean']>pick(summary,method='resnet50',N=50)['recall1_mean']
assert pick(fs,method='resnet50',N=20)['recall1_mean']>pick(fs,method='resnet50',N=30)['recall1_mean']

# Cross-check the previously issued prompt; source CSV precision governs geometry.
prompt=(PACK/'F4/content.txt').read_text(encoding='utf-8')
for row in summary:
    line=next(s for s in prompt.splitlines() if s.startswith(f"| {row['method']} | {row['N']} |"))
    nums=[float(v.strip()) for v in line.split('|')[3:-1]]
    vals=[row[k] for k in ['recall1_mean','recall1_pool_sd','between_pool_angle_mean','true_margin_mean','true_margin_pool_sd']]
    assert np.allclose(nums,vals,rtol=0,atol=6e-8)
write('F4_leather_nested_views_plotted',summary);write('F4_leather_B_effects_plotted',bpair)
write('F4_fabric_nested_views_plotted',fs);write('F4_fabric_A30_paired_effect_plotted',[fabric_delta])
write('F4_fabric_real_photos_plotted',rsummary)
for p in SOURCES:
    dest=OUT/'source_data'/('fabric' if p.parent==F else 'leather')/p.name
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)

style={'font.family':'Arial','font.size':8,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
       'svg.fonttype':'none','svg.hashsalt':'JDIS-F4-20261029','pdf.fonttype':42,'ps.fonttype':42,
       'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,
       'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','figure.facecolor':'white'}
W,H=180,210
with mpl.rc_context(style):
    fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=150)
    can=fig.add_axes([0,0,1,1]);can.set(xlim=(0,W),ylim=(0,H));can.axis('off')
    def tx(x,y,s,size=8,weight='normal',**kw):return can.text(x,y,s,fontsize=size,fontweight=weight,va='center',**kw)
    def axis(x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
    def basic(ax):
        ax.tick_params(labelsize=8,length=2.5,pad=2)
        ax.grid(axis='y',color='#E6E9EC',lw=.5)
    def line_series(ax,rows,key,sdkey=None,nb=ns):
        for i,m in enumerate(methods if rows is summary else fabric):
            rr=[pick(rows,method=m,N=n) for n in nb];v=np.array([r[key] for r in rr])
            if sdkey:
                ax.errorbar(nb,v,yerr=[r[sdkey] for r in rr],color=colors[i],fmt=markers[i]+'-',
                            markersize=3,linewidth=1,capsize=2,capthick=.7,mec=INK,mew=.35)
            else:ax.plot(nb,v,markers[i]+'-',c=colors[i],ms=3,lw=1,mec=INK,mew=.35)
        ax.set_xticks(nb);ax.set_xlim(0,nb[-1]+2);basic(ax)
    legend=[Line2D([],[],color=colors[i],marker=markers[i],linestyle='-',ms=4,lw=1,label=lab) for i,lab in enumerate(['ResNet50','DINOv2'])]
    tx(4,204,'One original photo: digital construction budget N',10.7,'bold')
    can.legend(handles=legend,loc='center right',bbox_to_anchor=(178/W,204/H),frameon=False,ncol=1,fontsize=8,
               labelspacing=.25,handlelength=1.3,borderaxespad=0)
    can.plot([4,176],[198.5,198.5],color='#B8C2CA',lw=.7)
    tx(4,194.5,'Leather · 201 Q identities · 203-reference gallery',8.4)
    tx(4,188.8,'(a)  Query accuracy',10.3,'bold')
    tx(102,188.8,'(b)  A30 − center under Protocol B',10.3,'bold')
    tx(16,183.8,'Mean ± pool SD; dashed: center',8)
    tx(105,183.8,'Equal mean over six families; 95% cluster CI',8)
    a=axis(16,155,66,25);line_series(a,summary,'recall1_mean','recall1_pool_sd')
    for i,m in enumerate(methods):a.axhline(number(pick(clean,method=m,variant='frozen'),'recall1'),color=colors[i],ls='--',lw=.8)
    a.set(ylim=(82,101),yticks=[85,90,95,100],ylabel='Recall@1 (%)',xlabel='Digital views N')
    a.xaxis.label.set_size(8);a.yaxis.label.set_size(8)
    tx(16,143,'Random N=1 is not center cropping',8)
    b=axis(120,155,54,25)
    for i,m in enumerate(methods):
        rr=[pick(bpair,model=m,severity=s) for s in ['mild','moderate']]
        v=np.array([number(r,'difference_pp') for r in rr]);lo=np.array([number(r,'ci_low') for r in rr]);hi=np.array([number(r,'ci_high') for r in rr])
        b.errorbar(v,np.array([0,1])+(i-.5)*.23,xerr=[v-lo,hi-v],fmt=markers[i],color=colors[i],ms=3.6,capsize=2,lw=1,mec=INK,mew=.35)
    b.axvline(0,c='#555555',lw=.8);b.set(xlim=(-2.3,3.5),ylim=(1.5,-.5),yticks=[0,1],yticklabels=['Mild','Moderate'],xticks=[-2,0,2],xlabel='A30 − center (pp)')
    b.tick_params(labelsize=8,length=2.5,pad=2);b.grid(axis='x',c='#E6E9EC',lw=.5);b.spines['left'].set_visible(False)
    tx(4,134,'(c)  Between-pool angle',10.3,'bold')
    tx(102,134,'(d)  True-match margin',10.3,'bold')
    tx(16,129,'Descriptive; dependent seed pairs',8)
    tx(113,129,'Mean ± pool SD',8)
    c=axis(16,99,66,25);line_series(c,summary,'between_pool_angle_mean')
    c.set(ylim=(0,26),yticks=[0,10,20],ylabel='Angle (°)',xlabel='Digital views N')
    d=axis(113,99,61,25);line_series(d,summary,'true_margin_mean','true_margin_pool_sd')
    d.set(ylim=(.100,.171),yticks=[.11,.13,.15,.17],ylabel='True − strongest wrong',xlabel='Digital views N')
    for ax in [b,c,d]:ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8)
    tx(4,88,'Angles describe construction repeatability; they do not establish imaging invariance.',8)
    can.plot([4,176],[83.8,83.8],color='#B8C2CA',lw=.7)
    tx(4,79.5,'Fabric: digital views N and original-photo budget R',10.7,'bold')
    tx(4,73.5,'(e)  Digital views of one photo',10.3,'bold')
    tx(101,73.5,'(f)  Original photos R=1 → R=2',10.3,'bold')
    tx(16,68.8,'1,082 test records; mean ± pool SD',8)
    tx(102,68.8,'Same 727 queries (725 distinct photos)',8)
    e1=axis(16,49,66,15);line_series(e1,fs,'recall1_mean','recall1_pool_sd',nb=nf)
    for i,m in enumerate(fabric):e1.axhline(number(pick(fm,study='primary355',method=m+'_center'),'recall1'),color=colors[i],ls='--',lw=.8)
    e1.set(ylim=(90.5,98),yticks=[92,95,98],ylabel='Recall@1 (%)');e1.tick_params(labelbottom=False)
    tx(16,46,'Between-pool angle; descriptive',8)
    e2=axis(16,32,66,11);line_series(e2,fs,'between_pool_angle_mean',nb=nf)
    e2.set(ylim=(0,28),yticks=[0,10,20],ylabel='Angle (°)',xlabel='Digital views N')
    for ax in [e1,e2]:ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8)
    dv=number(fabric_delta,'recall1_difference_pp');dl=number(fabric_delta,'ci_low_pp');du=number(fabric_delta,'ci_high_pp')
    tx(16,21.5,f'DINOv2 A30 − center: {dv:+.2f} [{dl:.2f}, {du:.2f}] pp',8)
    ei=axis(16,11.5,66,5)
    ei.errorbar(dv,0,xerr=[[dv-dl],[du-dv]],fmt='s',c=colors[1],ms=3.5,lw=1,capsize=2,mec=INK,mew=.35)
    ei.axvline(0,c='#555555',lw=.8);ei.set(xlim=(-.5,2.8),ylim=(-1,1),yticks=[],xticks=[0,1,2])
    ei.spines[['left','top','right']].set_visible(False);ei.tick_params(labelsize=8,length=2,pad=2)
    tx(16,3.5,'Paired Δ (pp); 95% source-cluster CI',8)

    tx(132,64.8,'Recall@1 (%)',8,ha='center');tx(167.5,64.8,'R2 − R1 (pp)',8,ha='center')
    f1=axis(118,35,32,26);f2=axis(158,35,19,26)
    rcolors=['#568C76','#696E76','#8C6D46',colors[0],colors[1],'#887CA7']
    rmarkers=['^','D','v','o','s','P'];rlabels=['RGB','Gray','LBP','ResNet50','DINOv2','CLIP']
    for i,(m,col,mark) in enumerate(zip(real_methods,rcolors,rmarkers)):
        r=pick(rsummary,method=m);v1=r['R1_recall1'];v2=r['R2_recall1'];delta=r['difference_pp']
        f1.plot([v1,v2],[i,i],c=col,lw=1,zorder=1)
        f1.plot(v1,i,mark,c=col,mfc='white',mec=col,ms=3.8,zorder=2)
        f1.plot(v2,i,mark,c=col,mfc=col,mec=INK,mew=.35,ms=3.8,zorder=3)
        f2.errorbar(delta,i,xerr=[[delta-r['ci_low_pp']],[r['ci_high_pp']-delta]],fmt=mark,c=col,ms=3.8,lw=1,capsize=2,mec=INK,mew=.35)
    f1.set(xlim=(55,102),ylim=(5.5,-.5),yticks=range(6),yticklabels=rlabels,xticks=[60,80,100])
    f2.set(xlim=(-.5,10.6),ylim=(5.5,-.5),yticks=[],xticks=[0,5,10]);f2.axvline(0,c='#555555',lw=.7)
    for ax in [f1,f2]:ax.tick_params(labelsize=8,length=2,pad=2);ax.grid(axis='x',c='#E6E9EC',lw=.5);ax.spines['left'].set_visible(False)
    tx(102,28,'Open: R=1; filled: R=2',8)
    tx(102,23.5,'Δ bars: 95% source-cluster intervals',8)
    can.add_patch(FancyBboxPatch((101,7),76,13,boxstyle='round,pad=0,rounding_size=1.3',fc='#FCF1E3',ec='#A9A9A9',lw=.65))
    din=pick(rsummary,method='dinov2')
    tx(104,15.5,f"DINOv2: {din['R1_recall1']:.2f}% → {din['R2_recall1']:.2f}%",9,'bold')
    tx(104,10,f"{din['difference_pp']:+.2f} pp [{din['ci_low_pp']:.2f}, {din['ci_high_pp']:.2f}]",9,'bold')
    tx(102,3.5,'354 source clusters; conditional, unadjusted CIs',8)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    overflow=[]
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text():continue
        box=t.get_window_extent(renderer)
        if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5:
            overflow.append({'text':t.get_text(),'bbox':list(box.bounds)})
    assert not overflow,overflow
    assert np.allclose([f1.transData.transform((60,i))[1] for i in range(6)],[f2.transData.transform((0,i))[1] for i in range(6)],atol=.01)
    axes_map={'a':a,'b':b,'c':c,'d':d,'e_accuracy':e1,'e_angle':e2,'e_paired':ei,'f_accuracy':f1,'f_paired':f2}
    stem=OUT/'F4_reference_construction_and_photo_budget'
    for ext in ['svg','pdf','png']:fig.savefig(stem.with_suffix('.'+ext),dpi=600,bbox_inches=None,facecolor='white')
    fig.savefig(OUT/'F4_preview.png',dpi=170,bbox_inches=None,facecolor='white')
    Image.open(OUT/'F4_preview.png').convert('L').save(OUT/'F4_grayscale_QA.png')
    plt.close(fig)

svg=ET.parse(stem.with_suffix('.svg'));nsx={'s':'http://www.w3.org/2000/svg'}
assert not svg.findall('.//s:image',nsx)
png=Image.open(stem.with_suffix('.png'))
manifest={'date':'2026-10-29','figure':'F4','physical_mm':[W,H],'pixels':list(png.size),'dpi':list(png.info.get('dpi',[])),
          'SVG_text_elements':len(svg.findall('.//s:text',nsx)),'SVG_raster_images':0,
          'source_files':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in SOURCES],
          'leather_summary_points':len(summary),'fabric_digital_summary_points':len(fs),'real_photo_paired_methods':6,
          'leather_query_count':201,'Fabric_digital_query_records':1082,'Fabric_real_photo_query_records_both_arms':727,
          'Fabric_real_photo_distinct_queries':725,'Fabric_source_clusters':354,'leather_seeds':[42,43,44],
          'angle_seed_pairs_dependent':True,'pool_SD_ddof':1,'saved_source_CIs_used_without_recomputation':True,
          'DINOv2_Fabric_A30_paired_effect':fabric_delta,'ResNet50_leather_A30_to_A50_nonmonotonic_retained':True,
          'ResNet50_Fabric_A20_to_A30_nonmonotonic_retained':True,'random_A1_is_not_center':True,
          'text_overflow':overflow,'python':sys.executable,'matplotlib':mpl.__version__,'numpy':np.__version__,
          'model_API_calls':0,'API_keys_used':False,'training_or_new_bootstrap_run':False,'manuscript_overwritten':False}
(OUT/'validation_and_provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
caption='Reference construction and reference acquisition are different investments. (a) Leather clean-Q Recall@1 for nested digital-view budgets, mean ± sample SD across three shared reference pools (42–44); dashed lines denote deterministic center cropping. (b) Equal-family-mean Protocol B A30-minus-center effects with saved conditional 95% original-query-cluster intervals, retaining adverse and zero-crossing effects. (c) Descriptive mean prototype angles over 203 identities and three dependent seed pairs; no independence-based error bars. (d) Per-query true-reference-minus-strongest-wrong cosine margin, averaged across queries, with three-pool SD. (e) Strict 355-label Fabric digital-view accuracy and descriptive angles for N=1,5,10,20,30, using 1082 test records. The separate inset reports the DINOv2 A30-minus-center source-cluster interval, distinct from pool SD and crossing zero. (f) One versus two original reference photos for six frozen descriptors, both evaluated on the same 727 query records (725 distinct photos; 354 source clusters), with paired source-cluster intervals. Open/filled marks denote R=1/R=2. The second reference was removed from both evaluation conditions before scoring; these are not the primary 1082-query results. Digital A1 is random and differs from deterministic center cropping. All effect intervals condition on fixed photographs, reference choices and models; exploratory comparisons lack multiplicity correction. Smaller construction angles do not establish physical imaging invariance or explain real-photo gains.'
(OUT/'F4_caption.txt').write_text(caption,encoding='utf-8')
(OUT/'F4_alt_text.txt').write_text('Six analyses distinguish digital views of one photo from additional original photographs. Leather accuracy, prototype angles and margins are shown across nested budgets, with Protocol B paired effects. Fabric accuracy remains nonmonotonic even as construction angles decline, and the DINOv2 digital-A30 interval crosses zero. A separate paired comparison uses exactly the same 727 queries for one versus two real references: DINOv2 improves from 95.60% to 99.04%, a 3.44-percentage-point effect with source-cluster interval 1.93 to 5.08.',encoding='utf-8')
shutil.copy2(Path(__file__),OUT/Path(__file__).name)
print(json.dumps({k:v for k,v in manifest.items() if k!='source_files'},ensure_ascii=False,indent=2))
