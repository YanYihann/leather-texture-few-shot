"""Exact local F7, after studying the supplied journal-figure design package.
Only reads saved results; no model APIs, keys, training, inference or bootstrap.
"""
from pathlib import Path
import csv,json,hashlib,shutil,sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from PIL import Image
import xml.etree.ElementTree as ET

ROOT=Path(r'D:/科研/IC_project');OUT=Path(__file__).resolve().parent
PACK=ROOT/'output/JDIS_main8_PaperBanana_prompts_20261029'
FAB=ROOT/'output/JDIS_revision_20261029/source_data/fabric'
ASTRA=ROOT/'output/Sensors_figures_Astra_20261027'
LEARN=ROOT/'output/journal_figure_design_20261027/顶刊图表设计学习包'
sources=[]
def preserve(p,rel):
    dest=OUT/'source_data'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if dest.exists():assert dest.read_bytes()==p.read_bytes(),str(p)+' changed after initial copy'
        else:shutil.copy2(p,dest)
    assert dest.exists(),dest
    sources.append(dict(original_path=str(p),bundled_path=str(dest.relative_to(OUT)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    return dest
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(n,rows):
    with (OUT/(n+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def pick(rows,**kw):
    rr=[r for r in rows if all(str(r[k])==str(v) for k,v in kw.items())]
    assert len(rr)==1,(kw,len(rr));return rr[0]
def num(r,k):return float(r[k])
for n in ['risk_coverage.csv','open_set_disjoint_results.csv','external_unknown_results.csv']:
    p=preserve(PACK/'data/leather'/n,Path('leather')/n)
    assert p.read_bytes()==(ASTRA/'data'/n).read_bytes()
for n in ['open_set_partitions.csv','open_set_protocol.json']:
    preserve(ROOT/'output/Sensors_revision_20261010/new_experiments'/n,Path('leather')/n)
preserve(ROOT/'output/Sensors_revision_20261010/scripts/recompute_diagnostics.py',Path('owning_scripts/leather/recompute_diagnostics.py'))
preserve(ROOT/'output/Sensors_revision_20261011/scripts/aggregate_experiments.py',Path('owning_scripts/leather/aggregate_experiments.py'))
preserve(FAB/'results/open_set_metrics.csv',Path('fabric/open_set_metrics.csv'))
preserve(FAB/'protocol/open_set_class_splits.json',Path('fabric/open_set_class_splits.json'))
preserve(FAB/'scripts/evaluate_results.py',Path('owning_scripts/fabric/evaluate_results.py'))
for n in ['content.txt','caption.txt','checklist.md']:
    preserve(PACK/'F7'/n,Path('input_contract')/n)
for n in ['00_使用说明.md','04_GPT学习资料/给GPT的完整学习提示词.md','04_GPT学习资料/中文图表设计方法指南.md','04_GPT学习资料/图表索引.csv']:
    preserve(LEARN/n,Path('design_guidance')/Path(n).name)
# Reference PNGs are design sources only; retain their identities/hashes without redistributing figures.
reference_paths=['01_重点三图/J01_RADAM_Fig1_p04_with_caption.png','01_重点三图/J02_DTPNet_Fig3_p08_with_caption.png','01_重点三图/J03_PATSP-Net_Fig2_p03_with_caption.png','02_高清图表/01_高水平期刊/J06_LibFewShot/J06_LibFewShot_Fig1_p03.png','02_高清图表/03_顶会补充/C02_FroFA/C02_FroFA_Fig3_p06.png','02_高清图表/01_高水平期刊/J05_IFR/J05_IFR_Fig6_p12.png','02_高清图表/01_高水平期刊/J02_DTPNet/J02_DTPNet_Table3_p14.png']
reference_manifest=[]
for p in reference_paths:
    f=LEARN/p
    if f.exists():reference_manifest.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),use='Viewed before design; organization/encoding only, not experiment evidence'))
(OUT/'design_reference_manifest.json').write_text(json.dumps(reference_manifest,ensure_ascii=False,indent=2),encoding='utf-8')
risk=read(OUT/'source_data/leather/risk_coverage.csv')
held=read(OUT/'source_data/leather/open_set_disjoint_results.csv')
ext=read(OUT/'source_data/leather/external_unknown_results.csv')
parts=read(OUT/'source_data/leather/open_set_partitions.csv')
fabric=read(OUT/'source_data/fabric/open_set_metrics.csv')
plan=json.loads((OUT/'source_data/fabric/open_set_class_splits.json').read_text())
prompt=(OUT/'source_data/input_contract/content.txt').read_text(encoding='utf-8')
models=['resnet50','dinov2_s14'];rules=['score','margin'];qs=[0,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9]
assert len(risk)==1320 and len(ext)==40
assert {num(r,'unknown_prevalence') for r in risk}=={.1,.5,.9}
partition_summary=[]
for k in range(10):
    groups={role:{r['class_id'] for r in parts if int(r['split'])==k and r['role']==role} for role in ['unknown_evaluation','known_calibration','known_evaluation']}
    u,c,e=[groups[r] for r in ['unknown_evaluation','known_calibration','known_evaluation']]
    assert (len(u),len(c),len(e))==(20,90,91)
    assert not(u&c or u&e or c&e) and len(u|c|e)==201
    partition_summary.append(dict(split=k,unknown_evaluation=20,known_calibration=90,known_evaluation=91,known_Q_total=181,gallery_references=183))
curve=[]
for m in models:
    for rule in rules:
        for q in qs:
            rr=[r for r in risk if r['model']==m and r['rule']==rule and num(r,'calibration_quantile')==q and num(r,'unknown_prevalence')==.5]
            assert len(rr)==10 and {int(r['split']) for r in rr}==set(range(10))
            row=dict(method=m,rule=rule,calibration_quantile=q,unknown_prevalence=.5,partitions=10,coverage_mean=float(np.mean([num(r,'coverage') for r in rr])),risk_mean=float(np.mean([num(r,'selective_risk') for r in rr])))
            line=next(s for s in prompt.splitlines() if s.startswith(f'| {m} | {rule} | {q:g} |'))
            vals=[x.strip() for x in line.split('|')[1:-1]]
            assert np.allclose([float(vals[3]),float(vals[4])],[row['coverage_mean'],row['risk_mean']],atol=6e-10,rtol=0)
            curve.append(row)
endpoints=[]
for m in models:
    for rule in rules:
        h=[r for r in held if r['method']==m and r['rule']==rule]
        x=[r for r in ext if r['model']==m and r['rule']==rule]
        assert len(h)==len(x)==10
        assert all((int(r['calibration_n']),int(r['known_evaluation_n']),int(r['unknown_n']),int(r['library_n']))==(90,91,20,183) for r in h)
        assert all((int(r['known_n']),int(r['external_unknown_n']))==(91,79) for r in x)
        a=100*np.mean([num(r,'unknown_rejection') for r in h]);b=100*np.mean([num(r,'external_rejection') for r in x])
        line=next(s for s in prompt.splitlines() if s.startswith(f'| {m} | {rule} |') and len(s.split('|'))==6)
        vals=[v.strip() for v in line.split('|')[1:-1]]
        assert np.allclose([a,b],[float(vals[2]),float(vals[3])],atol=6e-8,rtol=0)
        endpoints.append(dict(method=m,rule=rule,held_out_rejection_pct=float(a),external_rejection_pct=float(b),held_out_unknown_Q=20,held_out_gallery=183,external_unknown_photos=79,external_gallery=203,known_evaluation_queries_per_partition=91,overlapping_partitions=10,threshold_rule='known-only 5th percentile, recomputed for each gallery'))
sets=[set(plan[k]) for k in ['known','calibration_unknown','test_unknown']]
assert [len(x) for x in sets]==[282,35,36]
assert len(set.union(*sets))==353 and all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3))
assert set(plan['excluded_ambiguous_classes'])=={'a452','a456'}
fm=['rgb_center','gray_center','lbp_center','resnet50_center','dinov2_center','clip_center']
fl=['RGB','Gray','LBP','ResNet50','DINOv2','CLIP']
fc=['#568C76','#696E76','#8C6D46','#3677A8','#C67B39','#887CA7'];mk=['^','D','v','o','s','P']
operating=[]
for m in fm:
    r=pick(fabric,method=m,detector='maximum_cosine')
    assert tuple(int(r[k]) for k in ['known_classes','calibration_unknown_classes','test_unknown_classes','known_test_queries','unknown_test_queries'])==(282,35,36,855,181)
    accepted=round(num(r,'known_accept_rate')*855);joint=round(num(r,'known_identification_and_acceptance')*855)
    rejected=round(num(r,'unknown_reject_rate')*181)
    assert np.isclose(accepted/855,num(r,'known_accept_rate')) and np.isclose(joint/855,num(r,'known_identification_and_acceptance')) and np.isclose(rejected/181,num(r,'unknown_reject_rate'))
    out=dict(method=m,threshold=num(r,'threshold'),known_acceptance_pct=100*num(r,'known_accept_rate'),unknown_rejection_pct=100*num(r,'unknown_reject_rate'),joint_correct_accepted_pct=100*num(r,'known_identification_and_acceptance'),AUROC=num(r,'auroc'),known_test_queries=855,unknown_test_photos=181,accepted_known_count=accepted,correct_accepted_known_count=joint,rejected_unknown_count=rejected,conditional_correct_among_accepted_pct=100*joint/accepted,diagnostic_final_test_FPR_at_known_TPR95_pct=100*num(r,'unknown_fpr_at_known_tpr95'))
    line=next(s for s in prompt.splitlines() if s.startswith(f'| {m} |'))
    vals=[v.strip() for v in line.split('|')[1:-1]]
    assert np.allclose([float(v) for v in vals[1:]],[out[k] for k in ['threshold','known_acceptance_pct','unknown_rejection_pct','joint_correct_accepted_pct','AUROC','diagnostic_final_test_FPR_at_known_TPR95_pct']],atol=6e-8,rtol=0)
    operating.append(out)
assert round(pick(operating,method='resnet50_center')['AUROC'],4)==.9652
write('F7_leather_risk44_mean_points_plotted',curve);write('F7_leather_risk1320_all_scenarios',risk)
write('F7_leather_unknown_population_endpoints_plotted',endpoints);write('F7_leather_partition_roles',partition_summary)
write('F7_fabric6_maximum_cosine_plotted',operating);write('F7_fabric12_all_detectors_complete',fabric)

W,H=180,270;INK='#202020';BLUE='#3677A8';ORANGE='#C67B39'
style={'font.family':'Arial','font.size':8,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'svg.fonttype':'none','svg.hashsalt':'JDIS-F7-20261029','pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'}
with mpl.rc_context(style):
    fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=170)
    can=fig.add_axes([0,0,1,1]);can.set(xlim=(0,W),ylim=(0,H));can.axis('off')
    def tx(x,y,s,size=8,weight='normal',**kw):return can.text(x,y,s,fontsize=size,fontweight=weight,va='center',**kw)
    def axis(x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
    def box(x,y,w,h,s,fc='#F4F6F8'):
        can.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1',fc=fc,ec='#8A949C',lw=.65))
        tx(x+w/2,y+h/2,s,8,ha='center',linespacing=1.2)
    def path(points,arrow=True):
        if len(points)>2:can.plot([p[0] for p in points[:-1]],[p[1] for p in points[:-1]],c='#65717B',lw=.8)
        can.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>' if arrow else '-',mutation_scale=7,lw=.8,color='#65717B',shrinkA=0,shrinkB=0))
    def basic(ax):
        ax.tick_params(labelsize=8,length=2,pad=2)
        ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8)
        ax.grid(c='#E6E9EC',lw=.5)
    tx(4,264,'Unknown rejection depends on calibration and population',10.6,'bold')
    can.plot([4,176],[257.5,257.5],c='#B8C2CA',lw=.7)
    tx(4,253,'(a)  Leather · known-only calibration',10.3,'bold')
    tx(4,247,'Disjoint roles within each of ten overlapping Q partitions',8)
    box(4,211,35,24,'201 Q identities\n10 overlapping\npartitions')
    box(50,228,56,14,'20 unknown evaluation Q\nremove their 20 references')
    box(120,228,56,14,'183-reference gallery\n203 − 20 held-out references')
    box(50,207,56,14,'181 known Q identities')
    box(120,210,56,14,'90 known calibration Q')
    box(120,192,56,14,'91 known evaluation Q')
    path([(39,223),(44,223),(44,235),(50,235)])
    path([(44,223),(44,214),(50,214)])
    path([(106,235),(120,235)])
    path([(106,214),(114,214),(114,217),(120,217)])
    path([(114,214),(114,199),(120,199)])
    tx(4,194,'Final evaluation: 91 known + 20 unknown Q',8)
    tx(4,188,'The gallery retains the two reference IDs absent from Q.',8)
    can.plot([4,176],[183.5,183.5],c='#B8C2CA',lw=.7)
    tx(4,178,'(b)  Leather risk–coverage',10.1,'bold')
    tx(98,178,'(c)  Different unknown populations',9.5,'bold')
    tx(4,172,'Specified unknown prevalence = 0.50',8)
    tx(98,172,'Known-only 5th-percentile rule',8)
    b=axis(16,123,68,43)
    for i,(m,co,mark) in enumerate(zip(models,[BLUE,ORANGE],['o','s'])):
        for rule,ls in [('score','-'),('margin','--')]:
            rr=[r for r in curve if r['method']==m and r['rule']==rule]
            b.plot([r['coverage_mean'] for r in rr],[r['risk_mean'] for r in rr],c=co,ls=ls,marker=mark,mfc=co if rule=='score' else 'white',mec=co,ms=3,lw=1,label=('ResNet50' if i==0 else 'DINOv2')+' '+rule)
    b.set(xlim=(0,1.025),ylim=(-.025,.6),xticks=[0,.25,.5,.75,1],yticks=[0,.2,.4,.6],xlabel='Coverage',ylabel='Selective risk');basic(b)
    b.axhline(0,c='#A2AAB0',lw=.6,zorder=0)
    b.legend(loc='upper left',bbox_to_anchor=(-.19,-.28),frameon=False,fontsize=8,ncol=2,handlelength=1.5,columnspacing=.8,handletextpad=.4,labelspacing=.4,borderaxespad=0)
    c=axis(134,133,41,30)
    rowlabels=['ResNet50\nscore','ResNet50\nmargin','DINOv2\nscore','DINOv2\nmargin']
    for i,r in enumerate(endpoints):
        co=BLUE if r['method']=='resnet50' else ORANGE
        c.plot(r['held_out_rejection_pct'],i-.13,'o',c=co,mfc='white',mec=co,ms=4,linestyle='none')
        c.plot(r['external_rejection_pct'],i+.13,'s',c=co,mfc=co,mec=INK,mew=.35,ms=3.6,linestyle='none')
    c.set(xlim=(-3,105),ylim=(3.6,-.6),yticks=range(4),yticklabels=rowlabels,xticks=[0,50,100],xlabel='Unknown rejection (%)');basic(c);c.spines['left'].set_visible(False)
    tx(98,116,'○ Held-out: 20 Q; 183 references',8)
    tx(98,111,'■ External: 79; full 203 references',8)
    tx(98,106,'Means over ten partitions; no pairing',8)
    tx(4,98,'50% is a reweighted scenario, not measured field prevalence.',8)
    can.plot([4,176],[93.5,93.5],c='#B8C2CA',lw=.7)
    tx(4,87,'(d)  Fabric · separate calibration with known and unknown photos',10,'bold')
    tx(4,80,'282 known / 35 calibration-unknown / 36 final-unknown IDs; frozen descriptors only',8)
    tx(4,75,'Final test: 855 known records / 181 unknown photos; maximum-cosine rule',8)
    tx(4,70,'Known validation + calibration-unknown maximize balanced accuracy; higher threshold wins ties',8)
    tx(4,64.5,'Method',8,'bold')
    tx(52,62.5,'Known\naccept (%)',8,'bold',ha='center',linespacing=1.15)
    tx(96,62.5,'Unknown\nreject (%)',8,'bold',ha='center',linespacing=1.15)
    tx(140,62.5,'Joint correct\n+ accept (%)',8,'bold',ha='center',linespacing=1.15)
    tx(169.5,62.5,'AUROC',8,'bold',ha='center')
    can.plot([4,176],[58.5,58.5],c='#79858E',lw=.7)
    rateaxes=[]
    for x,key,nx in [(33,'known_acceptance_pct',70),(77,'unknown_rejection_pct',114),(121,'joint_correct_accepted_pct',158)]:
        ax=axis(x,22,23,34);rateaxes.append(ax)
        for i,(r,co,mark) in enumerate(zip(operating,fc,mk)):
            ax.plot(r[key],i,mark,c=co,mec=INK,mew=.35,ms=4,linestyle='none',zorder=3)
            yy=22+(5.5-i)/6*34
            tx(nx,yy,f'{r[key]:.2f}',8,ha='right')
        ax.set(xlim=(0,100),ylim=(5.5,-.5),yticks=[],xticks=[0,100]);basic(ax);ax.spines['left'].set_visible(False)
    for i,(r,label,co,mark) in enumerate(zip(operating,fl,fc,mk)):
        yy=22+(5.5-i)/6*34
        tx(4,yy,label,8)
        tx(169.5,yy,f'{r["AUROC"]:.4f}',8,ha='center')
    tx(4,12,'Joint correct + accepted / all 855 known queries; AUROC: known positive, 0–1.',8)
    tx(4,6,'Acceptance is not identification accuracy. Leather and Fabric use different calibration.',8)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();overflow=[]
    texts=[t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text()]
    for t in texts:
        z=t.get_window_extent(renderer)
        if z.x0<-.5 or z.y0<-.5 or z.x1>fig.bbox.width+.5 or z.y1>fig.bbox.height+.5:overflow.append(dict(text=t.get_text(),bbox=list(z.bounds)))
    assert not overflow,overflow
    assert min(t.get_fontsize() for t in texts)>=8
    assert len(b.lines)==5 and all(len(l.get_xdata())==11 for l in b.lines[:4])
    assert len(c.lines)==8 and all(len(l.get_xdata())==1 for l in c.lines)
    assert all(np.allclose([ax.transData.transform((50,i))[1] for i in range(6)],[rateaxes[0].transData.transform((50,i))[1] for i in range(6)]) for ax in rateaxes)
    stem=OUT/'F7_calibration_rejection_and_review_risk'
    for extn in ['svg','pdf','png']:fig.savefig(stem.with_suffix('.'+extn),dpi=600,bbox_inches=None,facecolor='white')
    fig.savefig(OUT/'F7_preview.png',dpi=170,bbox_inches=None,facecolor='white')
    Image.open(OUT/'F7_preview.png').convert('L').save(OUT/'F7_grayscale_QA.png')
    plt.close(fig)
svg=ET.parse(stem.with_suffix('.svg'));ns={'s':'http://www.w3.org/2000/svg'}
assert not svg.findall('.//s:image',ns)
im=Image.open(stem.with_suffix('.png'))
manifest=dict(date='2026-10-29',figure='F7',physical_mm=[W,H],PNG_pixels=list(im.size),PNG_dpi=im.info['dpi'],SVG_images=0,SVG_text_elements=len(svg.findall('.//s:text',ns)),source_files=sources,learning_package_studied_before_drawing=True,viewed_reference_figures=len(reference_manifest),risk_points=44,risk_points_per_series=11,risk_input_rows_all_scenarios=1320,unknown_prevalence=.5,reweighted_scenario_not_field_estimate=True,endpoint_points=8,endpoint_connections=0,endpoint_summary=endpoints,Fabric_rows=6,Fabric_metrics_per_row=4,Fabric_complete_detector_rows=12,Fabric_class_roles=[282,35,36],Fabric_known_test_records=855,Fabric_unknown_test_photos=181,Fabric_threshold_selection='Known validation plus calibration-unknown only; maximum balanced accuracy; higher threshold wins ties',minimum_font_pt=8,text_overflow=overflow,python=sys.executable,matplotlib=mpl.__version__,numpy=np.__version__,model_API_calls=0,API_keys_used=False,training_inference_bootstrap=False,manuscript_overwritten=False)
(OUT/'validation_and_provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
caption='Unknown rejection depends on calibration and unknown population. (a) Leather roles are disjoint within each of ten overlapping partitions of 201 original Q identities: 20 held-out identities lose their references, leaving a 183-entry gallery; the other 181 Q identities split into 90 known calibration and 91 known evaluation queries. Two reference IDs without Q remain in the gallery. Branches indicate role assignment, not test-data flow through calibration. Thresholds use only known-query score or top-two-margin quantiles. (b) Selective risk versus coverage at specified unknown prevalence 0.50, a reweighted scenario rather than measured field prevalence. Points are arithmetic means over the ten partitions at the eleven saved calibration quantiles (0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9); straight segments connect successive recorded operating points without smoothing. Risk includes both accepted unknowns and accepted incorrectly identified known queries. Solid/filled and dashed/open marks denote maximum-cosine and top-two-margin rules, respectively; blue circles and orange squares denote ResNet50 and DINOv2. All .10/.50/.90 scenarios accompany the figure. (c) Rejection at the known-calibration fifth percentile, averaged across the same ten partitions: open circles, 20 held-out leather Q against 183 references; filled squares, 79 masked external MatSim photos against the full 203-entry leather gallery. The threshold is recomputed for the applicable gallery using only known calibration Q. Endpoints are unconnected because unknown populations and gallery sizes differ. The external source records use 91 known evaluation queries per partition, not 201. No error bars are added to these descriptive means. (d) Fabric uses a separate class-disjoint protocol after excluding ambiguous exact-copy labels a452/a456: 282 known, 35 calibration-unknown and 36 final-unknown IDs. Known validation and calibration-unknown photos maximize balanced accuracy; a higher threshold breaks ties. Final evaluation uses 855 known records and 181 unknown photos. Only six frozen descriptors are shown, with maximum-cosine operating points; both recorded detectors remain in the complete data. Known acceptance and joint correct-and-accepted each divide by all 855 known queries, whereas conditional correctness divides by accepted known queries and is retained separately in the plotted-data CSV. Unknown rejection divides by 181 unknown photos. AUROC treats known as positive and uses the 0–1 scale. The optional ResNet50 final-test ROC diagnostic gives unknown false acceptance 24.86% at 95% known acceptance; this is not the calibration-selected threshold or a calibrated guarantee. Overlapping partitions are not independent datasets. Neither protocol measures deployed human-review savings or certifies rejection of professionally verified near-equivalent unknown leather.'
caption=caption.replace('Endpoints are unconnected because unknown populations and gallery sizes differ.','Endpoints are unconnected because unknown populations and gallery sizes differ; symbols are offset only vertically within each row for legibility.')
(OUT/'F7_caption.txt').write_text(caption,encoding='utf-8')
(OUT/'F7_alt_text.txt').write_text('Four panels separate leather role assignment, leather risk–coverage at a reweighted 50% unknown scenario, two distinct leather unknown populations, and a separately calibrated Fabric evaluation. Leather holds out 20 identities, retains 183 references, calibrates on 90 known queries and evaluates 91 known plus 20 unknown queries. Forty-four saved mean operating points are shown. Eight unconnected endpoint symbols keep held-out rejection and external masked-MatSim rejection distinct. Fabric six frozen descriptors report known acceptance, unknown rejection, joint correct-and-accepted and AUROC, using 855 known queries and 181 unknown photos. ResNet50 is 90.18%, 90.61%, 88.54%, .9652, respectively. Acceptance is distinct from correct identification.',encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k not in ['source_files','endpoint_summary']},ensure_ascii=False,indent=2))
