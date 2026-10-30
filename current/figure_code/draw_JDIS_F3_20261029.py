"""Figure 3: deterministic data-bound scientific figure, no model/API calls."""
from pathlib import Path
import csv, json, hashlib, math, shutil, re, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from PIL import Image
import xml.etree.ElementTree as ET

ROOT=Path(r'D:/科研/IC_project')
REV=ROOT/'output/JDIS_revision_20261029'
PROMPTS=ROOT/'output/JDIS_main8_PaperBanana_prompts_20261029'
OUT=ROOT/'output/JDIS_F3_local_20261029'
OUT.mkdir(exist_ok=True)
DATA=REV/'source_data/leather/results'
SOURCE_NAMES=['clean_metrics.csv','c1_results.csv','c1_paired.csv']
def read(n):
    with (DATA/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

METHODS=['resnet50','dinov2_s14','rgb_histogram','gray_histogram']
LABELS=['ResNet50','DINOv2-S/14','RGB histogram','Gray histogram']
COLORS=['#3677A8','#C67B39','#568C76','#696E76']
FILLS=['#EAF2F8','#FCF1E3','#EAF2ED','#F1F1F3']
FAMILIES=['regional_crop','perspective','resolution','illumination','device_pipeline','compound']
ROWLABELS=['Regional crop','2D perspective\nwarp','Resolution loss','Shading / highlight\noverlay','Color / noise / JPEG','Ordered compound']
INK='#202020'
clean=[r for r in read('clean_metrics.csv') if r['variant']=='frozen']
br=[r for r in read('c1_results.csv') if r['protocol']=='B']
paired=[r for r in read('c1_paired.csv') if r['protocol']=='B' and r['severity']=='moderate' and r['method'] in METHODS[:2]]
assert len(br)==48 and len(paired)==12
assert len({(r['family'],r['severity'],r['method']) for r in br})==48
assert all(int(r['original_queries'])==201 and int(r['synthetic_inputs'])==603 for r in br)
matrix=np.array([[float(next(r for r in br if r['family']==f and r['severity']==s and r['method']==m)['recall1']) for m in METHODS for s in ['mild','moderate']] for f in FAMILIES])
assert matrix.shape==(6,8) and np.isfinite(matrix).all() and matrix.min()>=0 and matrix.max()<=100
assert [round(float(next(r for r in clean if r['method']==m)['recall1'])*201/100) for m in METHODS]==[197,190,197,161]
for r in paired:
    assert r['comparison']=='minus_RGB'
    method_value=float(next(z for z in br if z['method']==r['method'] and z['family']==r['family'] and z['severity']=='moderate')['recall1'])
    rgb_value=float(next(z for z in br if z['method']=='rgb_histogram' and z['family']==r['family'] and z['severity']=='moderate')['recall1'])
    assert math.isclose(method_value-rgb_value,float(r['difference_pp']),abs_tol=1e-10)
    assert float(r['ci_low'])<=float(r['difference_pp'])<=float(r['ci_high'])

# Validate source precision against the previously delivered prompt, without
# using rounded prompt text to determine plot geometry.
prompt=(PROMPTS/'F3/content.txt').read_text(encoding='utf-8')
for f,vals in zip(FAMILIES,matrix):
    line=next(s for s in prompt.splitlines() if s.startswith('| '+f+' |') and len(s.split('|'))==11)
    numbers=[float(s.strip()) for s in line.split('|')[2:-1]]
    assert np.allclose(numbers,vals,rtol=0,atol=6e-8),(f,numbers,vals)

for n in SOURCE_NAMES:
    (OUT/'source_data').mkdir(exist_ok=True)
    shutil.copy2(DATA/n,OUT/'source_data'/n)
writecsv('F3_clean_plotted.csv',[dict(method=m,recall1=float(next(r for r in clean if r['method']==m)['recall1']),hits=round(float(next(r for r in clean if r['method']==m)['recall1'])*201/100),queries=201,gallery_candidates=203) for m in METHODS])
writecsv('F3_B_all48_plotted.csv',br)
writecsv('F3_moderate_12contrasts_plotted.csv',paired)
writecsv('F3_heatmap_wide.csv',[dict(family=f,**{f'{m}_{s}':float(matrix[i,j*2+k]) for j,m in enumerate(METHODS) for k,s in enumerate(['mild','moderate'])}) for i,f in enumerate(FAMILIES)])

mm=1/25.4
style={'font.family':'Arial','font.size':8,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
       'svg.fonttype':'none','svg.hashsalt':'JDIS-F3-20261029','pdf.fonttype':42,'ps.fonttype':42,
       'axes.linewidth':.7,'xtick.major.width':.7,'ytick.major.width':.7,'axes.spines.top':False,'axes.spines.right':False,
       'savefig.facecolor':'white','figure.facecolor':'white','text.usetex':False}
with mpl.rc_context(style):
    fig=plt.figure(figsize=(180*mm,120*mm),dpi=150)
    canvas=fig.add_axes([0,0,1,1]);canvas.set(xlim=(0,180),ylim=(0,120));canvas.axis('off')
    def text(x,y,s,size=8,weight='normal',**kw):return canvas.text(x,y,s,fontsize=size,fontweight=weight,va='center',**kw)
    def axmm(x,y,w,h):return fig.add_axes([x/180,y/120,w/180,h/120])
    text(4,115,'(a)  Clean leather Q',11,'bold')
    text(176,115,'201 original queries · 203 gallery candidates',8,ha='right')
    cards=[]
    for i,(m,label,c,fc) in enumerate(zip(METHODS,LABELS,COLORS,FILLS)):
        x=4+43.5*i; y=89; width=41.5; height=20.5
        canvas.add_patch(FancyBboxPatch((x,y),width,height,boxstyle='round,pad=0,rounding_size=1.4',facecolor=fc,edgecolor=INK,linewidth=.7))
        value=float(next(r for r in clean if r['method']==m)['recall1']);hits=round(value*201/100)
        text(x+3,y+15.5,label,8.7,'bold',color=INK)
        text(x+3,y+7.1,f'{value:.2f}%',16,'bold',color=c)
        text(x+width-2.5,y+6.4,f'{hits}/201',8,ha='right')
        cards.append((x,y,width,height))
    text(4,83,'(b)  All 12 Protocol B conditions',10.3,'bold')
    text(101,83,'(c)  Moderate: neural − RGB',10.3,'bold')
    text(101,76.8,'95% original-query-cluster intervals',8)

    hm=axmm(34,24,58,41)
    cmap=mpl.colormaps['Blues'];norm=mpl.colors.Normalize(vmin=0,vmax=100)
    mesh=hm.pcolormesh(np.arange(9)-.5,np.arange(7)-.5,matrix,cmap=cmap,norm=norm,
                       edgecolors='white',linewidth=.65,shading='flat',rasterized=False)
    hm.set(xlim=(-.5,7.5),ylim=(5.5,-.5),xticks=range(8),yticks=range(6),yticklabels=ROWLABELS)
    hm.set_xticklabels(['Mild','Moderate']*4,rotation=90,ha='left',va='center',rotation_mode='anchor',fontsize=8)
    hm.tick_params(axis='x',top=True,labeltop=True,bottom=False,labelbottom=False,length=0,pad=2.5)
    hm.tick_params(axis='y',length=0,pad=5,labelsize=8)
    for spine in hm.spines.values():spine.set_visible(False)
    cell_text=[];contrast=[]
    def luminance(rgb):
        a=np.array(rgb);a=np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)
        return float(a@[.2126,.7152,.0722])
    for i in range(6):
        for j in range(8):
            lum=luminance(cmap(norm(matrix[i,j]))[:3]); black=(lum+.05)/.05;white=1.05/(lum+.05)
            c='black' if black>=white else 'white';contrast.append(max(black,white))
            cell_text.append(hm.text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center',fontsize=8,color=c))
    for j,(label,c) in enumerate(zip(['ResNet50','DINOv2','RGB','Gray'],COLORS)):
        text(34+(2*j+1)*58/8,79,label,8.3,'bold',ha='center',color=c)
    for v in [1.5,3.5,5.5]:hm.axvline(v,color='white',lw=1.4)
    cax=axmm(34,16.5,58,2.25)
    cbar=fig.colorbar(mesh,cax=cax,orientation='horizontal',ticks=[0,25,50,75,100])
    cbar.solids.set_rasterized(False)
    cbar.outline.set_linewidth(.55);cbar.ax.tick_params(labelsize=8,length=2,pad=2)
    cbar.set_label('Recall@1 (%)',fontsize=8,labelpad=1.5)

    effects=axmm(137,24,39,41)
    effects.set(xlim=(-48,65),ylim=(5.5,-.5),yticks=range(6),yticklabels=ROWLABELS,xticks=[-40,0,40,60])
    effects.tick_params(axis='y',length=0,pad=5,labelsize=8)
    effects.tick_params(axis='x',labelsize=8,length=3)
    effects.spines['left'].set_visible(False)
    effects.axvline(0,color='#555555',linewidth=.85,zorder=1)
    effects.grid(axis='x',color='#E4E7EA',linewidth=.5)
    for row in range(6):
        if row%2==1:
            effects.axhspan(row-.5,row+.5,facecolor='#F7F8FA',edgecolor='none',zorder=0)
    for j,(m,marker) in enumerate(zip(METHODS[:2],['o','s'])):
        rr=[next(r for r in paired if r['method']==m and r['family']==f) for f in FAMILIES]
        v=np.array([float(r['difference_pp']) for r in rr]);lo=np.array([float(r['ci_low']) for r in rr]);hi=np.array([float(r['ci_high']) for r in rr])
        y=np.arange(6)+(j-.5)*.28
        effects.errorbar(v,y,xerr=[v-lo,hi-v],fmt=marker,color=COLORS[j],mec=INK,mew=.45,
                         markersize=3.8,linewidth=1,capsize=2.3,capthick=.8,zorder=3)
    effects.set_xlabel('Neural − RGB (pp)',fontsize=8,labelpad=4)
    handles=[Line2D([],[],color=COLORS[j],marker=m,markersize=4,linestyle='-',linewidth=1,label=lab) for j,(m,lab) in enumerate([('o','ResNet50'),('s','DINOv2')])]
    canvas.legend(handles=handles,loc='center',bbox_to_anchor=(137/180,12/120),ncol=2,frameon=False,
                  fontsize=8,handlelength=1.5,columnspacing=1.2,handletextpad=.4)
    text(4,5.5,'B: 201 original queries; 3 descendants per condition. Intervals condition on the fixed models and inputs.',8)
    text(4,1.7,'Exploratory; no multiplicity correction. Heatmap labels are rounded; exact values accompany the figure.',8)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    # Physical geometry: same row centers and height in the two lower panels.
    row_positions={'heatmap':[],'effects':[]}
    for i in range(6):
        row_positions['heatmap'].append(float(hm.transData.transform((0,i))[1]))
        row_positions['effects'].append(float(effects.transData.transform((0,i))[1]))
    assert np.allclose(row_positions['heatmap'],row_positions['effects'],atol=.01)
    assert min(contrast)>=4.5
    overflow=[]
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        if b.x0 < -.5 or b.y0 < -.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5:
            overflow.append({'text':t.get_text(),'bbox':list(b.bounds)})
    # Cell labels must fit inside their own numeric cell.
    cell_overflow=[]
    for idx,t in enumerate(cell_text):
        i,j=divmod(idx,8);b=t.get_window_extent(renderer);p=hm.transData.transform([[j-.5,i-.5],[j+.5,i+.5]])
        if b.width>abs(p[1,0]-p[0,0]) or b.height>abs(p[1,1]-p[0,1]):cell_overflow.append([i,j])
    assert not overflow,overflow
    assert not cell_overflow,cell_overflow
    # Y-label columns and the adjacent axes must have actual whitespace.
    hm_label_right=max(t.get_window_extent(renderer).x1 for t in hm.get_yticklabels())
    effect_label_left=min(t.get_window_extent(renderer).x0 for t in effects.get_yticklabels())
    assert hm_label_right<hm.get_window_extent(renderer).x0
    assert effect_label_left>hm.get_window_extent(renderer).x1+2
    stem=OUT/'F3_appearance_protocol_dependence'
    for ext in ['svg','pdf','png']:
        fig.savefig(stem.with_suffix('.'+ext),dpi=600,facecolor='white',bbox_inches=None)
    fig.savefig(OUT/'F3_preview.png',dpi=190,facecolor='white',bbox_inches=None)
    # Raster grayscale is a QA derivative, not a substitute for the color figure.
    Image.open(OUT/'F3_preview.png').convert('L').save(OUT/'F3_grayscale_QA.png')
    plt.close(fig)

svg=ET.parse(stem.with_suffix('.svg'));ns={'s':'http://www.w3.org/2000/svg'}
assert len(svg.findall('.//s:image',ns))==0,'SVG contains raster content'
svg_texts=svg.findall('.//s:text',ns)
assert len(svg_texts)>=70
png=Image.open(stem.with_suffix('.png'))
manifest={'date':'2026-10-29','figure':'F3','width_mm':180,'height_mm':120,
          'output_files':{ext:str(stem.with_suffix('.'+ext)) for ext in ['png','pdf','svg']},
          'png_pixels':list(png.size),'png_dpi':list(png.info.get('dpi',[])),
          'matplotlib_version':mpl.__version__,'numpy_version':np.__version__,'python':sys.executable,
          'SVG_native_text_elements':len(svg_texts),'SVG_embedded_rasters':0,
          'heatmap_cells':48,'paired_contrasts':12,'query_clusters':201,'descendants_per_condition':603,
          'heatmap_color_limits':[0,100],'minimum_heatmap_label_contrast':min(contrast),
          'paired_effect_geometry':'source difference_pp and source ci_low/ci_high; not inferred from rounded cells',
          'row_alignment_verified':True,'text_overflow':overflow,'cell_label_overflow':cell_overflow,
          'CI_crossing_zero_retained':sum(float(r['ci_low'])<0<float(r['ci_high']) for r in paired),
          'new_bootstrap_or_training':False,'model_API_calls':0,'API_keys_used':False,
          'source_files':[{'path':str(DATA/n),'sha256':hashlib.sha256((DATA/n).read_bytes()).hexdigest()} for n in SOURCE_NAMES]}
(OUT/'validation_and_provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
caption='Appearance cues and digital-query protocol dependence on leather. (a) Current clean-Q Recall@1 and hit counts for 201 original queries against all 203 reference candidates. Equal hit counts do not establish equivalence. (b) Every Protocol B family at both levels, using identical transformed pixels for the four fixed descriptors; cells show rounded percentages and the shared color scale spans 0–100%. (c) Moderate-condition neural-minus-RGB paired effects with conditional 95% intervals from the stored 2000-resample original-query-cluster analysis. Each query’s three descendants stay together, giving 603 derived inputs per condition but 201 resampling units. These exploratory intervals have no multiplicity correction and condition on the fixed models and inputs. Protocol B follows Resize256/CenterCrop224; highlight overlays, 2D warps and the photometric pipeline are digital operations, not measurements of physical reflectance or cross-device/session generalization.'
(OUT/'F3_caption.txt').write_text(caption,encoding='utf-8')
(OUT/'F3_alt_text.txt').write_text('Three-panel leather retrieval comparison. Clean RGB and ResNet50 both retrieve 197 of 201 queries, DINOv2 retrieves 190 and Gray 161. A complete heatmap shows that performance ordering changes across regional crop, perspective, resolution loss, shading/highlight overlay, color/noise/JPEG and compound digital operations. Moderate neural-minus-RGB effects are negative for resolution and positive for several photometric or compound conditions; three original-query-cluster intervals cross zero. All exact values and intervals accompany the figure.',encoding='utf-8')
shutil.copy2(Path(__file__),OUT/Path(__file__).name)
print(json.dumps(manifest,ensure_ascii=False,indent=2))
