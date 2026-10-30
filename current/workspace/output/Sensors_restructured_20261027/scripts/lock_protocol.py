from pathlib import Path
import json, hashlib, datetime, shutil, sys, platform
O=Path(__file__).resolve().parents[1]; ROOT=O.parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
config={
 'release':'current-20261027-r2','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'status':'locked before reading new C1-C5 outputs; exploratory follow-up, not preregistration',
 'base_pdf_sha256':sha(ROOT/'output/Sensors_revision_20261027/package/main.pdf'),
 'task':'203 physical swatch identities; 201 original Q images; S is spatial development only',
 'methods':['rgb_histogram_8x8x8','gray_histogram_32','resnet50','dinov2_s14'],
 'C1':{'A':'all 8 x 3 original conditions, identical legacy SHA-derived inputs before crop','B':'all 6 x 2 x 3 original conditions, verify every pixel hash','histograms':'single deterministic reference only','statistics':'2000 query-cluster bootstrap seed 20261027, paired differences to RGB'},
 'C2':{'ranking':'R@1/3/5 and MRR from all 203 scores, stable tie break','hard_subgroups':'union of identities incident to either legacy reference challenge graph, evaluate against all 203 references; report both graphs separately','stability':'36 descendants per original query, within pool then mean over three pools','examples':'all four current frozen ResNet errors; strongest wrong ref; lexicographic order'},
 'C3':{'data':'existing 5640 DTD jpg only','protocol':'reuse exact legacy dtd_protocol.json refs and official test list for all 10 splits','methods':['rgb_histogram','gray_histogram','resnet50','resnet50_A5','dinov2_s14','dinov2_s14_A5'],'cache_validation':'names labels unit norms; recompute selected fixed features and first class A5 of every split; full hash inventory','A5':'legacy seeds and transforms, identical view hashes for both backbones','no_download':True},
 'C4':{'device':'CPU','threads':3,'batch_query':1,'input':224,'warmup':10,'query_repeats':100,'registration_repeats':3,'registration_views':[1,5,30],'query_selection':'cycle Q in sorted folder order, first 100; warmup first 10','registration':'all 203 references, on-the-fly legacy seed42 view generation, batch16 encoding, pool normalize-mean-normalize','timing':'separate preprocessing, encoding and matching; independently measured end-to-end; decoded PIL input, disk IO and weight loading excluded; cold post-load first query recorded separately','memory':'Windows process peak working set; includes loaded weights and runtime, not incremental allocations'},
 'C5':{'factors':['channel_gains','gamma','noise','jpeg'],'levels':['mild','moderate'],'repeats':[0,1,2],'inputs':4824,'sampling':'replay B device_pipeline RNG exactly; same gains/gamma/sigma/quality and same Gaussian draw on clean 224 image; isolate each operation','resampling':{'down':['BOX','LANCZOS'],'up':'BICUBIC','repeat':0,'levels':['mild','moderate'],'inputs':804,'intermediate':'same B resolution per-query sampled side'},'methods':'all four single-reference methods; no search'},
 'statistics':{'unit':'201 original query identities; descendants clustered','bootstrap_reps':2000,'seed':20261027,'interpretation':'conditional descriptive intervals, no multiplicity-adjusted confirmatory claims; pool SD and training SD separate'},
 'excluded':['new captures','metadata backfill','expert labels','new in-domain unknown swatches','C6 CLIP/RADAM/VORTEX compute','extra finetuning searches','uploads or submission'],
 'figure_assignment':{'F1':'task and A/B protocols','F2':'dual-path frozen retrieval and normalize-mean-normalize inset','F3':'clean appearance controls, full B heatmap, paired gain to RGB','F4':'N, angle, margin and paired A30 effects','F5':'all-query ECDF and all four ResNet errors','F6':'separate MatSim and DTD comparisons','F7':'calibration, risk-coverage and operating points','F8':'measured CPU query and registration cost'},
 'software':{'python':sys.version,'platform':platform.platform()}}
p=O/'audit/locked_protocol.json'
if p.exists(): raise RuntimeError('Refusing to overwrite protocol lock')
p.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
(O/'audit/locked_protocol.sha256').write_text(sha(p)+'  locked_protocol.json\n')
for n in ['revision_plan_20261027/具体修改方案_待用户审核.md','competitive_review_20261027/论文与高水平期刊差距_详细中文清单.md']:
 shutil.copy2(ROOT/'output'/n,O/'audit'/Path(n).name)
print('LOCKED',sha(p),flush=True)
