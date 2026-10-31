"""Reanalyse stored predictions on the existing paired 727-query role.

Standalone replay: python fixed727_analysis.py --input-dir INPUT --output-dir OUTPUT
INPUT must contain main_query_predictions.csv and reference_budget_predictions.csv.
No feature extraction, model inference, training, or test-based selection occurs.
"""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

METHODS=['dinov2_center','dinov2_A5','dinov2_A30','dinov2_rgb_fusion','dinov2_R2']
CONTRASTS=[('dinov2_A30','dinov2_center'),('dinov2_R2','dinov2_center'),('dinov2_R2','dinov2_A30'),('dinov2_R2','dinov2_rgb_fusion')]

def write_csv(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)

def run(input_dir, output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    main_path=input_dir/'main_query_predictions.csv'
    ref_path=input_dir/'reference_budget_predictions.csv'
    with ref_path.open(encoding='utf-8-sig',newline='') as f:
        ref=[r for r in csv.DictReader(f) if r['study']=='paired_real_reference_budget' and r['method']=='dinov2']
    r1={int(r['query_index']):r for r in ref if r['run']=='1'}
    r2={int(r['query_index']):r for r in ref if r['run']=='2'}
    assert set(r1)==set(r2) and len(r1)==727
    indices=sorted(r1)
    saved={m:{} for m in METHODS[:-1]}
    selected_inputs=[]
    with main_path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            i=int(r['query_index']); m=r['method']
            if r['study']!='primary355' or m not in saved or i not in r1: continue
            assert r['query_path']==r1[i]['query_path']==r2[i]['query_path']
            assert r['true_class']==r1[i]['true_class']==r2[i]['true_class']
            assert int(r['rank']) in range(1,356)
            assert float(r['recall1'])==int(int(r['rank'])==1)
            key=(int(r['run']),i)
            assert key not in saved[m]
            saved[m][key]=r
            selected_inputs.append(r)
    outcomes={}; run_scores={}; aligned=[]
    for m in METHODS[:-1]:
        seeds=sorted({s for s,i in saved[m]})
        expected=[42,43,44] if m in ['dinov2_A5','dinov2_A30'] else [0]
        assert seeds==expected,(m,seeds)
        for s in seeds: assert {i for run,i in saved[m] if run==s}==set(indices)
        mat=np.array([[float(saved[m][s,i]['recall1']) for i in indices] for s in seeds])
        outcomes[m]=mat.mean(axis=0);run_scores[m]=mat.mean(axis=1)*100
    outcomes['dinov2_R2']=np.array([float(r2[i]['recall1']) for i in indices])
    run_scores['dinov2_R2']=np.array([outcomes['dinov2_R2'].mean()*100])
    keys=['duplicate:a452+a456' if r1[i]['true_class'] in {'a452','a456'} else r1[i]['true_class'] for i in indices]
    unique=sorted(set(keys)); assert len(unique)==354
    cluster=np.array([unique.index(k) for k in keys]); counts=np.bincount(cluster,minlength=354)
    old_center=outcomes['dinov2_center'].copy()
    existing_r1=np.array([float(r1[i]['recall1']) for i in indices])
    different=np.flatnonzero(old_center!=existing_r1)
    assert all(r1[indices[j]]['true_class'] in {'a452','a456'} for j in different)
    assert np.array_equal(np.bincount(cluster,weights=old_center,minlength=354),np.bincount(cluster,weights=existing_r1,minlength=354))
    # Preserve the saved R1 assignments; the exact-copy pair swaps which label
    # receives rank one between two saved scoring paths, with identical cluster sums.
    outcomes['dinov2_center']=existing_r1
    weights=np.random.default_rng(20261029).multinomial(354,np.full(354,1/354),size=2000)
    denominators=weights@counts
    def interval(v):
        sums=np.bincount(cluster,weights=v,minlength=354)
        return (100*np.quantile((weights@sums)/denominators,[.025,.975])).tolist()
    summaries=[]
    for m in METHODS:
        lo,hi=interval(outcomes[m]);n=len(run_scores[m])
        summaries.append(dict(method=m,n_queries=727,clusters=354,fixed_runs=n,original_reference_photos=2 if m=='dinov2_R2' else 1,digital_views_per_photo={'dinov2_A5':5,'dinov2_A30':30}.get(m,1),fusion_alpha=0.25 if m=='dinov2_rgb_fusion' else '',recall1=100*float(outcomes[m].mean()),ci_low=lo,ci_high=hi,run_sd=float(np.std(run_scores[m],ddof=1)) if n>1 else '',stored_prototype_vectors=2 if m=='dinov2_rgb_fusion' else 1))
    contrasts=[]
    for a,b in CONTRASTS:
        delta=outcomes[a]-outcomes[b];lo,hi=interval(delta)
        contrasts.append(dict(treatment=a,baseline=b,n_queries=727,clusters=354,difference_pp=100*float(delta.mean()),ci_low_pp=lo,ci_high_pp=hi))
    for j,i in enumerate(indices):
        aligned.append(dict(query_index=i,query_path=r1[i]['query_path'],true_class=r1[i]['true_class'],source_cluster=keys[j],**{m:float(outcomes[m][j]) for m in METHODS}))
    write_csv(output_dir/'fixed727_summary.csv',summaries)
    write_csv(output_dir/'fixed727_paired_contrasts.csv',contrasts)
    write_csv(output_dir/'fixed727_aligned_correctness.csv',aligned)
    write_csv(output_dir/'fixed727_main_prediction_subset.csv',selected_inputs)
    write_csv(output_dir/'fixed727_reference_prediction_subset.csv',ref)
    result={'methods':summaries,'contrasts':contrasts,'checks':{'query_alignment':True,'center_uses_existing_R1':True,'primary_center_agrees_with_R1_cluster_sums':True,'ambiguous_pair_changed_label_assignments':[indices[j] for j in different],'n_records':727,'n_clusters':354,'planned_contrasts':4,'bootstrap_draws':2000,'bootstrap_seed':20261029,'multiplicity_adjusted':False,'inference':False,'training':False},'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [main_path,ref_path]}}
    (output_dir/'fixed727_results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();run(args.input_dir,args.output_dir)
