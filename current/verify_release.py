"""Offline integrity and saved-rank arithmetic; no training, inference or CI resampling."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json,hashlib,math,sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def metrics(rows):
    ranks=[int(r['rank']) for r in rows]
    return [100*sum(r<=k for r in ranks)/len(ranks) for k in (1,3,5)]+[sum(1/r for r in ranks)/len(ranks)]
def near(a,b,tol=1e-7):
    assert math.isclose(float(a),float(b),abs_tol=tol,rel_tol=0),(a,b)
def main():
    manifest=read(HERE/'FILE_MANIFEST.csv')
    for r in manifest:
        p=ROOT/r['path'];assert p.is_file(),p
        assert p.stat().st_size==int(r['bytes']),p
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],p
    leather=HERE/'source_data/leather/results'
    groups=defaultdict(list)
    for r in read(leather/'clean_predictions.csv'):groups[r['method'],r['variant'],r['reference_seed']].append(r)
    for r in read(leather/'clean_metrics.csv'):
        vals=metrics(groups[r['method'],r['variant'],r['seed']])
        assert len(groups[r['method'],r['variant'],r['seed']])==201
        for a,k in zip(vals,['recall1','recall3','recall5','mrr']):near(a,r[k])
    fab=HERE/'workspace/output/T1K_fabric_experiments_20261029'
    photo=json.loads((fab/'protocol/photo_splits.json').read_text(encoding='utf-8'))
    assert len(photo)==1792
    assert Counter(r['role'] for r in photo)=={'reference':355,'validation':355,'test':1082}
    assert len({r['sha256'] for r in photo})==1787
    for field in ['sha256','decoded_pixel_sha256']:
        roles=defaultdict(set)
        for r in photo:roles[r[field]].add(r['role'])
        assert all(len(v)==1 for v in roles.values()),field
    groups=defaultdict(list)
    for r in read(fab/'predictions/main_query_predictions.csv'):groups[r['study'],r['method'],r['run']].append(r)
    checked=0
    for r in read(fab/'results/main_metrics.csv'):
        if r['study']!='primary355':continue
        sub=[v for (study,method,run),v in groups.items() if study==r['study'] and method==r['method']]
        assert len(sub)==int(r['fixed_runs']),(r['method'],len(sub))
        vals=[metrics(v) for v in sub]
        assert all(len(v)==int(r['n_queries']) for v in sub)
        for i,key in enumerate(['recall1','recall3','recall5','mrr']):near(sum(v[i] for v in vals)/len(vals),r[key])
        checked+=1
    report={'manifest_files':len(manifest),'leather_clean_groups':len(read(leather/'clean_metrics.csv')),'fabric_metric_rows':checked,'fabric_records':len(photo),'fabric_unique_files':1787,'fabric_no_cross_role_file_or_pixel_duplicates':True,'scope':'SHA-256 integrity, saved-rank arithmetic and fixed-split audit only; no fresh training/inference, raw-data recapture or confidence-interval resampling'}
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
