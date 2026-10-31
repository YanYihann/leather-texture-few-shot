"""Verify stored fixed-query point estimates; no models, training or resampling."""
from pathlib import Path
import csv,json,math

HERE=Path(__file__).resolve().parent
DATA=HERE/'reanalyses/fixed727'
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
rows=read(DATA/'results/fixed727_aligned_correctness.csv')
summary=read(DATA/'results/fixed727_summary.csv')
contrasts=read(DATA/'results/fixed727_paired_contrasts.csv')
assert len(rows)==727
assert len({r['query_index'] for r in rows})==727
assert len({r['source_cluster'] for r in rows})==354
methods=[r['method'] for r in summary]
assert len(methods)==5 and len(set(methods))==5
values={m:100*sum(float(r[m]) for r in rows)/len(rows) for m in methods}
for r in summary:
    assert int(r['n_queries'])==727 and int(r['clusters'])==354
    assert math.isclose(values[r['method']],float(r['recall1']),abs_tol=1e-9)
    assert float(r['ci_low'])<=float(r['recall1'])<=float(r['ci_high'])
assert len(contrasts)==4
print(json.dumps({'query_records':727,'source_clusters':354,'methods':5,'stored_contrasts':4,'point_estimates':values,'scope':'Stored alignment and point-estimate arithmetic only; no new training, inference or interval resampling.'},indent=2))
