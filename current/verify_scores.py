"""Recompute 355/353-label Fabric scores from stored arrays; no bootstrap or model."""
from pathlib import Path
import csv,json
import sys
import numpy as np
P=Path(__file__).resolve().parent/'workspace/output/T1K_fabric_experiments_20261029'
if not (P/'predictions/scores').is_dir():
    sys.exit('Full score arrays are omitted from the compact GitHub tree. Use the complete version 2.0.0 archive at https://doi.org/10.5281/zenodo.23794034.')
with (P/'results/main_metrics.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
for row in rows:
    with np.load(P/'predictions/scores'/f"{row['method']}.npz",allow_pickle=False) as z:
        classes=z['classes'];labels=z['query_labels'];candidates=np.arange(len(classes));mask=np.ones(len(labels),dtype=bool)
        if row['study']=='clean353':
            candidates=np.flatnonzero(~np.isin(classes,['a452','a456']))
            mask=np.isin(labels,candidates)
        truth=labels[mask];values=[]
        for key in sorted(k for k in z.files if k.startswith('test_')):
            scores=z[key][mask][:,candidates]
            order=candidates[np.argsort(-scores,axis=1,kind='stable')]
            ranks=(order==truth[:,None]).argmax(1)+1
            values.append([*(100*np.mean(ranks<=k) for k in (1,3,5)),np.mean(1/ranks)])
        assert len(values)==int(row['fixed_runs'])
        assert len(truth)==int(row['n_queries'])
        for value,name in zip(np.mean(values,axis=0),['recall1','recall3','recall5','mrr']):
            assert abs(value-float(row[name]))<1e-7,(row['study'],row['method'],name,value,row[name])
print(json.dumps({'fabric_scopes':['primary355','clean353'],'rows_recomputed':len(rows),'scope':'stored score arithmetic only; no training, inference or resampling'},indent=2))
