"""End-to-end synthetic evaluation check. Never reads actual cached model features."""
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('t1k_eval', HERE/'evaluate_results.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
original = evaluation.OUT
evaluation.self_test()
with tempfile.TemporaryDirectory(prefix='t1k-synthetic-evaluation-') as temporary:
    destination = Path(temporary).resolve()
    assert destination.name.startswith('t1k-synthetic-evaluation-')
    evaluation.OUT = destination
    for name in ['protocol','cache','results','predictions']:
        (destination/name).mkdir()
    for filename in ['experiment_protocol.json','photo_splits.json','reference_sensitivity.json','open_set_class_splits.json']:
        (destination/'protocol'/filename).write_bytes((original/'protocol'/filename).read_bytes())
    rows = json.loads((destination/'protocol/photo_splits.json').read_text(encoding='utf-8'))
    y = np.array([row['label'] for row in rows])
    classes = sorted({row['class_name'] for row in rows})
    ri = np.array([row['index'] for row in rows if row['role']=='reference'])
    vi = np.array([row['index'] for row in rows if row['role']=='validation'])
    qi = np.array([row['index'] for row in rows if row['role']=='test'])
    rng = np.random.default_rng(12345)
    arrays = {}
    for name, dim in [('rgb',8),('gray',7),('lbp',10),('resnet50',12),('dinov2',14),('clip',16)]:
        centers = rng.normal(size=(355,dim)).astype(np.float32)
        centers[classes.index('a456')] = centers[classes.index('a452')]
        array = evaluation.normalize(centers[y]+rng.normal(0,.05,size=(1792,dim)).astype(np.float32))
        # Synchronize the known exact-copy photos across the duplicate class labels.
        for field_hash in {row['sha256'] for row in rows if row['class_name']=='a452'}:
            members = [row['index'] for row in rows if row['sha256']==field_hash]
            for index in members[1:]:
                array[index]=array[members[0]]
        arrays[name]=array
        if name in ['resnet50','dinov2','clip']:
            np.save(destination/f'cache/{name}_base.npy',array)
        if name in ['resnet50','dinov2']:
            augmented=evaluation.normalize(array[ri][None,:,None,:]+rng.normal(0,.03,size=(3,355,30,dim)).astype(np.float32))
            np.save(destination/f'cache/{name}_aug.npy',augmented)
    np.savez_compressed(destination/'cache/handcrafted.npz',**{name:arrays[name] for name in ['rgb','gray','lbp']})
    adaptation={}
    for mode in ['linear','partial','full']:
        for seed in [42,43,44]:
            adaptation[f'{mode}_{seed}_validation']=20*(arrays['resnet50'][vi]@arrays['resnet50'][ri].T)
            adaptation[f'{mode}_{seed}_test']=20*(arrays['resnet50'][qi]@arrays['resnet50'][ri].T)
    np.savez_compressed(destination/'results/adaptation_scores.npz',**adaptation)
    selected = [dict(mode=mode,seed=seed,lr=.001,selected_epoch=1,epochs_executed=10,
                     presentations=3550,updates=230,checkpoint_sha256='0'*64,
                     validation_correct=int(evaluation.score_details(adaptation[f'{mode}_{seed}_validation'],y[vi])['r1'].sum()))
                for mode in ['linear','partial','full'] for seed in [42,43,44]]
    (destination/'results/adaptation_selection.json').write_text(json.dumps(selected),encoding='utf-8')
    inputs=evaluation.load_inputs()
    protocol,rows,classes,labels,ix,features,augmented,adaptation,selection=inputs
    summary,paired,stability,count=evaluation.main_analysis(rows,classes,labels,ix,features,augmented,adaptation)
    budgets,budget_paired,sensitivity=evaluation.reference_and_budget_sensitivity(rows,classes,labels,ix,features)
    open_results=evaluation.open_set_analysis(rows,classes,labels,ix,features)
    evaluation.report(protocol,summary,paired,stability,budgets,budget_paired,sensitivity,open_results,selection)
    assert len(summary)==42 and len(stability)==10 and len(paired)==16
    assert count==1082*47, count
    assert len(budgets)==12 and len(budget_paired)==6 and len(sensitivity)==30
    assert len(open_results)==12
    assert (destination/'results/实验结果报告.md').exists()
    assert len([row for row in summary if row['study']=='primary355'])==21
    assert all(row['bootstrap_clusters']==354 for row in summary if row['study']=='primary355')
    assert all(row['bootstrap_clusters']==353 for row in summary if row['study']=='clean353')
    assert all(0<=row['auroc']<=1 for row in open_results)
    assert all(row['recall1_ci_low']<=row['recall1_ci_high'] for row in summary)
    print('PASS: synthetic end-to-end42summary rows/47runs,50854prediction rows, paired/reference-choice/open-set analyses and Chinese report; no actual test features read.')
    # Windows keeps mapped .npy files locked until their mappings are closed.
    for mapping in [features, augmented]:
        for array in mapping.values():
            if isinstance(array,np.memmap):
                array._mmap.close()
    adaptation.close()
evaluation.OUT=original
