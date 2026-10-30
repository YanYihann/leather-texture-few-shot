"""Independent artifact verification. Does not load or run any neural model.

Requires evaluation_complete.json before reading prediction/metric outputs.
Replays scoring/statistics from saved scores, checks validation-only choices,
checkpoint bytes and unchanged source/manuscript files. Not independent training.
"""
import ast
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
AMBIGUOUS = {'a452', 'a456'}
SEEDS = [42, 43, 44]
METRICS = ['recall1', 'recall3', 'recall5', 'macro_recall1', 'mrr',
           'true_cosine_or_logit', 'true_margin']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for data in iter(lambda: file.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def close(actual, expected, label, atol=1e-8):
    assert np.isfinite(actual) and np.isfinite(expected), label
    assert np.isclose(actual, expected, rtol=1e-9, atol=atol), (label, actual, expected)


def ranking(matrix, truth, candidates):
    """Independent rank from direct greater-than/tie counts, not argsort inversion."""
    matrix = np.asarray(matrix)
    truth = np.asarray(truth)
    assert matrix.shape == (len(truth), len(candidates))
    assert np.isfinite(matrix).all()
    positions = np.searchsorted(candidates, truth)
    assert np.array_equal(candidates[positions], truth)
    real = matrix[np.arange(len(truth)), positions]
    greater = matrix > real[:, None]
    earlier_tie = (matrix == real[:, None]) & (candidates[None, :] < truth[:, None])
    rank = 1 + greater.sum(axis=1) + earlier_tie.sum(axis=1)
    top = np.argsort(-matrix, axis=1, kind='stable')[:, :5]
    wrong = matrix.copy()
    wrong[np.arange(len(truth)), positions] = -np.inf
    return {'rank': rank, 'r1': (rank == 1).astype(float), 'r3': (rank <= 3).astype(float),
            'r5': (rank <= 5).astype(float), 'mrr': 1/rank, 'true_score': real,
            'margin': real - wrong.max(axis=1), 'top': candidates[top],
            'top_score': np.take_along_axis(matrix, top[:, :1], axis=1)[:, 0]}


def metrics(outcome, labels):
    return {'recall1': 100*float(np.mean(outcome['r1'])),
            'recall3': 100*float(np.mean(outcome['r3'])),
            'recall5': 100*float(np.mean(outcome['r5'])),
            'macro_recall1': 100*float(np.mean([
                outcome['r1'][labels == label].mean() for label in np.unique(labels)])),
            'mrr': float(np.mean(outcome['mrr'])),
            'true_cosine_or_logit': float(np.mean(outcome['true_score'])),
            'true_margin': float(np.mean(outcome['margin']))}


def bootstrap_values(outcome, labels, classes):
    """Resample whole source-ID clusters; duplicate-label pair shares one unit."""
    names = ['duplicate:a452+a456' if classes[label] in AMBIGUOUS else classes[label]
             for label in labels]
    names_sorted = sorted(set(names))
    cluster = np.array([names_sorted.index(name) for name in names])
    n = len(names_sorted)
    draws = np.random.default_rng(20261029).multinomial(n, np.full(n, 1/n), size=2000)
    denominators = np.bincount(cluster, minlength=n)
    result = {}
    for metric, key in [('recall1','r1'), ('recall3','r3'), ('recall5','r5'), ('mrr','mrr')]:
        numerators = np.bincount(cluster, weights=outcome[key], minlength=n)
        values = (draws @ numerators) / (draws @ denominators)
        scale = 1 if key == 'mrr' else 100
        result[metric+'_ci_low'], result[metric+'_ci_high'] = scale*np.quantile(values, [.025,.975])
    # Macro means give equal weight to official labels, while their resampling unit
    # remains the connected source cluster (two labels in one cluster where needed).
    numerator = np.zeros(n)
    denominator = np.zeros(n)
    for label in np.unique(labels):
        group = labels == label
        cl = cluster[np.flatnonzero(group)[0]]
        numerator[cl] += outcome['r1'][group].mean()
        denominator[cl] += 1
    lo, hi = 100*np.quantile((draws @ numerator)/(draws @ denominator), [.025,.975])
    result['macro_recall1_ci_low'] = lo
    result['macro_recall1_ci_high'] = hi
    return n, result


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def validate_main_predictions(rows, classes, labels, indices, summary, run_rows):
    prediction_rows = read_csv(OUT/'predictions/main_query_predictions.csv')
    predictions = collections.defaultdict(list)
    for row in prediction_rows:
        assert row['study'] == 'primary355'
        predictions[(row['method'], int(row['run']))].append(row)
    run_table = {(row['study'],row['method'],int(row['run'])):row for row in run_rows}
    assert len(run_table) == len(run_rows)
    summary_table = {(row['study'],row['method']):row for row in summary}
    assert len(summary_table) == len(summary) == 42
    methods = sorted({row['method'] for row in summary})
    assert len(methods) == 21
    clean_candidate = np.array([label for label,name in enumerate(classes) if name not in AMBIGUOUS])
    clean = ~np.isin(labels[indices['test']], [classes.index(name) for name in AMBIGUOUS])
    clean_count = int(clean.sum())
    completed_keys = set()
    calculated_outcomes = {}
    total_scores = 0
    for method in methods:
        archive = np.load(OUT/f'predictions/scores/{method}.npz')
        assert np.array_equal(archive['query_indices'], indices['test'])
        assert np.array_equal(archive['query_labels'], labels[indices['test']])
        assert archive['classes'].tolist() == classes
        runs = sorted(int(key.split('_',1)[1]) for key in archive.files if key.startswith('test_'))
        assert runs in [[0], SEEDS], (method,runs)
        outcomes = {'primary355': [], 'clean353': []}
        for run in runs:
            pred = predictions[(method,run)]
            assert len(pred) == 1082
            assert [int(row['query_index']) for row in pred] == indices['test'].tolist()
            details = ranking(archive[f'test_{run}'], labels[indices['test']], np.arange(355))
            for i, row in enumerate(pred):
                index = int(row['query_index'])
                assert row['query_path'] == rows[index]['relative_path']
                assert row['true_class'] == rows[index]['class_name']
                assert int(row['rank']) == details['rank'][i]
                for k in range(1,6):
                    assert row[f'top{k}'] == classes[details['top'][i,k-1]]
                for k in [1,3,5]:
                    assert int(row[f'recall{k}']) == details[f'r{k}'][i]
                close(float(row['top1_score']), float(details['top_score'][i]), 'prediction top score')
                close(float(row['true_score']), float(details['true_score'][i]), 'prediction true score')
                close(float(row['true_margin']), float(details['margin'][i]), 'prediction margin')
            completed_keys.add((method,run))
            for study, matrix, truth, candidate in [
                ('primary355', archive[f'test_{run}'], labels[indices['test']], np.arange(355)),
                ('clean353', archive[f'test_{run}'][clean][:,clean_candidate], labels[indices['test']][clean],clean_candidate),
                ('validation355', archive[f'validation_{run}'], labels[indices['validation']], np.arange(355))]:
                out = ranking(matrix,truth,candidate)
                recorded = run_table[(study,method,run)]
                assert int(recorded['n_queries']) == len(truth)
                for metric,value in metrics(out,truth).items():
                    close(float(recorded[metric]), value, f'{study}/{method}/{run}/{metric}')
                if study != 'validation355':
                    outcomes[study].append(out)
                total_scores += matrix.size
        for study in ['primary355','clean353']:
            truth = labels[indices['test']] if study == 'primary355' else labels[indices['test']][clean]
            averaged = {key: np.mean([run[key] for run in outcomes[study]],axis=0)
                        for key in ['r1','r3','r5','mrr','true_score','margin']}
            recorded = summary_table[(study,method)]
            assert int(recorded['n_queries']) == (1082 if study == 'primary355' else clean_count)
            assert int(recorded['n_classes']) == (355 if study == 'primary355' else 353)
            assert int(recorded['fixed_runs']) == len(runs)
            assert int(recorded['unique_query_source_photos']) == len({
                rows[int(i)]['sha256'] for i in (indices['test'] if study == 'primary355' else indices['test'][clean])})
            for metric,value in metrics(averaged,truth).items():
                close(float(recorded[metric]), value, f'{study}/{method}/{metric}')
            n_clusters, intervals = bootstrap_values(averaged,truth,classes)
            assert int(recorded['bootstrap_clusters']) == n_clusters == (354 if study=='primary355' else 353)
            for key,value in intervals.items():
                close(float(recorded[key]),float(value), f'{study}/{method}/{key}')
            for metric,key in [('recall1','r1'),('recall3','r3'),('recall5','r5'),('mrr','mrr')]:
                scale=1 if key=='mrr' else 100
                values=[scale*run[key].mean() for run in outcomes[study]]
                value=float(np.std(values,ddof=1)) if len(values)>1 else 0.
                close(float(recorded[metric+'_run_sd']),value,f'{study}/{method}/{metric} SD')
            calculated_outcomes[(study,method)] = averaged
    assert set(predictions) == completed_keys
    assert len(prediction_rows) == sum((3 if method.startswith(('resnet50_A','dinov2_A')) or
        method in ['resnet50_linear','resnet50_partial','resnet50_full'] else 1)*1082 for method in methods)
    return dict(methods=len(methods),summary_rows=len(summary),prediction_rows=len(prediction_rows),
                independently_ranked_score_values=total_scores,clean_query_count=clean_count,
                source_identity_clusters={'primary355':354,'clean353':353}), calculated_outcomes


def validate_score_origins(indices,labels):
    handcrafted=np.load(OUT/'cache/handcrafted.npz')
    features={name:handcrafted[name] for name in ['rgb','gray','lbp']}
    for name in ['resnet50','dinov2','clip']:
        features[name]=np.load(OUT/f'cache/{name}_base.npy',mmap_mode='r')
    sampled=np.linspace(0,len(indices['test'])-1,11,dtype=int)
    for name,feature in features.items():
        saved=np.load(OUT/f'predictions/scores/{name}_center.npz')['test_0'][sampled]
        expected=np.asarray(feature[indices['test'][sampled]]) @ np.asarray(feature[indices['reference']]).T
        assert np.allclose(saved,expected,atol=2e-6),name
    for name in ['resnet50','dinov2']:
        aug=np.load(OUT/f'cache/{name}_aug.npy',mmap_mode='r')
        for n in [1,5,10,20,30]:
            saved=np.load(OUT/f'predictions/scores/{name}_A{n}.npz')
            for si,seed in enumerate(SEEDS):
                proto=normalize(aug[si,:,:n].mean(axis=1))
                expected=np.asarray(features[name][indices['test'][sampled]]) @ proto.T
                assert np.allclose(saved[f'test_{seed}'][sampled],expected,atol=2e-6),(name,n,seed)
    # Recompute all fusion validation candidates before verifying the chosen test matrix.
    choices=read_json(OUT/'results/fusion_selection.json')
    for record in choices:
        hand=record['component']
        dino=np.load(OUT/'predictions/scores/dinov2_center.npz')
        other=np.load(OUT/f'predictions/scores/{hand}_center.npz')
        counts=[]
        for alpha in [0,.1,.25,.5,.75,1]:
            val=(1-alpha)*dino['validation_0']+alpha*other['validation_0']
            count=int(ranking(val,labels[indices['validation']],np.arange(355))['r1'].sum())
            counts.append((alpha,count))
        chosen=max(counts,key=lambda pair:pair[1])
        assert record['alpha']==chosen[0] and record['validation_correct']==chosen[1]
        assert [(item['alpha'],item['validation_correct']) for item in record['candidates']]==counts
        saved=np.load(OUT/f'predictions/scores/dinov2_{hand}_fusion.npz')['test_0']
        expected=(1-chosen[0])*dino['test_0']+chosen[0]*other['test_0']
        assert np.array_equal(saved,expected)
    adaptation=np.load(OUT/'results/adaptation_scores.npz')
    for mode in ['linear','partial','full']:
        saved=np.load(OUT/f'predictions/scores/resnet50_{mode}.npz')
        for seed in SEEDS:
            for role in ['validation','test']:
                assert np.array_equal(saved[f'{role}_{seed}'],adaptation[f'{mode}_{seed}_{role}'])
    return {'frozen_score_reconstruction_sampled_queries':len(sampled),
            'all_augmented_N_and_seed_score_reconstruction':True,
            'all_fusion_candidates_selected_validation_only':True,'adaptation_score_export_exact':True}


def validate_training(protocol_hash):
    candidates=read_json(OUT/'results/adaptation_candidates.json')
    selected=read_json(OUT/'results/adaptation_selection.json')
    assert len(candidates)==12 and len(selected)==9
    assert {(x['mode'],x['seed']) for x in selected}=={
        (mode,seed) for mode in ['linear','partial','full'] for seed in SEEDS}
    by_key={}
    hashes={}
    for record in candidates:
        key=(record['mode'],record['seed'],record['lr'])
        assert key not in by_key
        checkpoint=OUT/'checkpoints'/record['checkpoint']
        checksum=hash_file(checkpoint)
        assert checksum==record['checkpoint_sha256']
        hashes[checkpoint.name]=checksum
        history=read_csv(Path(str(checkpoint)+'.history.csv'))
        assert len(history)==10
        best=max(int(x['validation_correct']) for x in history)
        epoch=next(int(x['epoch']) for x in history if int(x['validation_correct'])==best)
        assert record['validation_correct']==best and record['selected_epoch']==epoch
        for i,row in enumerate(history,1):
            assert int(row['epoch'])==i and int(row['seed'])==record['seed']
            assert row['mode']==record['mode'] and float(row['lr'])==record['lr']
            assert int(row['updates'])==23*i and int(row['presentations'])==355*i
        assert record['epochs_executed']==10 and record['updates']==230 and record['presentations']==3550
        sidecar=read_json(Path(str(checkpoint)+'.record.json'))
        assert sidecar['checkpoint_sha256']==checksum
        assert sidecar['selected_epoch']==epoch and sidecar['validation_correct']==best
        by_key[key]=record
    score=np.load(OUT/'results/adaptation_scores.npz')
    rows=read_json(OUT/'protocol/photo_splits.json')
    val_labels=np.array([x['label'] for x in rows if x['role']=='validation'])
    for mode in ['linear','partial','full']:
        rates=[.001,.01] if mode=='linear' else [1e-5,1e-4]
        assert all((mode,42,rate) in by_key for rate in rates)
        best=max([by_key[(mode,42,rate)] for rate in rates],key=lambda x:x['validation_correct'])
        mode_selected=[x for x in selected if x['mode']==mode]
        for record in mode_selected:
            assert record['lr']==best['lr']
            match=by_key[(mode,record['seed'],record['lr'])]
            for field in ['selected_epoch','validation_correct','checkpoint','checkpoint_sha256']:
                assert record[field]==match[field]
            actual=ranking(score[f'{mode}_{record["seed"]}_validation'],val_labels,np.arange(355))
            assert int(actual['r1'].sum())==record['validation_correct']
    # Inspect selection execution structure without executing training or models.
    source=(OUT/'scripts/train_adaptation.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    run=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run')
    literals={node.value for node in ast.walk(run) if isinstance(node,ast.Constant) and isinstance(node.value,str)}
    assert 'test' not in literals
    assert 'validation' in literals and 'reference' in literals
    assert "if correct>best:" in source
    assert "chosen=max(candidates,key=lambda row:row['validation_correct'])" in source
    assert source.index('    final_scores(selected,ix)')>source.index("for mode in ['linear','partial','full']:")
    done=read_json(OUT/'results/adaptation.done.json')
    assert done['status']=='completed' and done['protocol_sha256']==protocol_hash
    return {'candidate_runs':12,'selected_runs':9,'checkpoint_sha256':hashes,
            'earliest_best_validation_epoch':True,'first_learning_rate_on_validation_tie':True,
            'all_selected_validation_hit_counts_match_saved_scores':True,
            'test_excluded_from_selection_static_code_check':True}


def validate_unchanged_inputs():
    manuscript=read_json(OUT/'audit/manuscript_unchanged_before.json')
    checked=[]
    for item in manuscript:
        path=Path(item['path'])
        assert path.is_file(),str(path)
        assert path.stat().st_size==item['bytes'] and hash_file(path)==item['sha256'],str(path)
        checked.append(str(path))
    images=read_json(OUT/'audit/image_manifest.json')
    for row in images:
        path=ROOT/'T1k_dataset/fabric_images'/row['relative_path']
        assert path.stat().st_size==row['bytes'] and hash_file(path)==row['sha256'],str(path)
    return {'unchanged_manuscript_files':checked,'unchanged_original_photo_entries':len(images),
            'all_original_photo_hashes_rechecked':True}


def main():
    started=time.time()
    complete_path=OUT/'results/evaluation_complete.json'
    if not complete_path.exists():
        raise RuntimeError('Evaluation is incomplete; no metrics/predictions were read.')
    completion=read_json(complete_path)
    assert completion['status']=='completed'
    protocol_hash=hash_file(OUT/'protocol/experiment_protocol.json')
    assert completion['protocol_sha256']==protocol_hash
    assert completion['script_sha256']==hash_file(OUT/'scripts/evaluate_results.py')
    rows=read_json(OUT/'protocol/photo_splits.json')
    classes=sorted({x['class_name'] for x in rows})
    labels=np.array([x['label'] for x in rows],dtype=int)
    indices={role:np.array([x['index'] for x in rows if x['role']==role],dtype=int)
             for role in ['reference','validation','test']}
    summary=read_csv(OUT/'results/main_metrics.csv')
    run_rows=read_csv(OUT/'results/per_run_metrics.csv')
    checks,outcomes=validate_main_predictions(rows,classes,labels,indices,summary,run_rows)
    assert completion['per_query_prediction_rows']==checks['prediction_rows']
    assert completion['main_summary_rows']==checks['summary_rows']
    source_checks=validate_score_origins(indices,labels)
    training_checks=validate_training(protocol_hash)
    unchanged_checks=validate_unchanged_inputs()
    result={'status':'passed','scope':'Independent artifact/statistic replay; no independent model retraining or new image acquisition',
            'protocol_sha256':protocol_hash,'verification_script_sha256':hash_file(__file__),
            'main_metrics':checks,'score_origins':source_checks,'training_selection':training_checks,
            'unchanged_inputs':unchanged_checks,'elapsed_seconds':time.time()-started,
            'limits':['Raw score export and summary metrics replay are verified, not new pretraining or independent retraining.',
                      'Hash disjointness does not certify near-duplicate absence or acquisition-session/device independence.',
                      'clean353 is evaluation-only sensitivity; adapted models were trained with355 labels.',
                      'Open-set and supplementary sensitivity CSVs remain evaluator-generated unless separately replayed.']}
    path=OUT/'audit/verification.json';temporary=Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)
    print('PASS: independently recomputed all primary/clean353 metrics, fixed-run SD and cluster intervals; checked raw rankings, selection histories, checkpoint hashes, manuscript and source image hashes.')
    print(json.dumps({key:value for key,value in checks.items()},ensure_ascii=False))


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    main()
