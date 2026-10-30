"""Evaluate the locked T1K355 photo-disjoint experiment; never tune on test results.

All cached features and every selected adaptation score array are required before
evaluation. This file only reads inputs and writes results inside this experiment.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

OUT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44]
BOOTSTRAPS = 2000
BOOT_SEED = 20261029
AMBIGUOUS = {'a452', 'a456'}
FROZEN_NAMES = {
    'rgb': 'RGB histogram', 'gray': 'Gray histogram', 'lbp': 'LBP',
    'resnet50': 'ResNet50 center', 'dinov2': 'DINOv2 center', 'clip': 'CLIP center',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def score_details(scores, labels, candidates=None):
    """Stable ascending-class tie breaks. Return querywise outcomes and margins."""
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=np.int64)
    candidates = np.arange(scores.shape[1]) if candidates is None else np.asarray(candidates)
    assert scores.shape == (len(labels), len(candidates))
    assert np.all(np.isfinite(scores)) and len(set(candidates.tolist())) == len(candidates)
    assert np.all(candidates[:-1] < candidates[1:])
    local_true = np.searchsorted(candidates, labels)
    assert np.all(local_true < len(candidates)) and np.all(candidates[local_true] == labels)
    order = np.argsort(-scores, axis=1, kind='stable')
    ranks = np.argmax(order == local_true[:, None], axis=1) + 1
    true_score = scores[np.arange(len(labels)), local_true]
    wrong = scores.copy()
    wrong[np.arange(len(labels)), local_true] = -np.inf
    margins = true_score - wrong.max(axis=1)
    return {
        'rank': ranks, 'r1': (ranks == 1).astype(float),
        'r3': (ranks <= 3).astype(float), 'r5': (ranks <= 5).astype(float),
        'mrr': 1.0 / ranks, 'true_score': true_score,
        'margin': margins, 'order': candidates[order[:, :5]],
        'top_scores': np.take_along_axis(scores, order[:, :5], axis=1),
    }


class ClusterBootstrap:
    """Class clustering, with exact-copy a452/a456 in one connected cluster."""
    def __init__(self, labels, classes, n_boot=BOOTSTRAPS):
        labels = np.asarray(labels, dtype=np.int64)
        self.labels = labels
        self.present = np.unique(labels)
        keys = ['duplicate:a452+a456' if classes[label] in AMBIGUOUS else classes[label]
                for label in self.present]
        unique_keys = sorted(set(keys))
        class_to_cluster = {int(label): unique_keys.index(key) for label, key in zip(self.present, keys)}
        self.clusters = np.array([class_to_cluster[int(label)] for label in labels], dtype=np.int64)
        self.n_clusters = len(unique_keys)
        self.counts = np.bincount(self.clusters, minlength=self.n_clusters)
        rng = np.random.default_rng(BOOT_SEED)
        self.weights = rng.multinomial(self.n_clusters, np.full(self.n_clusters, 1 / self.n_clusters), size=n_boot)
        self.class_counts = np.bincount(labels, minlength=len(classes))
        self.class_cluster = np.array([class_to_cluster[int(label)] for label in self.present])

    def interval(self, per_query, macro=False):
        values = np.asarray(per_query, dtype=float)
        assert values.shape == self.labels.shape
        if macro:
            means = np.bincount(self.labels, weights=values, minlength=len(self.class_counts))[self.present] / self.class_counts[self.present]
            sums = np.bincount(self.class_cluster, weights=means, minlength=self.n_clusters)
            denominators = np.bincount(self.class_cluster, minlength=self.n_clusters)
        else:
            sums = np.bincount(self.clusters, weights=values, minlength=self.n_clusters)
            denominators = self.counts
        draws = (self.weights @ sums) / (self.weights @ denominators)
        return [float(x) for x in np.quantile(draws, [.025, .975])]


def metrics(details, labels):
    present = np.unique(labels)
    per_class = [details['r1'][np.asarray(labels) == label].mean() for label in present]
    return dict(recall1=100 * float(details['r1'].mean()),
                recall3=100 * float(details['r3'].mean()),
                recall5=100 * float(details['r5'].mean()),
                macro_recall1=100 * float(np.mean(per_class)), mrr=float(details['mrr'].mean()),
                true_cosine_or_logit=float(details['true_score'].mean()),
                true_margin=float(details['margin'].mean()))


def summary_metrics(study, method, runs, labels, classes, bootstrap):
    # Average outcomes, not score matrices/prototypes, across the fixed three runs.
    outcome = {key: np.mean([run[key] for run in runs], axis=0)
               for key in ['r1', 'r3', 'r5', 'mrr', 'true_score', 'margin']}
    summary = dict(study=study, method=method, n_classes=len(np.unique(labels)),
                   n_queries=len(labels), fixed_runs=len(runs), bootstrap_clusters=bootstrap.n_clusters,
                   **metrics(outcome, labels))
    for name, key in [('recall1', 'r1'), ('recall3', 'r3'), ('recall5', 'r5'), ('mrr', 'mrr')]:
        lo, hi = bootstrap.interval(outcome[key])
        factor = 1 if key == 'mrr' else 100
        summary[name + '_ci_low'] = factor * lo
        summary[name + '_ci_high'] = factor * hi
        run_values = [factor * float(run[key].mean()) for run in runs]
        summary[name + '_run_sd'] = float(np.std(run_values, ddof=1)) if len(runs) > 1 else 0.0
    lo, hi = bootstrap.interval(outcome['r1'], macro=True)
    summary['macro_recall1_ci_low'] = 100 * lo
    summary['macro_recall1_ci_high'] = 100 * hi
    return summary, outcome


PREDICTION_FIELDS = ['study', 'method', 'run', 'query_index', 'query_path', 'true_class',
                     'rank', 'top1', 'top2', 'top3', 'top4', 'top5', 'top1_score',
                     'true_score', 'true_margin', 'recall1', 'recall3', 'recall5']


class Predictions:
    def __init__(self, path):
        self.file = path.open('w', encoding='utf-8-sig', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=PREDICTION_FIELDS)
        self.writer.writeheader()
        self.count = 0

    def add(self, study, method, run, query_indices, rows, classes, details):
        for i, index in enumerate(query_indices):
            ranking = details['order'][i]
            record = dict(study=study, method=method, run=run, query_index=int(index),
                          query_path=rows[index]['relative_path'], true_class=rows[index]['class_name'],
                          rank=int(details['rank'][i]), top1_score=float(details['top_scores'][i, 0]),
                          true_score=float(details['true_score'][i]), true_margin=float(details['margin'][i]),
                          recall1=int(details['r1'][i]), recall3=int(details['r3'][i]), recall5=int(details['r5'][i]))
            for n, label in enumerate(ranking):
                record[f'top{n+1}'] = classes[label]
            self.writer.writerow(record)
            self.count += 1

    def close(self):
        self.file.close()


def load_inputs():
    protocol = read_json(OUT / 'protocol/experiment_protocol.json')
    assert protocol['version'] == 'T1K355_photo_disjoint_v1'
    assert hashlib.sha256((OUT/'protocol/photo_splits.json').read_bytes()).hexdigest() == protocol['photo_split_sha256']
    rows = read_json(OUT / 'protocol/photo_splits.json')
    classes = sorted({row['class_name'] for row in rows})
    labels = np.array([row['label'] for row in rows], dtype=np.int64)
    indices = {role: np.array([row['index'] for row in rows if row['role'] == role], dtype=np.int64)
               for role in ['reference', 'validation', 'test']}
    assert len(rows) == 1792 and len(classes) == 355
    assert {key: len(value) for key, value in indices.items()} == protocol['split']
    assert all(labels[indices[role]].tolist() == list(range(355)) for role in ['reference', 'validation'])
    for field in ['sha256', 'decoded_pixel_sha256']:
        by_hash = {}
        for row in rows:
            by_hash.setdefault(row[field], set()).add(row['role'])
        assert all(len(roles) == 1 for roles in by_hash.values())
    handcrafted = np.load(OUT / 'cache/handcrafted.npz')
    features = {key: np.asarray(handcrafted[key], dtype=np.float32) for key in ['rgb', 'gray', 'lbp']}
    for name in ['resnet50', 'dinov2', 'clip']:
        features[name] = np.load(OUT / f'cache/{name}_base.npy', mmap_mode='r')
    augmented = {name: np.load(OUT / f'cache/{name}_aug.npy', mmap_mode='r') for name in ['resnet50', 'dinov2']}
    for name, array in features.items():
        assert array.shape[0] == 1792 and np.all(np.isfinite(array)), name
        assert np.allclose(np.linalg.norm(array, axis=-1), 1, atol=2e-4), name
    for name, array in augmented.items():
        assert array.shape[:3] == (3, 355, 30) and np.all(np.isfinite(array)), name
        assert np.allclose(np.linalg.norm(array, axis=-1), 1, atol=2e-4), name
    # Require the complete adaptation matrix before exposing any primary test metric.
    adaptation = np.load(OUT / 'results/adaptation_scores.npz')
    selection = read_json(OUT / 'results/adaptation_selection.json')
    expected_selected = {(mode, seed) for mode in ['linear','partial','full'] for seed in SEEDS}
    assert isinstance(selection,list) and len(selection)==9
    assert {(row['mode'],row['seed']) for row in selection} == expected_selected
    for record in selection:
        assert 1 <= record['selected_epoch'] <= 10
        assert record['epochs_executed']==10 and record['presentations']==3550 and record['updates']==230
        assert len(record['checkpoint_sha256'])==64
    for mode in ['linear', 'partial', 'full']:
        for seed in SEEDS:
            for role in ['validation', 'test']:
                key = f'{mode}_{seed}_{role}'
                assert adaptation[key].shape == (len(indices[role]), 355), key
                assert np.all(np.isfinite(adaptation[key])), key
    for record in selection:
        score=adaptation[f"{record['mode']}_{record['seed']}_validation"]
        assert int(score_details(score,labels[indices['validation']])['r1'].sum()) == record['validation_correct']
    return protocol, rows, classes, labels, indices, features, augmented, adaptation, selection


def main_analysis(rows, classes, labels, ix, features, augmented, adaptation):
    output = OUT / 'results'
    scores_dir = OUT / 'predictions/scores'
    scores_dir.mkdir(parents=True, exist_ok=True)
    scores = {name + '_center': {0: {role: np.asarray(array[ix[role]]) @ np.asarray(array[ix['reference']]).T
                                    for role in ['validation', 'test']}}
              for name, array in features.items()}
    stability = []
    for name, array in augmented.items():
        for n in [1, 5, 10, 20, 30]:
            method = f'{name}_A{n}'
            scores[method] = {}
            prototypes = [normalize(array[seed_index, :, :n].mean(axis=1)) for seed_index in range(3)]
            angles = [np.degrees(np.arccos(np.clip(np.sum(prototypes[a] * prototypes[b], axis=1), -1, 1)))
                      for a, b in [(0, 1), (0, 2), (1, 2)]]
            for seed, prototype in zip(SEEDS, prototypes):
                scores[method][seed] = {role: np.asarray(features[name][ix[role]]) @ prototype.T
                                        for role in ['validation', 'test']}
            stability.append(dict(method=method, n_views=n, n_pools=3,
                                  mean_between_pool_angle_deg=float(np.mean(angles)),
                                  median_between_pool_angle_deg=float(np.median(angles)),
                                  p95_between_pool_angle_deg=float(np.quantile(angles, .95))))
    fusion_choices = []
    for hand in ['rgb', 'lbp']:
        candidates = []
        for alpha in [0, .1, .25, .5, .75, 1]:
            score = (1-alpha) * scores['dinov2_center'][0]['validation'] + alpha * scores[hand+'_center'][0]['validation']
            details = score_details(score, labels[ix['validation']])
            candidates.append(dict(component=hand, alpha=alpha, validation_correct=int(details['r1'].sum())))
        # Python max retains the first grid entry on ties, as locked in the protocol.
        chosen = max(candidates, key=lambda value: value['validation_correct'])
        fusion_choices.append(dict(**chosen, candidates=candidates, selection='validation only; first alpha on ties'))
        alpha = chosen['alpha']
        scores[f'dinov2_{hand}_fusion'] = {0: {
            role: (1-alpha) * scores['dinov2_center'][0][role] + alpha * scores[hand+'_center'][0][role]
            for role in ['validation', 'test']}}
    write_json(output / 'fusion_selection.json', fusion_choices)
    for mode in ['linear', 'partial', 'full']:
        scores['resnet50_' + mode] = {seed: {role: adaptation[f'{mode}_{seed}_{role}']
                                           for role in ['validation', 'test']} for seed in SEEDS}
    clean_candidates = np.array([index for index, name in enumerate(classes) if name not in AMBIGUOUS])
    clean_mask = ~np.isin(labels[ix['test']], [classes.index(name) for name in AMBIGUOUS])
    bootstraps = {'primary355': ClusterBootstrap(labels[ix['test']], classes),
                  'clean353': ClusterBootstrap(labels[ix['test']][clean_mask], classes)}
    summary, run_metrics, outcomes = [], [], {}
    prediction_sink = Predictions(OUT / 'predictions/main_query_predictions.csv')
    try:
        for method, method_runs in scores.items():
            packed = {'query_indices': ix['test'], 'query_labels': labels[ix['test']], 'classes': np.array(classes)}
            for run, values in method_runs.items():
                packed[f'test_{run}'] = values['test']
                packed[f'validation_{run}'] = values['validation']
            np.savez_compressed(scores_dir / f'{method}.npz', **packed)
            for study in ['primary355', 'clean353']:
                y = labels[ix['test']] if study == 'primary355' else labels[ix['test']][clean_mask]
                runs = []
                for run, values in method_runs.items():
                    matrix = values['test'] if study == 'primary355' else values['test'][clean_mask][:, clean_candidates]
                    candidates = None if study == 'primary355' else clean_candidates
                    details = score_details(matrix, y, candidates)
                    runs.append(details)
                    run_metrics.append(dict(study=study, method=method, run=run, n_queries=len(y), **metrics(details, y)))
                    if study == 'primary355':
                        prediction_sink.add(study, method, run, ix['test'], rows, classes, details)
                record, averaged = summary_metrics(study, method, runs, y, classes, bootstraps[study])
                query_indices = ix['test'] if study=='primary355' else ix['test'][clean_mask]
                record['unique_query_source_photos'] = len({rows[index]['sha256'] for index in query_indices})
                summary.append(record)
                outcomes[(study, method)] = averaged
            # Validation scores remain inspectable, separate from test rows.
            for run, values in method_runs.items():
                details = score_details(values['validation'], labels[ix['validation']])
                run_metrics.append(dict(study='validation355', method=method, run=run, n_queries=355,
                                        **metrics(details, labels[ix['validation']])) )
            print('scored', method, flush=True)
    finally:
        prediction_sink.close()
    paired = []
    comparisons = [('resnet50_A30', 'resnet50_center'), ('dinov2_A30', 'dinov2_center')]
    comparisons += [('resnet50_' + mode, 'resnet50_center') for mode in ['linear', 'partial', 'full']]
    comparisons += [('resnet50_center', 'rgb_center'), ('dinov2_center', 'rgb_center'), ('clip_center', 'rgb_center')]
    for study in ['primary355', 'clean353']:
        for treatment, baseline in comparisons:
            difference = outcomes[(study, treatment)]['r1'] - outcomes[(study, baseline)]['r1']
            lo, hi = bootstraps[study].interval(difference)
            paired.append(dict(study=study, treatment=treatment, baseline=baseline,
                               n_queries=len(difference), bootstrap_clusters=bootstraps[study].n_clusters,
                               recall1_difference_pp=100*float(difference.mean()), ci_low_pp=100*lo, ci_high_pp=100*hi))
    for record in stability:
        values = list(scores[record['method']].values())
        hits = np.stack([score_details(value['test'], labels[ix['test']])['r1'] for value in values])
        record['queries_correct_in_all_three_pools_pct'] = 100*float(hits.all(axis=0).mean())
        record['meaning'] = 'Reference construction variation for the same real queries; not repeated query capture robustness'
    write_csv(output / 'main_metrics.csv', summary)
    write_json(output / 'main_metrics.json', summary)
    write_csv(output / 'per_run_metrics.csv', run_metrics)
    write_csv(output / 'paired_recall1_differences.csv', paired)
    write_csv(output / 'reference_pool_stability.csv', stability)
    class_rows = []
    for (study, method), outcome in outcomes.items():
        if study != 'primary355':
            continue
        y = labels[ix['test']]
        for label, class_name in enumerate(classes):
            mask = y == label
            class_rows.append(dict(method=method, class_name=class_name, n_queries=int(mask.sum()),
                                   ambiguous_duplicate_label=class_name in AMBIGUOUS,
                                   recall1=100*float(outcome['r1'][mask].mean()),
                                   recall3=100*float(outcome['r3'][mask].mean()),
                                   recall5=100*float(outcome['r5'][mask].mean()),
                                   mrr=float(outcome['mrr'][mask].mean()),
                                   true_margin=float(outcome['margin'][mask].mean())))
    write_csv(output / 'per_class_metrics.csv', class_rows)
    return summary, paired, stability, prediction_sink.count


def reference_and_budget_sensitivity(rows, classes, labels, ix, features):
    summaries, paired = [], []
    alternative = read_json(OUT / 'protocol/reference_sensitivity.json')
    alternate_metrics = []
    for plan in alternative:
        ri, qi = np.array(plan['reference']), np.array(plan['query'])
        assert labels[ri].tolist() == list(range(355))
        for name, feature in features.items():
            matrix = np.asarray(feature[qi]) @ np.asarray(feature[ri]).T
            details = score_details(matrix, labels[qi])
            alternate_metrics.append(dict(method=name+'_center', gallery_seed=plan['seed'],
                                          n_queries=len(qi), **metrics(details, labels[qi])))
    qi = np.array([row['index'] for row in rows if row['paired_two_reference_role'] == 'test'])
    second = np.array([row['index'] for row in rows if row['within_class_order'] == 2])
    assert len(qi) == 727 and labels[second].tolist() == list(range(355))
    bootstrap = ClusterBootstrap(labels[qi], classes)
    sink = Predictions(OUT / 'predictions/reference_budget_predictions.csv')
    try:
        for name, feature in features.items():
            one = np.asarray(feature[ix['reference']])
            two = normalize(one + np.asarray(feature[second]))
            outcomes = []
            for budget, proto in [(1, one), (2, two)]:
                details = score_details(np.asarray(feature[qi]) @ proto.T, labels[qi])
                record, outcome = summary_metrics('paired_real_reference_budget', name+f'_R{budget}', [details], labels[qi], classes, bootstrap)
                record['real_references_per_class'] = budget
                record['unique_query_source_photos'] = len({rows[index]['sha256'] for index in qi})
                summaries.append(record)
                outcomes.append(outcome)
                sink.add('paired_real_reference_budget', name, budget, qi, rows, classes, details)
            difference = outcomes[1]['r1'] - outcomes[0]['r1']
            lo, hi = bootstrap.interval(difference)
            paired.append(dict(method=name, n_queries=len(qi), bootstrap_clusters=bootstrap.n_clusters,
                               recall1_two_minus_one_pp=100*float(difference.mean()), ci_low_pp=100*lo, ci_high_pp=100*hi))
    finally:
        sink.close()
    write_csv(OUT / 'results/reference_choice_sensitivity.csv', alternate_metrics)
    write_csv(OUT / 'results/real_reference_budget.csv', summaries)
    write_csv(OUT / 'results/real_reference_budget_paired.csv', paired)
    return summaries, paired, alternate_metrics


def roc_summary(positive, negative):
    """Positive means a known-instance query; larger score means accept as known."""
    positive, negative = np.asarray(positive), np.asarray(negative)
    scores = np.concatenate([positive, negative])
    truth = np.concatenate([np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int)])
    order = np.argsort(-scores, kind='stable')
    sorted_scores, sorted_truth = scores[order], truth[order]
    last_in_tie = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores)-1]
    tp = np.r_[0, np.cumsum(sorted_truth)[last_in_tie]] / len(positive)
    fp = np.r_[0, np.cumsum(1-sorted_truth)[last_in_tie]] / len(negative)
    auroc = float(np.trapezoid(tp, fp))
    precision = np.divide(tp*len(positive), tp*len(positive)+fp*len(negative),
                          out=np.ones_like(tp), where=(tp+fp)>0)
    average_precision = float(np.sum(np.diff(tp) * precision[1:]))
    return dict(auroc=auroc, known_average_precision=average_precision,
                unknown_fpr_at_known_tpr95=float(fp[np.flatnonzero(tp >= .95)[0]]))


def choose_threshold(known, unknown):
    known, unknown = np.asarray(known), np.asarray(unknown)
    thresholds = np.r_[np.unique(np.concatenate([known, unknown])),
                       np.nextafter(max(known.max(), unknown.max()), np.inf)]
    balanced = .5*((known[:, None] >= thresholds).mean(axis=0) + (unknown[:, None] < thresholds).mean(axis=0))
    best = np.flatnonzero(balanced == balanced.max())[-1]
    return float(thresholds[best]), float(balanced[best])


def open_set_analysis(rows, classes, labels, ix, features):
    plan = read_json(OUT / 'protocol/open_set_class_splits.json')
    known = np.sort([classes.index(name) for name in plan['known']])
    cal_unknown = [classes.index(name) for name in plan['calibration_unknown']]
    test_unknown = [classes.index(name) for name in plan['test_unknown']]
    assert len(known) == 282 and len(cal_unknown) == 35 and len(test_unknown) == 36
    assert not (set(known) & set(cal_unknown) or set(known) & set(test_unknown) or set(cal_unknown) & set(test_unknown))
    role_indices = {
        'known_calibration': ix['validation'][np.isin(labels[ix['validation']], known)],
        'unknown_calibration': np.flatnonzero(np.isin(labels, cal_unknown)),
        'known_test': ix['test'][np.isin(labels[ix['test']], known)],
        'unknown_test': np.flatnonzero(np.isin(labels, test_unknown)),
    }
    ri = ix['reference'][known]
    results, predictions = [], []
    for name, feature in features.items():
        measures, correct = {}, {}
        for role, qi in role_indices.items():
            matrix = np.asarray(feature[qi]) @ np.asarray(feature[ri]).T
            order = np.argsort(-matrix, axis=1, kind='stable')
            best = matrix[np.arange(len(qi)), order[:, 0]]
            second = matrix[np.arange(len(qi)), order[:, 1]]
            measures[role] = {'maximum_cosine': best, 'top1_top2_margin': best-second}
            correct[role] = known[order[:, 0]] == labels[qi]
        for detector in ['maximum_cosine', 'top1_top2_margin']:
            threshold, balanced = choose_threshold(measures['known_calibration'][detector], measures['unknown_calibration'][detector])
            known_scores = measures['known_test'][detector]
            unknown_scores = measures['unknown_test'][detector]
            accept_known = known_scores >= threshold
            reject_unknown = unknown_scores < threshold
            results.append(dict(method=name+'_center', detector=detector, threshold=threshold,
                                calibration_balanced_accuracy=balanced,
                                known_classes=len(known), calibration_unknown_classes=len(cal_unknown),
                                test_unknown_classes=len(test_unknown), known_test_queries=len(known_scores),
                                unknown_test_queries=len(unknown_scores),
                                known_accept_rate=float(accept_known.mean()),
                                unknown_reject_rate=float(reject_unknown.mean()),
                                test_balanced_accuracy=.5*float(accept_known.mean()+reject_unknown.mean()),
                                known_identification_and_acceptance=float((accept_known & correct['known_test']).mean()),
                                **roc_summary(known_scores, unknown_scores)))
            for role, qi in role_indices.items():
                value = measures[role][detector]
                predictions.extend(dict(method=name+'_center', detector=detector, role=role,
                                        query_index=int(index), query_path=rows[index]['relative_path'],
                                        true_class=rows[index]['class_name'], score=float(score),
                                        threshold=threshold, accepted_as_known=bool(score>=threshold))
                                   for index, score in zip(qi, value))
    write_csv(OUT / 'results/open_set_metrics.csv', results)
    write_csv(OUT / 'predictions/open_set_scores.csv', predictions)
    return results


def report(protocol, summary, paired, stability, budget, budget_paired, reference_choice, open_results, selection):
    primary = {row['method']: row for row in summary if row['study']=='primary355'}
    clean = {row['method']: row for row in summary if row['study']=='clean353'}
    aliases = {'rgb_center':'RGB直方图','gray_center':'灰度直方图','lbp_center':'LBP',
               'resnet50_center':'冻结ResNet50','dinov2_center':'冻结DINOv2','clip_center':'冻结CLIP',
               'resnet50_A30':'ResNet50 A30','dinov2_A30':'DINOv2 A30',
               'resnet50_linear':'ResNet50线性头训练','resnet50_partial':'ResNet50末阶段微调',
               'resnet50_full':'ResNet50全量微调','dinov2_rgb_fusion':'DINOv2＋RGB融合','dinov2_lbp_fusion':'DINOv2＋LBP融合'}
    shown = ['rgb_center','gray_center','lbp_center','resnet50_center','dinov2_center','clip_center',
             'resnet50_A30','dinov2_A30','resnet50_linear','resnet50_partial','resnet50_full',
             'dinov2_rgb_fusion','dinov2_lbp_fusion']
    lines = ['# T1K+ Fabric 训练与测试结果', '',
             '## Material Passport', '',
             '- Origin Skill: academic-research-suite / experiment-agent',
             '- Origin Mode: run', '- Origin Date: 2026-10-29',
             '- Verification Status: ANALYZED（计算结果已生成；不等于独立重训复现）',
             '- Version Label: T1K355_photo_disjoint_v1', '',
             '本报告仅交付新增数据实验结果，未修改论文、提交材料或上传原始数据。', '',
             '## 数据与划分', '',
             '官方 Fabric 子集共有355个编号、1792个照片文件。每类固定1张原始照片作参考、1张不同原始照片作验证，剩余1082张作测试。所有裁剪/增强均在整张原图划分后生成；图库始终包含全部355个官方编号。', '',
             '**数据审计发现：a452与a456的5张照片逐张字节相同。** 因此实际只有1787张不同照片文件。这两个编号保留在355类严格编号主结果中，并同步照片集合角色，防止相同照片跨入参考、验证或测试。另给出排除这两个歧义编号及候选的353类敏感性结果；这只是同一模型的评价范围变化，训练仍使用355类，未据此宣称身份标签已纠正。', '',
             '**来源照片数与带标签行数不同：**参考、验证、测试分别为355/355/1082行，实际不同原始照片分别为354/354/1079张。主指标按官方标签行计分，但推断时将重复编号对合为同一来源簇，不能把1082行全称为独立照片。', '',
             '353类评价中的测试照片数为'+str(clean['rgb_center']['n_queries'])+'。原始文件和EXIF校验不等于跨设备、跨会话独立采集认证；本实验是彩色布料实例检索外部验证，不能直接替代原203类皮革的补拍与来源证明。', '',
             '## 主要结果', '',
             'R@1/3/5以百分比表示，MRR为0–1。区间为2000次来源身份簇重采样；a452/a456合为同一个推断簇，所以355标签主分析使用354个簇。三个固定参考池或训练种子的结果先在每张查询内平均，再重采样；运行间SD另列，不能把三个种子当成三批新照片。', '',
             '| 方法 | 355类 R@1 [95% CI] | R@3 | R@5 | MRR | 运行间R@1 SD | 353类 R@1 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for method in shown:
        row = primary[method]
        lines.append(f"| {aliases[method]} | {row['recall1']:.2f} [{row['recall1_ci_low']:.2f}, {row['recall1_ci_high']:.2f}] | {row['recall3']:.2f} | {row['recall5']:.2f} | {row['mrr']:.4f} | {row['recall1_run_sd']:.2f} | {clean[method]['recall1']:.2f} |")
    lines += ['', '完整21种方法与所有N值见 `main_metrics.csv`；各运行与验证集结果见 `per_run_metrics.csv`。单个查询的预测、真实编号排名和候选分数见 `../predictions/main_query_predictions.csv`，完整分数矩阵见 `../predictions/scores/`。', '',
              '## 配对差异', '',
              '以下差值使用相同1082张测试照片，单位为百分点。区间条件于现有照片、已冻结设置和固定运行，不表示跨会话或更大训练预算下的普遍差异；区间跨零不代表模型等价。多项比较为探索性描述，未进行家族错误率控制。', '',
              '| 比较 | R@1差值 | 配对95% CI |', '|---|---:|---:|']
    for row in paired:
        if row['study']=='primary355':
            lines.append(f"| {aliases.get(row['treatment'],row['treatment'])} − {aliases.get(row['baseline'],row['baseline'])} | {row['recall1_difference_pp']:+.2f} | [{row['ci_low_pp']:+.2f}, {row['ci_high_pp']:+.2f}] |")
    lines += ['', '## 训练与模型选择', '',
              'ResNet50三种范围均执行10轮、3550次图像呈现、230次更新，批大小16；每轮每类只使用该参考原图的一个预先缓存增强视图。BatchNorm运行统计固定，余弦头由中心参考特征初始化、尺度20。线性头学习率候选为0.001/0.01，局部或全量微调为0.00001/0.0001，微调的头学习率为骨干的10倍；Adam权重衰减0.0001。种子42仅用验证集选择学习率和最早最佳轮次，种子43/44按所选学习率独立训练、验证选择轮次。测试集不参与这些选择。', '',
              '因此训练结果回答的是此明确预算下的适配效果，不是充分调优后的最优迁移能力。末阶段微调复用float32冻结layer3输出，只更新layer4和头；完整训练保留全部骨干梯度。所选配置见 `adaptation_selection.json`，逐轮历史及实际检查点由训练脚本保存。', '',
              '| 训练范围 | 种子 | 所选学习率 | 所选轮次 | 验证命中/355 |',
              '|---|---:|---:|---:|---:|']
    for record in selection:
        lines.append(f"| {record['mode']} | {record['seed']} | {record['lr']:g} | {record['selected_epoch']} | {record['validation_correct']} |")
    lines += ['',
              'DINOv2＋RGB/LBP融合的权重在预定网格中仅按355张验证照片选择。融合权重及全部候选的验证成绩见 `fusion_selection.json`。', '',
              '## 参考平均与真实参考数量', '',
              'N=1、5、10、20、30是同一参考原图的数字视图，不是新增真实照片。随机增强N=1与中心裁剪冻结基线不同。`reference_pool_stability.csv`报告池间角度及三池同时命中率，不能把构建更稳定写成所有查询必然更可靠。', '',
              '| 方法 | 随机N1池间平均角度 | N30池间平均角度 | N30三池全命中率 |', '|---|---:|---:|---:|']
    stab = {row['method']:row for row in stability}
    for name in ['resnet50','dinov2']:
        one, thirty = stab[name+'_A1'], stab[name+'_A30']
        lines.append(f"| {name} | {one['mean_between_pool_angle_deg']:.2f}° | {thirty['mean_between_pool_angle_deg']:.2f}° | {thirty['queries_correct_in_all_three_pools_pct']:.2f}% |")
    lines += ['', '1张与2张真实参考的比较使用相同727张测试照片；第二张参考在比较前从原测试池移出，避免两种预算使用不同查询集。', '',
              '| 冻结方法 | 2参考−1参考 R@1 | 配对95% CI |', '|---|---:|---:|']
    for row in budget_paired:
        lines.append(f"| {FROZEN_NAMES[row['method']]} | {row['recall1_two_minus_one_pp']:+.2f} | [{row['ci_low_pp']:+.2f}, {row['ci_high_pp']:+.2f}] |")
    lines += ['', '五种替代图库的结果见 `reference_choice_sensitivity.csv`。这些划分重用同一批照片，属于参考选择的描述性敏感性，不能作为五个独立数据集或会话。', '',
              '## 未知实例拒识诊断', '',
              '排除重复标签对后，将353个编号分为282个已知、35个校准未知及36个测试未知。只评价未在这些未知编号上训练或调融合权重的冻结基础描述符；每种阈值仅由已知验证照片＋校准未知照片选择，再评价未触及的已知测试照片与测试未知编号的全部照片。', '',
              '| 冻结方法（最大余弦） | AUROC | 已知接受率 | 未知拒识率 | 测试平衡准确率 | FPR@95%已知TPR |', '|---|---:|---:|---:|---:|---:|']
    for row in open_results:
        if row['detector']=='maximum_cosine':
            lines.append(f"| {FROZEN_NAMES[row['method'].replace('_center','')]} | {row['auroc']:.4f} | {100*row['known_accept_rate']:.2f}% | {100*row['unknown_reject_rate']:.2f}% | {100*row['test_balanced_accuracy']:.2f}% | {100*row['unknown_fpr_at_known_tpr95']:.2f}% |")
    lines += ['', '最大余弦与前两候选间隔均逐项报告在 `open_set_metrics.csv`；没有按测试成绩挑选检测器。这里的未知是公开Fabric实例，不是专家验证的相近未知皮革；不能由此宣称工厂开放集拒识已解决。', '',
              '## 结果边界与可复现文件', '',
              '- 完全相同文件与像素检查已做；近重复、材料同源关系以及跨设备/会话记录仍不完整。',
              '- 外部Fabric原图多查询可补充真实照片变化证据，但不会修复原203类皮革的来源缺口。',
              '- a452/a456在主结果中仍是严格官方编号，不能同时把它们解释成已经验证的两个独立视觉身份。',
              '- 三种子SD与查询/来源簇区间描述不同变异；没有把1792张文件或增强后代都当作独立采集。',
              '- 结果来自单台CPU、固定预训练权重和有限训练预算；未声称SOTA、真实跨会话鲁棒性或部署效果。',
              '- 协议、逐张照片角色及输入哈希位于 `../protocol/` 与 `../audit/`；本脚本可从已保存特征和选定模型分数重新计算表格。',
              '- 当前交付仅为数据实验结果；没有修改论文，也没有将这些结果自动写入稿件。', '']
    (OUT / 'results/实验结果报告.md').write_text('\n'.join(lines), encoding='utf-8')


def self_test():
    classes = ['a452','a456','x','y']
    scores = np.array([[.9,.9,.2,.1], [.9,.9,.2,.1], [.1,.2,.8,.3], [.1,.2,.3,.7]])
    y = np.arange(4)
    details = score_details(scores, y)
    assert details['rank'].tolist() == [1,2,1,1]
    assert metrics(details,y)['recall1'] == 75
    bootstrap = ClusterBootstrap(y,classes,n_boot=100)
    assert bootstrap.n_clusters == 3
    assert bootstrap.clusters[0] == bootstrap.clusters[1]
    assert bootstrap.interval(np.ones(4)) == [1,1]
    assert bootstrap.interval(np.zeros(4)) == [0,0]
    clean = score_details(scores[2:,2:], y[2:], np.array([2,3]))
    assert clean['rank'].tolist() == [1,1]
    threshold, accuracy = choose_threshold(np.array([.8,.9]), np.array([.2,.3]))
    assert threshold == .8 and accuracy == 1
    assert roc_summary([.8,.9],[.2,.3])['auroc'] == 1
    assert roc_summary([.1,.2],[.8,.9])['auroc'] == 0
    assert roc_summary([.5,.5],[.5,.5])['auroc'] == .5
    assert roc_summary([.5,.5],[.5,.5])['unknown_fpr_at_known_tpr95'] == 1
    print('PASS: stable ID ties, duplicate-aware clustering, clean353 candidate remapping, bootstrap constants, calibration-only threshold, tied ROC.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    started = time.time()
    for directory in ['results','predictions']:
        (OUT/directory).mkdir(exist_ok=True)
    inputs = load_inputs()
    protocol, rows, classes, labels, ix, features, augmented, adaptation, selection = inputs
    summary, paired, stability, predictions = main_analysis(rows,classes,labels,ix,features,augmented,adaptation)
    budget, budget_paired, reference_choice = reference_and_budget_sensitivity(rows,classes,labels,ix,features)
    open_results = open_set_analysis(rows,classes,labels,ix,features)
    report(protocol,summary,paired,stability,budget,budget_paired,reference_choice,open_results,selection)
    result = dict(status='completed', evaluation_only=True, methods=21, main_summary_rows=len(summary),
                  per_query_prediction_rows=predictions, elapsed_seconds=time.time()-started,
                  primary_classes=355, primary_queries=1082, primary_bootstrap_clusters=354,
                  primary_unique_source_queries=1079,
                  clean_classes=353, protocol_sha256=hashlib.sha256((OUT/'protocol/experiment_protocol.json').read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(OUT/'results/evaluation_complete.json', result)
    print('COMPLETE EVALUATION', json.dumps(result), flush=True)


if __name__ == '__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
