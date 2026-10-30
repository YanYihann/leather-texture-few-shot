"""Independent replay of auxiliary results from saved features; no model execution.

Uses independent ranking/bootstrap code in verify_results.py and never imports
evaluate_results.py. Does not edit result tables or prediction files.
"""
import collections
import json
from pathlib import Path
import sys
import time
import numpy as np
from verify_results import OUT, AMBIGUOUS, SEEDS, read_json, read_csv, hash_file, close, ranking, metrics, bootstrap_values, normalize


def verify_metric_row(record, detail, labels):
    assert int(record['n_queries']) == len(labels)
    for key,value in metrics(detail,labels).items():
        close(float(record[key]),value,key)


def interval(values, labels, classes):
    keys=['duplicate:a452+a456' if classes[int(label)] in AMBIGUOUS else classes[int(label)] for label in labels]
    unique=sorted(set(keys));cluster=np.array([unique.index(key) for key in keys])
    counts=np.bincount(cluster,minlength=len(unique));sums=np.bincount(cluster,weights=values,minlength=len(unique))
    draws=np.random.default_rng(20261029).multinomial(len(unique),np.full(len(unique),1/len(unique)),size=2000)
    return 100*np.quantile((draws@sums)/(draws@counts),[.025,.975])


def verify_reference_choice(rows, classes, labels, indices, features):
    actual={(r['method'],int(r['gallery_seed'])):r for r in read_csv(OUT/'results/reference_choice_sensitivity.csv')}
    plans=read_json(OUT/'protocol/reference_sensitivity.json');assert len(plans)==5 and len(actual)==30
    for plan in plans:
        ri,qi=np.array(plan['reference']),np.array(plan['query'])
        assert len(ri)==355 and len(qi)==1437
        for field in ['sha256','decoded_pixel_sha256']:
            assert {rows[int(i)][field] for i in ri}.isdisjoint({rows[int(i)][field] for i in qi})
        for name,feature in features.items():
            scores=np.asarray(feature[qi])@np.asarray(feature[ri]).T
            out=ranking(scores,labels[qi],np.arange(355))
            verify_metric_row(actual[(name+'_center',plan['seed'])],out,labels[qi])
    return {'six_methods_five_galleries':30,'all_metrics_recomputed':True,'exact_hash_support_query_overlap':0}


def verify_budget(rows, classes, labels, indices, features):
    actual={r['method']:r for r in read_csv(OUT/'results/real_reference_budget.csv')}
    paired={r['method']:r for r in read_csv(OUT/'results/real_reference_budget_paired.csv')}
    predictions=collections.defaultdict(list)
    for row in read_csv(OUT/'predictions/reference_budget_predictions.csv'):
        predictions[(row['method'],int(row['run']))].append(row)
    qi=np.array([r['index'] for r in rows if r['paired_two_reference_role']=='test'])
    second=np.array([r['index'] for r in rows if r['within_class_order']==2])
    assert len(qi)==727 and len(second)==355 and np.array_equal(labels[second],np.arange(355))
    assert len(actual)==12 and len(paired)==6 and len(predictions)==12
    for field in ['sha256','decoded_pixel_sha256']:
        assert {rows[int(i)][field] for i in np.r_[indices['reference'],second]}.isdisjoint({rows[int(i)][field] for i in qi})
    for name,feature in features.items():
        first=np.asarray(feature[indices['reference']]);both=normalize(first+np.asarray(feature[second]))
        outcomes=[]
        for budget,proto in [(1,first),(2,both)]:
            detail=ranking(np.asarray(feature[qi])@proto.T,labels[qi],np.arange(355));outcomes.append(detail)
            record=actual[f'{name}_R{budget}'];verify_metric_row(record,detail,labels[qi])
            assert int(record['n_classes'])==355 and int(record['bootstrap_clusters'])==354
            assert int(record['real_references_per_class'])==budget and int(record['fixed_runs'])==1
            assert int(record['unique_query_source_photos'])==len({rows[int(i)]['sha256'] for i in qi})==725
            _,ci=bootstrap_values(detail,labels[qi],classes)
            for key,value in ci.items():close(float(record[key]),float(value),key)
            for key in ['recall1_run_sd','recall3_run_sd','recall5_run_sd','mrr_run_sd']:assert float(record[key])==0
            pred=predictions[(name,budget)];assert len(pred)==727
            for i,row in enumerate(pred):
                assert int(row['query_index'])==qi[i] and int(row['rank'])==detail['rank'][i]
                assert row['query_path']==rows[int(qi[i])]['relative_path'] and row['true_class']==classes[int(labels[qi[i]])]
                for k in range(5):assert row[f'top{k+1}']==classes[int(detail['top'][i,k])]
                close(float(row['true_score']),float(detail['true_score'][i]),'budget true score')
                close(float(row['true_margin']),float(detail['margin'][i]),'budget true margin')
        difference=outcomes[1]['r1']-outcomes[0]['r1'];lo,hi=interval(difference,labels[qi],classes)
        record=paired[name];assert int(record['n_queries'])==727 and int(record['bootstrap_clusters'])==354
        close(float(record['recall1_two_minus_one_pp']),100*float(difference.mean()),'budget difference')
        close(float(record['ci_low_pp']),float(lo),'budget difference CI low');close(float(record['ci_high_pp']),float(hi),'budget difference CI high')
    return {'same_query_count':727,'unique_query_photos':725,'methods':6,'budget_rows':12,
            'all_query_predictions_checked':12*727,'paired_cluster_intervals':6,'exact_reference_query_overlap':0}


def verify_pool_stability(rows, classes, labels, indices, features):
    actual={r['method']:r for r in read_csv(OUT/'results/reference_pool_stability.csv')};assert len(actual)==10
    for name in ['resnet50','dinov2']:
        aug=np.load(OUT/f'cache/{name}_aug.npy',mmap_mode='r')
        for n in [1,5,10,20,30]:
            protos=[normalize(aug[si,:,:n].mean(axis=1)) for si in range(3)]
            # All 3 unordered pool pairs, each containing all355 nominal class labels.
            angles=np.stack([np.rad2deg(np.arccos(np.clip(np.sum(protos[a].astype(np.float64)*protos[b].astype(np.float64),axis=1),-1,1)))
                             for a,b in [(0,1),(0,2),(1,2)]])
            record=actual[f'{name}_A{n}'];assert int(record['n_views'])==n and int(record['n_pools'])==3
            # Float64 reduction gives an independent numeric reference; original
            # float32 dot products may shift angles slightly near cosine one.
            close(float(record['mean_between_pool_angle_deg']),float(angles.mean()),'pool mean angle',atol=2e-4)
            close(float(record['median_between_pool_angle_deg']),float(np.median(angles)),'pool median angle',atol=2e-4)
            close(float(record['p95_between_pool_angle_deg']),float(np.quantile(angles,.95)),'pool p95 angle',atol=2e-4)
            hit=[]
            for proto in protos:
                out=ranking(np.asarray(features[name][indices['test']])@proto.T,labels[indices['test']],np.arange(355))
                hit.append(out['r1'])
            all_hit=100*np.all(np.stack(hit),axis=0).mean()
            close(float(record['queries_correct_in_all_three_pools_pct']),float(all_hit),'three-pool all-query consistency')
    return {'models':2,'nested_reference_N':[1,5,10,20,30],'all_pair_angles_and_consistency_replayed':True,
            'angle_absolute_tolerance_deg':.0002}


def choose_threshold(known,unknown):
    # Direct scalar sweep independently of evaluator broadcasted threshold code.
    candidates=sorted(set(known.tolist()+unknown.tolist()))
    candidates.append(float(np.nextafter(max(candidates),np.inf)))
    best=-1.;chosen=None
    for threshold in candidates:
        balanced=.5*(np.count_nonzero(known>=threshold)/len(known)+np.count_nonzero(unknown<threshold)/len(unknown))
        if balanced>=best:best=balanced;chosen=threshold
    return chosen,best


def roc(known,unknown):
    # Pairwise Mann-Whitney AUROC counts tied known/unknown pairs as half a win.
    auc=float(((known[:,None]>unknown[None,:])+.5*(known[:,None]==unknown[None,:])).mean())
    thresholds=sorted(set(known.tolist()+unknown.tolist()),reverse=True)
    prior_tpr=0.;ap=0.;fpr95=None
    for threshold in thresholds:
        tp=int(np.count_nonzero(known>=threshold));fp=int(np.count_nonzero(unknown>=threshold));tpr=tp/len(known)
        ap+=(tpr-prior_tpr)*(tp/(tp+fp));prior_tpr=tpr
        if fpr95 is None and tpr>=.95:fpr95=fp/len(unknown)
    return {'auroc':auc,'known_average_precision':ap,'unknown_fpr_at_known_tpr95':fpr95}


def verify_open(rows,classes,labels,indices,features):
    plan=read_json(OUT/'protocol/open_set_class_splits.json')
    known=np.sort([classes.index(x) for x in plan['known']]);cal={classes.index(x) for x in plan['calibration_unknown']};unknown={classes.index(x) for x in plan['test_unknown']}
    assert (len(known),len(cal),len(unknown))==(282,35,36)
    assert set(known).isdisjoint(cal) and set(known).isdisjoint(unknown) and cal.isdisjoint(unknown)
    assert not AMBIGUOUS & (set(plan['known'])|set(plan['calibration_unknown'])|set(plan['test_unknown']))
    roles={'known_calibration':indices['validation'][np.isin(labels[indices['validation']],known)],
           'unknown_calibration':np.flatnonzero(np.isin(labels,list(cal))),
           'known_test':indices['test'][np.isin(labels[indices['test']],known)],
           'unknown_test':np.flatnonzero(np.isin(labels,list(unknown)))}
    actual={(x['method'],x['detector']):x for x in read_csv(OUT/'results/open_set_metrics.csv')};assert len(actual)==12
    predictions=collections.defaultdict(list)
    for row in read_csv(OUT/'predictions/open_set_scores.csv'):predictions[(row['method'],row['detector'],row['role'])].append(row)
    assert len(predictions)==48
    for name,feature in features.items():
        scores={};correct={}
        for role,qi in roles.items():
            matrix=np.asarray(feature[qi])@np.asarray(feature[indices['reference'][known]]).T
            order=np.argsort(-matrix,axis=1,kind='stable');maxima=matrix[np.arange(len(qi)),order[:,0]]
            scores[role]={'maximum_cosine':maxima,'top1_top2_margin':maxima-matrix[np.arange(len(qi)),order[:,1]]}
            correct[role]=known[order[:,0]]==labels[qi]
        for detector in ['maximum_cosine','top1_top2_margin']:
            threshold,balance=choose_threshold(scores['known_calibration'][detector],scores['unknown_calibration'][detector])
            record=actual[(name+'_center',detector)];close(float(record['threshold']),float(threshold),'open threshold')
            close(float(record['calibration_balanced_accuracy']),float(balance),'open calibration balanced accuracy')
            ks,us=scores['known_test'][detector],scores['unknown_test'][detector]
            accept=ks>=threshold;reject=us<threshold
            assert int(record['known_test_queries'])==len(ks)==855 and int(record['unknown_test_queries'])==len(us)==181
            expected={'known_accept_rate':accept.mean(),'unknown_reject_rate':reject.mean(),
                      'test_balanced_accuracy':.5*(accept.mean()+reject.mean()),
                      'known_identification_and_acceptance':(accept&correct['known_test']).mean(),**roc(ks,us)}
            for key,value in expected.items():close(float(record[key]),float(value),key)
            for role,qi in roles.items():
                saved=predictions[(name+'_center',detector,role)];assert len(saved)==len(qi)
                for i,row in enumerate(saved):
                    assert int(row['query_index'])==qi[i] and row['query_path']==rows[int(qi[i])]['relative_path']
                    assert row['true_class']==classes[int(labels[qi[i]])]
                    close(float(row['score']),float(scores[role][detector][i]),'raw open-set score')
                    close(float(row['threshold']),float(threshold),'raw open-set threshold')
                    assert (row['accepted_as_known']=='True')==bool(scores[role][detector][i]>=threshold)
    # Test explicit tiedROC corner cases independently of real outcomes.
    close(roc(np.array([.5,.5]),np.array([.5,.5]))['auroc'],.5,'tied ROC')
    assert roc(np.array([.5,.5]),np.array([.5,.5]))['unknown_fpr_at_known_tpr95']==1.
    return {'methods':6,'detectors_per_method':2,'thresholds_checked':12,
            'calibration_only_threshold_choice':True,'raw_open_set_rows_checked':sum(map(len,predictions.values())),
            'known_test_queries':855,'unknown_test_queries':181,'pairwise_tie_aware_AUROC':True,'AP_and_FPR95_replayed':True}


def main():
    started=time.time();done=read_json(OUT/'results/evaluation_complete.json');assert done['status']=='completed'
    assert read_json(OUT/'audit/verification.json')['status']=='passed'
    unchanged_paths=list((OUT/'results').glob('*'))+[OUT/'predictions/reference_budget_predictions.csv',OUT/'predictions/open_set_scores.csv']
    before={str(path.relative_to(OUT)):hash_file(path) for path in unchanged_paths if path.is_file()}
    rows=read_json(OUT/'protocol/photo_splits.json');classes=sorted({x['class_name'] for x in rows});labels=np.array([x['label'] for x in rows])
    ix={role:np.array([x['index'] for x in rows if x['role']==role]) for role in ['reference','validation','test']}
    hand=np.load(OUT/'cache/handcrafted.npz');features={key:hand[key] for key in ['rgb','gray','lbp']}
    for key in ['resnet50','dinov2','clip']:features[key]=np.load(OUT/f'cache/{key}_base.npy',mmap_mode='r')
    checks={}
    checks['reference_choice']=verify_reference_choice(rows,classes,labels,ix,features);print('PASS five-gallery sensitivity',flush=True)
    checks['real_reference_budget']=verify_budget(rows,classes,labels,ix,features);print('PASS matched-query 1/2-reference budget and predictions',flush=True)
    checks['reference_pool_stability']=verify_pool_stability(rows,classes,labels,ix,features);print('PASS pooled prototype angle and query consistency',flush=True)
    checks['open_set']=verify_open(rows,classes,labels,ix,features);print('PASS calibration thresholds, raw scores, tie-aware ROC/AP/FPR95',flush=True)
    for rel,value in before.items():assert hash_file(OUT/rel)==value,rel
    result={'status':'passed','scope':'Independent auxiliary artifact/statistic replay from saved frozen features; no image encoding/model inference or retraining',
            'protocol_sha256':done['protocol_sha256'],'checks':checks,'original_result_and_prediction_files_unchanged':True,
            'checked_output_sha256':before,'script_sha256':hash_file(__file__),'elapsed_seconds':time.time()-started,
            'limits':['Feature caches and source photographs are reused, not independently regenerated.','Source/session independence is not certified.','Thresholds generalize only to the specified fixed Fabric unknown split.']}
    destination=OUT/'audit/auxiliary_verification.json';temp=Path(str(destination)+'.tmp');temp.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(destination)
    print('COMPLETE AUXILIARY VERIFICATION',round(time.time()-started,2),'seconds',flush=True)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    main()
