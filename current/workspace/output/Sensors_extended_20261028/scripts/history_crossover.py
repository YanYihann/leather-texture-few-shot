from pathlib import Path
import torch,numpy as np,csv,json
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1];DATA=ROOT/'IC_project/01_核心代码与数据集';RES=ROOT/'IC_project/02_实验结果与生成数据'
OLD=ROOT/'output/Sensors_revision_20261010/new_experiments'
classes=sorted(p.name for p in (DATA/'dataset_train').iterdir() if p.is_dir());rows=[];cases=[]
for model,split,hist in [('resnet50','dataset_val',RES/'results/stage2_frozen_resnet50_similarity_predictions.csv'),('dinov2_vits14','my_dataset_test_filtered',RES/'results_independent_filtered/filtered_dinov2_vits14_similarity_predictions.csv')]:
    fs=np.load(OLD/('resnet50_rgb_features.npz' if model=='resnet50' else 'dinov2_s14_rgb_features.npz'))
    p_old=torch.load(DATA/'features'/f'{model}_prototypes.pt',map_location='cpu')['prototypes'].numpy();p_new=fs['dataset_train'];q=fs[split]
    historical=list(csv.DictReader(hist.open(encoding='utf-8-sig')));labels=[classes.index(r['true_class']) for r in historical]
    for variant,p in [('current',p_new),('archived_prototype',p_old)]:
        score=q@p.T;order=np.argsort(-score,axis=1,kind='stable');top3=(order[:,:3]==np.array(labels)[:,None]).any(1)
        rows.append(dict(model=model,split=split,variant=variant,top1=int((order[:,0]==labels).sum()),top3=int(top3.sum())))
        for i,r in enumerate(historical):
            sfx='similarity' if model=='resnet50' else 'score'
            err=max(abs(float(r[f'top{j}_{sfx}'])-float(score[i,classes.index(r[f'top{j}_class'])])) for j in [1,2,3])
            if (r['top3_correct'].lower()=='true')!=bool(top3[i]) or err>1e-4:
                cases.append(dict(model=model,variant=variant,class_id=r['true_class'],historical_top3=r['top3_correct'],current_top3=bool(top3[i]),historical_classes=[r[f'top{j}_class'] for j in [1,2,3]],current_classes=[classes[x] for x in order[i,:5]],true_rank=int(np.where(order[i]==labels[i])[0][0])+1,historical_score_error=err,true_score=float(score[i,labels[i]]),third_score=float(score[i,order[i,2]])))
(OUT/'audit/history_crossover.json').write_text(json.dumps(dict(summary=rows,cases=cases),indent=2),encoding='utf8');print(json.dumps(dict(summary=rows,cases=cases[:15]),indent=2))
