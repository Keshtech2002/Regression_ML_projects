"""Offline, rule-based ML workflow automation; NOT an LLM agent."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

def validate_data(X,y):
    if len(X)!=len(y) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Dataset integrity check failed')

def experiment(output='outputs',seed=42):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    X,y=make_regression(n_samples=650,n_features=8,noise=13,random_state=seed)
    validate_data(X,y)
    split=int(len(X)*.6);val_end=int(len(X)*.8)
    runs=[]
    for depth in (3,7,15):
        for rate in (.03,.10):
            model=HistGradientBoostingRegressor(max_iter=90,max_leaf_nodes=depth,learning_rate=rate,random_state=seed)
            model.fit(X[:split],y[:split])
            score=float(mean_absolute_error(y[split:val_end],model.predict(X[split:val_end])))
            runs.append({'max_leaf_nodes':depth,'learning_rate':rate,'validation_mae':score})
    runs.sort(key=lambda r:r['validation_mae'])
    with (out/'experiment_log.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['max_leaf_nodes','learning_rate','validation_mae'])
        writer.writeheader();writer.writerows(runs)
    best=runs[0]
    final=HistGradientBoostingRegressor(max_iter=90,max_leaf_nodes=best['max_leaf_nodes'],learning_rate=best['learning_rate'],random_state=seed)
    final.fit(X[:val_end],y[:val_end])
    test_mae=float(mean_absolute_error(y[val_end:],final.predict(X[val_end:])))
    report={'type':'offline_rule_based_automation','dataset':'synthetic_sklearn_regression','seed':seed,
            'data_checks':'passed','experiments':len(runs),
            'selection_note':'Parameters tuned on validation only; test evaluated once after refit.',
            'best_validation_run':best,'final_test_mae':test_mae}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs');a=p.parse_args()
    print(json.dumps(experiment(a.output),indent=2))
