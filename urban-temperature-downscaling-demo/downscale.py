"""Synthetic spatial temperature downscaling demo; not real-weather validation."""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def make_data(size=48, seed=7):
    rng=np.random.default_rng(seed)
    yy,xx=np.mgrid[:size,:size]
    x=xx/(size-1); y=yy/(size-1)
    elevation=160+130*np.sin(2*np.pi*x)*np.cos(np.pi*y)
    vegetation=np.clip(.55+.3*np.sin(3*np.pi*x)*np.cos(2*np.pi*y),0,1)
    buildings=np.clip(.5+.35*np.cos(2*np.pi*x)*np.cos(2*np.pi*y),0,1)
    target=31+3.8*buildings-3.1*vegetation-.005*elevation+1.8*x+rng.normal(0,.35,(size,size))
    f=6
    coarse=target.reshape(size//f,f,size//f,f).mean(axis=(1,3))
    baseline=np.repeat(np.repeat(coarse,f,axis=0),f,axis=1)
    predictors=np.column_stack([baseline.ravel(),vegetation.ravel(),buildings.ravel(),elevation.ravel(),x.ravel(),y.ravel()])
    return target,baseline,predictors,xx

def run(output='outputs',size=48,seed=7):
    target,baseline,X,xx=make_data(size,seed)
    test=xx.ravel()>=int(size*.75)
    model=HistGradientBoostingRegressor(max_iter=120,learning_rate=.07,max_leaf_nodes=15,random_state=seed)
    model.fit(X[~test],target.ravel()[~test])
    pred=model.predict(X)
    truth=target.ravel()[test]
    metrics={'dataset':'synthetic_only','test_design':'held-out eastern spatial strip',
             'training_pixels':int((~test).sum()),'test_pixels':int(test.sum()),
             'baseline_mae_c':float(mean_absolute_error(truth,baseline.ravel()[test])),
             'model_mae_c':float(mean_absolute_error(truth,pred[test])),
             'baseline_rmse_c':float(np.sqrt(mean_squared_error(truth,baseline.ravel()[test]))),
             'model_rmse_c':float(np.sqrt(mean_squared_error(truth,pred[test])))}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    np.savez_compressed(out/'example_surfaces.npz',target=target,coarse_baseline=baseline,model_prediction=pred.reshape(target.shape))
    return metrics
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs');a=p.parse_args()
    print(json.dumps(run(a.output),indent=2))
