"""AI4I predictive maintenance benchmark. UCI dataset is synthetic, NOT factory telemetry."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier,IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,confusion_matrix,precision_score,recall_score,fbeta_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

NUM=['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']
CAT=['Type']
TARGET='Machine failure'
# Exclude UID, Product ID and five label-derived failure mode columns to avoid leakage.
def load(path):
 d=pd.read_csv(path)
 missing=set(NUM+CAT+[TARGET])-set(d.columns)
 if missing:raise ValueError('Missing columns: '+repr(sorted(missing)))
 if d[TARGET].isna().any() or not set(d[TARGET].unique()).issubset({0,1}):raise ValueError('Invalid failure target')
 if len(d)<70 or min(d[TARGET].value_counts())<8:raise ValueError('Insufficient samples per class')
 return d
def prep():
 return ColumnTransformer([
 ('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),
 ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CAT)],sparse_threshold=0)
def measure(y,prob,threshold):
 pred=(prob>=threshold).astype(int)
 tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return {'precision':float(precision_score(y,pred,zero_division=0)),
 'recall':float(recall_score(y,pred,zero_division=0)),
 'F2':float(fbeta_score(y,pred,beta=2,zero_division=0)),
 'average_precision':float(average_precision_score(y,prob)),
 'roc_auc':float(roc_auc_score(y,prob)),
 'TP':int(tp),'FP':int(fp),'FN':int(fn),'TN':int(tn)}
def threshold_search(y,prob,fn_cost=8.,fp_cost=1.):
 choices=np.unique(np.r_[np.linspace(.01,.99,99),prob])
 return min(((fn_cost*confusion_matrix(y,prob>=t,labels=[0,1])[1,0]+fp_cost*confusion_matrix(y,prob>=t,labels=[0,1])[0,1],float(t)) for t in choices))[1]
def run(csv,output='reports',seed=42):
 d=load(csv);X=d[NUM+CAT];y=d[TARGET].astype(int).to_numpy()
 Xwork,Xtest,ywork,ytest=train_test_split(X,y,test_size=.2,stratify=y,random_state=seed)
 Xtrain,Xval,ytrain,yval=train_test_split(Xwork,ywork,test_size=.25,stratify=ywork,random_state=seed)
 trials=[]
 for name,model in [
 ('balanced_logistic',LogisticRegression(max_iter=600,class_weight='balanced',random_state=seed)),
 ('gradient_boosting',HistGradientBoostingClassifier(max_iter=180,learning_rate=.06,max_leaf_nodes=13,random_state=seed))]:
  pipe=Pipeline([('preprocess',prep()),('model',model)])
  pipe.fit(Xtrain,ytrain);p=pipe.predict_proba(Xval)[:,1]
  t=threshold_search(yval,p);m=measure(yval,p,t)
  trials.append((8*m['FN']+m['FP'],name,pipe,t,m))
 trials.sort(key=lambda x:x[0])
 _,selected,model,threshold,validation=trials[0]
 test=measure(ytest,model.predict_proba(Xtest)[:,1],threshold)
 transformer=prep();healthy=transformer.fit_transform(Xtrain.loc[ytrain==0])
 anomaly=IsolationForest(n_estimators=150,contamination=.03,random_state=seed).fit(healthy)
 score=-anomaly.decision_function(transformer.transform(Xtest))
 report={'data':'UCI AI4I 2020 synthetic industrial benchmark','samples':len(d),'positive_rate':float(y.mean()),
 'split':'stratified train=60%, validation=20%, test=20%; not temporal or asset-level',
 'leakage':'IDs and failure-mode outcome labels excluded',
 'selected_model':selected,'validation_selected_threshold':threshold,
 'validation':validation,'held_out_test':test,
 'anomaly_average_precision':float(average_precision_score(ytest,score)),
 'anomaly_roc_auc':float(roc_auc_score(ytest,score)),
 'illustrative_costs':{'false_negative':8,'false_positive':1},
 'limitations':['Simulation data, not production machine telemetry','No timestamps; random split cannot validate prospective performance','Costs are hypothetical'] }
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
 (out/'SUMMARY.md').write_text(f"# Held-out results\n\nModel: {selected}; recall: {test['recall']:.3f}; precision: {test['precision']:.3f}; PR-AUC: {test['average_precision']:.3f}.\n\nThese results use simulated AI4I data, not live factory operations.\n")
 return report
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--csv',required=True);ap.add_argument('--output',default='reports');args=ap.parse_args()
 print(json.dumps(run(args.csv,args.output),indent=2))
