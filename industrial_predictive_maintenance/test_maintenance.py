import numpy as np,pandas as pd,pytest
from maintenance import load,run
def sample(n=400):
 rng=np.random.default_rng(11)
 d=pd.DataFrame({'Air temperature [K]':rng.normal(300,2,n),'Process temperature [K]':rng.normal(310,2,n),'Rotational speed [rpm]':rng.normal(1500,100,n),'Torque [Nm]':rng.normal(40,8,n),'Tool wear [min]':rng.integers(0,250,n),'Type':rng.choice(['L','M','H'],n)})
 d['Machine failure']=((d['Torque [Nm]']>39)&(d['Tool wear [min]']>170)).astype(int)
 d['TWF']=d['Machine failure'] # leaked outcome should not enter modeling features
 return d
def test_end_to_end(tmp_path):
 p=tmp_path/'sample.csv';sample().to_csv(p,index=False)
 a=run(p,tmp_path/'out')
 assert a['selected_model'] in ['balanced_logistic','gradient_boosting']
 assert 0<=a['held_out_test']['recall']<=1
 assert (tmp_path/'out/results.json').exists()
def test_schema_check(tmp_path):
 p=tmp_path/'bad.csv';sample().drop(columns=['Torque [Nm]']).to_csv(p,index=False)
 with pytest.raises(ValueError,match='Missing'):load(p)
