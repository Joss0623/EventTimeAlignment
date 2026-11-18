import os
os.environ.update(ATTENTION_PIT_DATA_ROOT='/home/home_new/qsmx/data/AttentionPIT/expanded01',ATTENTION_PIT_RUN_ROOT='/home/home_new/qsmx/runs/AttentionPIT/expanded01',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import json,numpy as np,torch,csv
from run import load,error,R,read_npz,dump
from neural_readouts import net,LLMS
O=R/'neural_readouts';rows,x,y=load();d=read_npz(O/'features.npz');saved=read_npz(O/'all_predictions.npz');sel=json.load(open(O/'selected_configs.json'));fresh=set(json.load(open(R/'protocol.json'))['fresh_test_pageids']);torch.set_num_threads(2)
frozen=np.array([next(j for j,w in enumerate(rows) if w['event_id']==z['event_id'] and w['kind']==z['kind'] and w['lead']==7) for z in rows]);new={};res={'seed':2024,'scope':'post-training fixed-weight diagnostic, not retraining, not checkpoint selection','order':['numeric/origin update','metadata update','retained word update'],'interpretation':'ordered readout attribution, path-dependent, not causal effects or information-theoretic bounds','groups':{}}
for k in LLMS:
 v=sel[k];f=O/f"{k}_d{v['dim']}_w{v['width']}_lr{v['lr']:g}_{v['loss']}";xx=d[k+'_'+str(v['dim'])];m=net(xx.shape[1],v['width'],.1);m.load_state_dict(torch.load(f/'best.pt',map_location='cpu',weights_only=True));m.eval();cf=xx.copy();cf[:,-35:]=xx[frozen,-35:]
 with torch.inference_mode():z=m(torch.tensor(xx)).numpy();zz=m(torch.tensor(cf)).numpy()
 reproduced=np.maximum(np.expm1(np.clip(np.log1p(d['base'])+z,0,25)),0);assert np.allclose(saved[k],reproduced,rtol=2e-6,atol=.02),k
 new[k+'AllTextFrozen7d']=np.maximum(np.expm1(np.clip(np.log1p(d['base'])+zz,0,25)),0)
for kind in ['event','fresh_event','calendar_control']:
 res['groups'][kind]={}
 for lead in [3,1]:
  ii=np.array([i for i,z in enumerate(rows) if z['split']=='test' and z['lead']==lead and z['kind']==('event' if kind=='fresh_event' else kind) and (kind!='fresh_event' or z['pageid'] in fresh)]);gg=np.array([rows[i]['pageid'] for i in ii]);unique=np.unique(gg);rng=np.random.default_rng(2024);boots=[np.concatenate([np.flatnonzero(gg==g) for g in rng.choice(unique,len(unique),replace=True)]) for _ in range(2000)];res['groups'][kind][str(lead)]={}
  for k in LLMS:
   ea=error(y[frozen[ii]],saved[k][frozen[ii]]);eb=error(y[ii],new[k+'AllTextFrozen7d'][ii]);ec=error(y[ii],saved[k+'Frozen7d'][ii]);ed=error(y[ii],saved[k][ii]);components={'numeric_origin':ea-eb,'metadata':eb-ec,'words':ec-ed,'total':ea-ed};assert np.allclose(components['numeric_origin']+components['metadata']+components['words'],components['total'])
   res['groups'][kind][str(lead)][k]={'n':len(ii),'WAPE_A7':float(ea.mean()),'WAPE_B_numeric_only_update':float(eb.mean()),'WAPE_C_metadata_update':float(ec.mean()),'WAPE_D_current':float(ed.mean()),**{term:{'pp':float(delta.mean()),'CI95':np.quantile([delta[b].mean() for b in boots],[.025,.975]).tolist()} for term,delta in components.items()}}
np.savez_compressed(O/'refresh_attribution_predictions.npz',**new);dump(O/'refresh_attribution.json',res)
with (O/'refresh_attribution.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['subset','lead','model','n','numeric_origin_gain_pp','metadata_gain_pp','word_refresh_gain_pp','total_gain_pp'])
 for kind,leads in res['groups'].items():
  for lead,models in leads.items():
   for k,z in models.items():w.writerow([kind,lead,k,z['n']]+[z[t]['pp'] for t in ['numeric_origin','metadata','words','total']])
print(json.dumps(res['groups']['event']['1'],indent=2))
