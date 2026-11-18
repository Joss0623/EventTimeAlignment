import os
os.environ.update(ATTENTION_PIT_DATA_ROOT='/home/home_new/qsmx/data/AttentionPIT/expanded01',ATTENTION_PIT_RUN_ROOT='/home/home_new/qsmx/runs/AttentionPIT/expanded01',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import numpy as np,json,hashlib
from run import load,error,R,D,read_npz,snapshot_key,dump
from neural_readouts import LLMS
O=R/'neural_readouts';rows,x,y=load();saved=read_npz(O/'all_predictions.npz');text_hash=[hashlib.sha256(str(read_npz(D/'representations'/(snapshot_key(z)+'.npz'))['text']).encode()).hexdigest() for z in rows];frozen=np.array([next(j for j,w in enumerate(rows) if w['event_id']==z['event_id'] and w['kind']==z['kind'] and w['lead']==7) for z in rows]);out={}
for kind in ['event','calendar_control']:
 ii=np.array([i for i,z in enumerate(rows) if z['split']=='test' and z['kind']==kind and z['lead']==1]);changed=np.array([i for i in ii if text_hash[i]!=text_hash[frozen[i]]]);out[kind]={'n':len(ii),'word_changed':len(changed),'on_changed':{},'origin_gain':{}}
 rng=np.random.default_rng(2024);boots=[rng.choice(len(changed),len(changed),replace=True) for _ in range(2000)]
 for k in LLMS:
  delta=error(y[changed],saved[k+'Frozen7d'][changed])-error(y[changed],saved[k][changed]);out[kind]['on_changed'][k]={'pp':float(delta.mean()),'CI95':np.quantile([delta[b].mean() for b in boots],[.025,.975]).tolist()}
 for k in ['Chronos2','NumericReadout']:
  delta=error(y[frozen[ii]],saved[k][frozen[ii]])-error(y[ii],saved[k][ii]);rng=np.random.default_rng(2024);out[kind]['origin_gain'][k]={'pp':float(delta.mean()),'CI95':np.quantile([delta[rng.choice(len(ii),len(ii),replace=True)].mean() for _ in range(2000)],[.025,.975]).tolist()}
dump(O/'changed_text_diagnostic.json',out);print(json.dumps(out,indent=2))
