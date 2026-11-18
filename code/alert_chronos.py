import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import json,numpy as np,torch,time
B=Path('/home/home_new/qsmx');D=B/'data/AttentionPIT/alert01';R=B/'runs/AttentionPIT/alert01'
from chronos import BaseChronosPipeline
from sklearn.preprocessing import StandardScaler
x=np.load(D/'panel.npz')['x'];rows=json.load(open(D/'rows.json'));tr=[i for i,r in enumerate(rows) if r['split']=='train'];torch.set_num_threads(2);torch.manual_seed(2024)
pipe=BaseChronosPipeline.from_pretrained(str(B/'pycodes/WebBurst/weights/amazon--chronos-2'),device_map='cuda',torch_dtype=torch.bfloat16,local_files_only=True);qi=pipe.quantiles.index(.5);ui=min(range(len(pipe.quantiles)),key=lambda j:abs(pipe.quantiles[j]-.9));parts=[];cache=D/'chronos_chunks';cache.mkdir(exist_ok=True)
for mode in ['raw','log']:
 chunks=[]
 for start in range(0,len(x),128):
  p=cache/(mode+'_'+str(start)+'.npy')
  if p.exists():ar=np.load(p)
  else:
   v=x[start:start+128];v=np.log1p(v) if mode=='log' else v;z=pipe.predict(torch.tensor(v).unsqueeze(1),prediction_length=7,batch_size=128,context_length=28,cross_learning=False);ar=np.stack([np.stack([s[0,qi,:].cpu().float().numpy(),s[0,ui,:].cpu().float().numpy()]) for s in z]);ar=np.log1p(np.maximum(ar,0)) if mode=='raw' else ar;assert np.isfinite(ar).all();np.save(p,ar)
  chunks.append(ar.reshape(len(ar),-1))
  if start%1280==0:print(mode,start,len(x),flush=True)
 parts.append(np.concatenate(chunks))
f=np.concatenate(parts+[np.zeros((len(x),4))],axis=1);sc=StandardScaler().fit(f[tr]);np.save(D/'Chronos2_features.npy',sc.transform(f).astype(np.float32));(R/'Chronos2_encoding.json').write_text(json.dumps({'seed':2024,'frozen':True,'inputs':'same28 history only','outputs':'raw/log median and90percentile next7day predictions; training-only scaling','adaptation':'same classification MLP trained with full grid and budget'},indent=2));print('CHRONOS COMPLETE',flush=True)
