import os
os.environ.update(ATTENTION_PIT_DATA_ROOT='/home/home_new/qsmx/data/AttentionPIT/expanded01',ATTENTION_PIT_RUN_ROOT='/home/home_new/qsmx/runs/AttentionPIT/expanded01',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import json,numpy as np,torch,hashlib
from run import load,error,R,D,read_npz,snapshot_key,dump
from neural_readouts import net,LLMS
O=R/'semantic_diag01';O.mkdir(exist_ok=True);base=R/'neural_readouts';rows,x,y=load();features=read_npz(base/'features.npz');selected=json.load(open(base/'selected_configs.json'));frozen=np.array([next(j for j,w in enumerate(rows) if w['event_id']==z['event_id'] and w['kind']==z['kind'] and w['lead']==7) for z in rows]);torch.set_num_threads(2);out={'seed':2024,'used_splits':['train','val'],'test_not_used_in_diagnosis':True,'scope':'same frozen readout weights, controlled input replacements; hidden coordinates never compared across different trained networks','models':{}};layers=['input','linear1','gelu1','dropout_eval','linear2','gelu2','log_residual'];aux={};tr=np.array([i for i,z in enumerate(rows) if z['split']=='train']);va=np.array([i for i,z in enumerate(rows) if z['split']=='val'])
def summarize(v):return {'mean':float(np.mean(v)),'median':float(np.median(v)),'p90':float(np.quantile(v,.9))}
for key in LLMS:
 v=selected[key];f=base/f"{key}_d{v['dim']}_w{v['width']}_lr{v['lr']:g}_{v['loss']}";m=net(72,v['width'],.1);m.load_state_dict(torch.load(f/'best.pt',map_location='cpu',weights_only=True));m.eval();real=features[key+'_'+str(v['dim'])];cf={'frozen_words7':features[key+'Frozen7d_'+str(v['dim'])],'wrong_words':features[key+'FixedMismatch_'+str(v['dim'])],'zero_words':features[key+'ZeroContent_'+str(v['dim'])]};n=real.copy();n[:,:37]=real[frozen,:37];cf['frozen_numeric7']=n;out['models'][key]={'selected':v,'splits':{}}
 if key=='Qwen3':raw=np.stack([read_npz(D/'representations'/(snapshot_key(z)+'.npz'))['embedding'] for z in rows])
 else:raw=np.load(D/({'Mistral7B':'mistral','GPToss20B':'gpt-oss-20b','DeepSeekV2Lite':'deepseek-v2-lite-chat','KimiVLA3B':'kimi-vl-a3b-instruct'}[key]+'_embeddings.npz'))['embedding']
 def trace(z):
  a=[z];t=torch.tensor(z)
  with torch.inference_mode():
   for layer in m:t=layer(t);a.append(t.numpy())
  return a
 traces=trace(real);pred=np.expm1(np.clip(np.log1p(features['base'])+traces[-1],0,25));predcf={}
 for c,z in cf.items():
  zz=trace(z);b=features['base'][frozen] if c=='frozen_numeric7' else features['base'];predcf[c]=np.expm1(np.clip(np.log1p(b)+zz[-1],0,25))
  for split,idx0 in [('train',tr),('val',va)]:
   idx=np.array([i for i in idx0 if rows[i]['lead']==1 and rows[i]['kind']=='event']);out['models'][key]['splits'].setdefault(split,{})[c]={'n':len(idx),'stage_relative_displacement':{name:summarize(np.linalg.norm(aa[idx]-bb[idx],axis=1)/np.maximum(np.linalg.norm(aa[idx],axis=1),1e-8)) for name,aa,bb in zip(layers,traces,zz)},'forecast_relative_L1_percent':summarize(100*np.abs(pred[idx]-predcf[c][idx]).sum(1)/np.maximum(pred[idx].sum(1),1)),'counterfactual_minus_real_WAPE_pp':float((error(y[idx],predcf[c][idx])-error(y[idx],pred[idx])).mean())}
 for split,idx0 in [('train',tr),('val',va)]:
  idx=np.array([i for i in idx0 if rows[i]['lead']==1 and rows[i]['kind']=='event']);changed=np.array([i for i in idx if np.linalg.norm(real[i,-32:]-real[frozen[i],-32:])>1e-9]);unit=raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-8)
  out['models'][key]['splits'][split]['representation']={'n':len(idx),'changed_projected_vectors':len(changed),'raw_pooled_relative_change':summarize(np.linalg.norm(raw[idx]-raw[frozen[idx]],axis=1)/np.maximum(np.linalg.norm(raw[idx],axis=1),1e-8)),'pooled_cosine_change':summarize(1-(unit[idx]*unit[frozen[idx]]).sum(1)),'projected_text_relative_change':summarize(np.linalg.norm(real[idx,-32:]-real[frozen[idx],-32:],axis=1)/np.maximum(np.linalg.norm(real[idx,-32:],axis=1),1e-8))}
 if key=='Qwen3':
  vv=np.array([i for i in va if rows[i]['lead']==1 and rows[i]['kind']=='event']);worst=vv[np.argsort(error(y[vv],pred[vv]))[::-1][:6]];aux['validation_cases']=[{'row':int(i),'title':rows[i]['title'],'origin':rows[i]['origin'],'target_start':rows[i]['target_start'],'WAPE':float(error(y[i:i+1],pred[i:i+1])[0]),'true':y[i].tolist(),'real_prediction':pred[i].tolist(),'frozen_words_prediction':predcf['frozen_words7'][i].tolist(),'wrong_words_prediction':predcf['wrong_words'][i].tolist(),'zero_words_prediction':predcf['zero_words'][i].tolist(),'source_key':snapshot_key(rows[i])} for i in worst]
  t=torch.tensor(real[vv],requires_grad=True);z=m(t).mean();grad=torch.autograd.grad(z,t)[0].detach().numpy();aux['Qwen3_validation_gradient_RMS']={name:float(np.sqrt(np.mean(grad[:,sl]**2))) for name,sl in [('numeric',slice(0,37)),('metadata',slice(37,40)),('text',slice(40,72))]}
 dump(O/'progress.json',{'completed_model':key});print(key,'VAL',json.dumps(out['models'][key]['splits']['val']),flush=True)
out['details']=aux;dump(O/'layer_diagnosis.json',out);print('TRAIN_VALIDATION_LAYER_DIAGNOSIS_COMPLETE',flush=True)
