"""Validation-selected ablation report plus signed correction diagnostic."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import json,time,csv,traceback
import numpy as np,torch
from posttrain_lora import design,head,O,R,D,dump,error,read_npz,snapshot_key
from mechanism_ablation import head_for,delta_controls
A=O/'mechanism_ablation01';conditions=['NumericOnly','MetadataOnly','CurrentText','OldText7d','WrongText','ZeroInitCurrent','CurrentText128','MetadataOnly128','DeltaText','WrongDelta'];alphas=[-1.,-.5,0.,.25,.5,1.,1.5]
def run():
 while not all((A/(f+suffix)).exists() for f in ['Qwen3','Mistral7B'] for suffix in ['_complete.json','_extension_complete.json']):
  failed=[A/(family+suffix) for family in ['Qwen3','Mistral7B'] for suffix in ['_failure.json','_extension_failure.json'] if (A/(family+suffix)).exists()]
  if failed:raise RuntimeError('Ablation failure '+str(failed))
  time.sleep(20)
 torch.set_num_threads(2);rows,f,base,y,target,texts,tr,va=design();oldid=np.array([next(j for j,w in enumerate(rows) if w['event_id']==z['event_id'] and w['kind']==z['kind'] and w['lead']==7) for z in rows]);snaps=[json.load(open(D/'snapshots'/(snapshot_key(z)+'.json'))) for z in rows];donors=[]
 for i,z in enumerate(rows):
  pool=[j for j in tr if rows[j]['pageid']!=z['pageid'] and snaps[j]['available'] and snaps[j]['revision_time']<z['origin'] and rows[j]['origin']<=z['origin']];donors.append(min(pool,key=lambda j:(rows[j]['lead']!=z['lead'],abs(len(texts[j])-len(texts[i])),rows[j]['event_id'])) if pool else -1)
 test=np.array([i for i,z in enumerate(rows) if z['split']=='test' and z['kind']=='event' and z['lead']==1]);changed=test[[texts[i]!=texts[oldid[i]] for i in test]];rng=np.random.default_rng(2024);groups=np.array([rows[i]['pageid'] for i in test]);ug=np.unique(groups);boots=[np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(ug,len(ug),replace=True)]) for _ in range(2000)];pred={};selection={};table=[];scales=[]
 def scale_diagnostic(label,p):
  delta=np.log1p(p)-np.log1p(base);trials=[];ps={}
  for alpha in alphas:
   pp=np.maximum(np.expm1(np.clip(np.log1p(base)+alpha*delta,0,25)),0);score=float(error(y[va],pp[va]).mean());trials.append({'alpha':alpha,'validation_WAPE':score});ps[alpha]=pp
  choice=min(trials,key=lambda v:v['validation_WAPE']);pp=ps[choice['alpha']];scales.append({'model':label,'selected_alpha':choice['alpha'],'validation_WAPE':choice['validation_WAPE'],'test_event1d_WAPE':float(error(y[test],pp[test]).mean()),'trials':trials});pred[label+'_ValScaled']=pp
 for family in ['Qwen3','Mistral7B']:
  raw=read_npz(O/(family+'_nf4_frozen_embeddings.npz'))['embedding'];vectors={'CurrentText':raw,'OldText7d':raw[oldid],'WrongText':np.array([raw[j] if j>=0 else np.zeros(raw.shape[1],np.float32) for j in donors])};vectors['ZeroInitCurrent']=raw;vectors['NumericOnly']=np.zeros_like(raw);vectors['MetadataOnly']=vectors['NumericOnly'];dd,wd,fb=delta_controls(raw,oldid,rows,tr,snaps,texts);vectors.update({'CurrentText128':raw,'MetadataOnly128':vectors['MetadataOnly'],'DeltaText':dd,'WrongDelta':wd})
  for condition in conditions:
   files=list(A.glob(f'{family}_{condition}_lr*/result.json'));assert len(files)==3;v=min((json.load(open(p)) for p in files),key=lambda z:z['validation_WAPE']);selection[family+'_'+condition]=v;h=head_for(raw.shape[1],v.get('projection_dim',32));h.load_state_dict(torch.load(v['checkpoint_source'],map_location='cpu',weights_only=True));h.eval();ff=f.copy()
   if condition=='NumericOnly':ff[:,37:]=0
   with torch.inference_mode():delta=h['predict'](torch.cat([torch.tensor(ff),h['project'](torch.tensor(vectors[condition]))],-1)).numpy()
   p=np.maximum(np.expm1(np.clip(np.log1p(base)+delta,0,25)),0);score=float(error(y[va],p[va]).mean());assert abs(score-v['validation_WAPE'])<.005,(family,condition,score,v['validation_WAPE']);label=family+'_'+condition;pred[label]=p;gain=error(y[test],base[test])-error(y[test],p[test]);entry={'model':label,'selected_lr':v['lr'],'epochs_trained':v['epochs_trained'],'validation_WAPE':score,'test_event1d_WAPE':float(error(y[test],p[test]).mean()),'gain_vs_numeric_pp':float(gain.mean()),'gain_CI95':np.quantile([gain[b].mean() for b in boots],[.025,.975]).tolist()}
   if condition in ['CurrentText','ZeroInitCurrent','CurrentText128','DeltaText']:
    with torch.inference_mode():deltaold=h['predict'](torch.cat([torch.tensor(ff),h['project'](torch.tensor(np.zeros_like(raw) if condition=='DeltaText' else raw[oldid]))],-1)).numpy()
    po=np.maximum(np.expm1(np.clip(np.log1p(base)+deltaold,0,25)),0);pred[label+'_FixedOld']=po;diff=error(y[test],po[test])-error(y[test],p[test]);entry.update(update_gain_pp=float(diff.mean()),update_CI95=np.quantile([diff[b].mean() for b in boots],[.025,.975]).tolist(),changed_n=len(changed),changed_update_gain_pp=float((error(y[changed],po[changed])-error(y[changed],p[changed])).mean()))
   table.append(entry);scale_diagnostic(label,p)
 # Apply the same predeclared signed-coefficient test to completed LoRA/frozen and numerical controls.
 for path in sorted((O/'completed_evaluation').glob('*.npz')):
  if not path.with_suffix('.json').exists():continue
  scale_diagnostic(path.stem,read_npz(path)['CurrentText'])
 old=read_npz(R/'neural_readouts/all_predictions.npz')
 for label in ['Chronos2','PatchTST','iTransformer','TimeMixer']:scale_diagnostic(label,old[label])
 np.savez_compressed(A/'predictions.npz',**pred);dump(A/'results.json',{'seed':2024,'selection':selection,'comparisons':table,'signed_correction':scales,'scope':'all60full-budget frozen-encoder module controls; two existing CurrentText configurations reused; alpha selected only validation; test previously inspected exploratory'})
 with open(A/'comparison.csv','w',newline='') as file:
  w=csv.writer(file);w.writerow(['model','selected_lr','epochs','validation_WAPE','test_event1d_WAPE','gain_vs_numeric_pp','update_gain_pp'])
  for v in table:w.writerow([v['model'],v['selected_lr'],v['epochs_trained'],v['validation_WAPE'],v['test_event1d_WAPE'],v['gain_vs_numeric_pp'],v.get('update_gain_pp')])
 with open(A/'signed_correction.csv','w',newline='') as file:
  w=csv.writer(file);w.writerow(['model','selected_alpha','validation_WAPE','test_event1d_WAPE'])
  for v in scales:w.writerow([v['model'],v['selected_alpha'],v['validation_WAPE'],v['test_event1d_WAPE']])
 print('MECHANISM ABLATION REPORT COMPLETE',json.dumps(table),flush=True)
if __name__=='__main__':
 try:run()
 except Exception as e:dump(A/'report_failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
