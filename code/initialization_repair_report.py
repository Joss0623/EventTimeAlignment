import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import json,time,csv,traceback,numpy as np,torch
from posttrain_lora import design,O,read_npz,dump,error
from mechanism_ablation import head_for
A=O/'mechanism_ablation01';P=A/'initialization_repair'
def main():
 while not all((P/(f+'_init128_complete.json')).exists() for f in ['Qwen3','Mistral7B']):
  failures=list(P.glob('*failure.json'))
  if failures:raise RuntimeError(str(failures))
  time.sleep(15)
 torch.set_num_threads(2);rows,f,base,y,target,texts,tr,va=design();old=np.array([next(j for j,w in enumerate(rows) if w['event_id']==z['event_id'] and w['kind']==z['kind'] and w['lead']==7) for z in rows]);te=np.array([i for i,z in enumerate(rows) if z['split']=='test' and z['kind']=='event' and z['lead']==1]);prior=json.load(open(A/'results.json'));table=[{'model':'NumericReadout','validation_WAPE':float(error(y[va],base[va]).mean()),'test_event1d_WAPE':float(error(y[te],base[te]).mean())}];pred={}
 for v in prior['comparisons']:
  if v['model'].endswith(('CurrentText128','MetadataOnly128')):table.append({'model':v['model'],'validation_WAPE':v['validation_WAPE'],'test_event1d_WAPE':v['test_event1d_WAPE']})
 selected={}
 for family in ['Qwen3','Mistral7B']:
  raw=read_npz(O/(family+'_nf4_frozen_embeddings.npz'))['embedding']
  for condition in ['ZeroInitCurrent128','ZeroInitMetadata128']:
   results=[json.load(open(p)) for p in P.glob(f'{family}_{condition}_lr*/result.json')];assert len(results)==3;v=min(results,key=lambda x:x['validation_WAPE']);label=family+'_'+condition;selected[label]=v;h=head_for(raw.shape[1],128);h.load_state_dict(torch.load(v['checkpoint_source'],map_location='cpu',weights_only=True));h.eval();pp={}
   with torch.inference_mode():
    for name,vec in [('Current',raw),('Old',raw[old])]:
     if condition=='ZeroInitMetadata128':vec=np.zeros_like(vec)
     d=h['predict'](torch.cat([torch.tensor(f),h['project'](torch.tensor(vec))],-1)).numpy();pp[name]=np.maximum(np.expm1(np.clip(np.log1p(base)+d,0,25)),0);pred[label+'_'+name]=pp[name]
   actual=float(error(y[va],pp['Current'][va]).mean());assert abs(actual-v['validation_WAPE'])<.005;delta=error(y[te],pp['Old'][te])-error(y[te],pp['Current'][te]);rng=np.random.default_rng(2024);boots=[rng.integers(0,len(te),len(te)) for _ in range(2000)];table.append({'model':label,'selected_lr':v['lr'],'validation_WAPE':actual,'test_event1d_WAPE':float(error(y[te],pp['Current'][te]).mean()),'text_update_gain_pp':float(delta.mean()),'text_update_CI95':np.quantile([delta[b].mean() for b in boots],[.025,.975]).tolist(),'validation_signed_total_correction_percent':float(np.mean(100*(pp['Current'][va]-base[va]).sum(1)/np.maximum(base[va].sum(1),1)))})
 dump(P/'results.json',{'seed':2024,'selected':selected,'comparison':table,'scope':'same information/seed/full30minimum; last-layer initialization only; no text utility inferred solely from calibration gains'});np.savez_compressed(P/'predictions.npz',**pred)
 with open(P/'comparison.csv','w',newline='') as file:
  w=csv.writer(file);w.writerow(['model','validation_WAPE','test_event1d_WAPE','text_update_gain_pp','validation_signed_correction_percent'])
  for v in table:w.writerow([v['model'],v['validation_WAPE'],v['test_event1d_WAPE'],v.get('text_update_gain_pp'),v.get('validation_signed_total_correction_percent')])
 print('INITIALIZATION REPAIR COMPLETE',json.dumps(table),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:dump(P/'report_failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
