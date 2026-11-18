"""Evaluate only completed configurations; fixed current/old/mismatched text controls."""
import os,sys,json,argparse,time,subprocess,traceback,csv,hashlib
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import numpy as np,torch
from posttrain_lora import design,head,loader,O,R,D,error,dump,read_npz,snapshot_key
E=O/'completed_evaluation';E.mkdir(exist_ok=True)
def worker(config):
 d=O/config;v=json.load(open(d/'result.json'));assert v['epochs_trained']>=30 and v['seed']==2024
 if (E/(config+'.json')).exists():return
 torch.set_num_threads(2);torch.manual_seed(2024);np.random.seed(2024);rows,f,base,y,target,texts,tr,va=design();family=v['family'];mode=v['mode'];frozen=[next(j for j,w in enumerate(rows) if w['event_id']==z['event_id'] and w['kind']==z['kind'] and w['lead']==7) for z in rows];snaps=[json.load(open(D/'snapshots'/(snapshot_key(z)+'.json'))) for z in rows];donors=[]
 for i,z in enumerate(rows):
  pool=[j for j,w in enumerate(rows) if w['pageid']!=z['pageid'] and snaps[j]['available'] and snaps[j]['revision_time']<z['origin'] and w['origin']<=z['origin']]
  donors.append(min(pool,key=lambda j:(rows[j]['lead']!=z['lead'],abs(len(texts[j])-len(texts[i])),rows[j]['event_id'])) if pool else i)
 if mode=='frozen':
  emb=read_npz(O/(family+'_nf4_frozen_embeddings.npz'))['embedding'];device='cpu'
 else:
  from peft import PeftModel
  tok,m=loader(family,False);m=PeftModel.from_pretrained(m,d/'adapter',is_trainable=False);m.eval();cache={};values=[];device='cuda'
  with torch.inference_mode():
   for i,text in enumerate(texts):
    key=hashlib.sha256(text.encode()).hexdigest()
    if key not in cache:
     t=tok(text,return_tensors='pt').to(device);out=m(**t,use_cache=False).last_hidden_state;mask=t.attention_mask.unsqueeze(-1).to(out.device);cache[key]=((out.float()*mask).sum(1)/mask.sum(1)).cpu().numpy()[0]
    values.append(cache[key])
    if i%50==0:print('EVALUATE',config,i,len(rows),'unique',len(cache),flush=True)
  emb=np.stack(values);del m,cache;torch.cuda.empty_cache()
 h=head(emb.shape[1]).to(device);h.load_state_dict(torch.load(d/'head.pt',map_location=device,weights_only=True));h.eval();fx=torch.tensor(f,device=device);pred={}
 with torch.inference_mode():
  for name,ii in [('CurrentText',list(range(len(rows)))),('OldText7d',frozen),('WrongText',donors)]:
   ee=torch.tensor(emb[ii],device=device);delta=h['predict'](torch.cat([fx,h['project'](ee)],-1)).cpu().numpy();pred[name]=np.maximum(np.expm1(np.clip(np.log1p(base)+delta,0,25)),0)
 val=float(error(y[va],pred['CurrentText'][va]).mean());assert abs(val-v['validation_WAPE'])<.005,(config,val,v['validation_WAPE'])
 np.savez_compressed(E/(config+'.npz'),**pred)
 out={'config':config,'training':v,'checkpoint_validation_WAPE':val,'validation_reproduction_absolute_difference':abs(val-v['validation_WAPE']),'curves':{},'gains':{},'scope':'completed-run evaluation; other configurations continue, no test-based selection','seed':2024}
 fresh=set(json.load(open(R/'protocol.json'))['fresh_test_pageids']);old=read_npz(R/'neural_readouts/all_predictions.npz')
 for split in ['val','test']:
  out['curves'][split]={};out['gains'][split]={}
  for kind in ['event','calendar_control','fresh_event','changed_text_event','pooled']:
   for lead in [7,3,1]:
    ii=np.array([i for i,z in enumerate(rows) if z['split']==split and z['lead']==lead and (kind=='pooled' or z['kind']==kind or kind=='fresh_event' and z['kind']=='event' and z['pageid'] in fresh or kind=='changed_text_event' and z['kind']=='event' and texts[i]!=texts[frozen[i]])]);
    if not len(ii):continue
    key=kind+'_'+str(lead)+'d';groups=np.array([rows[i]['pageid'] for i in ii]);unique=np.unique(groups);rng=np.random.default_rng(2024);boots=[np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)]) for _ in range(2000)];es={k:error(y[ii],p[ii]) for k,p in pred.items()};out['curves'][split][key]={k:{'WAPE':float(e.mean()),'n':len(ii)} for k,e in es.items()};gains={}
    for first in ['OldText7d','WrongText','NumericReadout','Chronos2','PatchTST','iTransformer','TimeMixer']:
     e=es[first] if first in es else error(y[ii],old[first][ii]);delta=e-es['CurrentText'];gains[first+' -> CurrentText']={'pp':float(delta.mean()),'CI95':np.quantile([delta[b].mean() for b in boots],[.025,.975]).tolist()}
    out['gains'][split][key]=gains
 dump(E/(config+'.json'),out);print('EVALUATION COMPLETE',config,val,out['curves']['test']['event_1d'],out['gains']['test']['event_1d']['OldText7d -> CurrentText'],flush=True)
def summarize():
 a=json.load(open(O/'completed_frozen_analysis.json'));results={p.stem:json.load(open(p)) for p in E.glob('*.json') if not p.stem.endswith('_failure') and (E/(p.stem+'.npz')).exists()};comparison=[];updates=[]
 for k in ['Chronos2','PatchTST','iTransformer','TimeMixer','NumericReadout']:
  comparison.append({'model':k,'status':'complete','epochs':None,'validation_WAPE':a['validation_comparison'][k],'test_event_lead1_WAPE':a['test_event_lead1_comparison'][k]})
 for config in sorted(results):
  v=results[config];comparison.append({'model':config,'status':'complete','epochs':v['training']['epochs_trained'],'validation_WAPE':v['checkpoint_validation_WAPE'],'test_event_lead1_WAPE':v['curves']['test']['event_1d']['CurrentText']['WAPE']})
  for subset in ['event_1d','changed_text_event_1d']:
   g=v['gains']['test'][subset]['OldText7d -> CurrentText'];c=v['curves']['test'][subset];updates.append({'model':config,'subset':subset,'n':c['CurrentText']['n'],'current_WAPE':c['CurrentText']['WAPE'],'old_WAPE':c['OldText7d']['WAPE'],'gain_pp':g['pp'],'CI95_low':g['CI95'][0],'CI95_high':g['CI95'][1]})
 dump(O/'completed_results_partial.json',{'seed':2024,'comparison':comparison,'updates':updates,'completed_evaluations':len(results),'protocol':'same fixed targets/splits/inputs; checkpoint selection only validation; final family selection waits all configs'})
 for name,data in [('completed_comparison_partial.csv',comparison),('completed_semantic_gains_partial.csv',updates)]:
  if data:
   with open(O/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 print(json.dumps({'comparison':comparison,'updates':updates},indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['worker','summarize']);p.add_argument('--config');a=p.parse_args()
 try:worker(a.config) if a.stage=='worker' else summarize()
 except Exception as e:dump(E/(str(a.config)+'_failure.json'),{'error':repr(e),'traceback':traceback.format_exc()});raise
