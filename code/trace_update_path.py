"""Train/validation-only paired traces and LoRA q/k/v/o knockout diagnostics."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import argparse,json,hashlib,traceback
import numpy as np,torch
from peft import PeftModel
from posttrain_lora import design,head,loader,O,dump,error
A=O/'mechanism_ablation01';A.mkdir(exist_ok=True)
def distance(a,b):
 return {'relative_L2':float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-8)),'cosine_change':float(1-np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-8))}
def run(config):
 d=O/config;record=json.load(open(d/'result.json'));assert record['epochs_trained']>=30;torch.set_num_threads(2);torch.manual_seed(2024);rows,f,base,y,target,texts,tr,va=design();pairs=[]
 for split in ['train','val']:
  unchanged=[]
  for i,r in enumerate(rows):
   if r['split']!=split or r['kind']!='event' or r['lead']!=1:continue
   j=next(j for j,w in enumerate(rows) if w['event_id']==r['event_id'] and w['kind']==r['kind'] and w['lead']==7)
   if texts[i]!=texts[j]:pairs.append({'current':i,'old':j,'split':split,'changed':True})
   else:unchanged.append({'current':i,'old':j,'split':split,'changed':False})
  pairs.extend(sorted(unchanged,key=lambda p:hashlib.sha256(texts[p['current']].encode()).hexdigest())[:10])
 ids=sorted({v[k] for v in pairs for k in ['current','old']});tok,m=loader(record['family'],False);m=PeftModel.from_pretrained(m,d/'adapter',is_trainable=False);m.eval();h=head(m.config.hidden_size);h.load_state_dict(torch.load(d/'head.pt',map_location='cpu',weights_only=True));h.eval();cache={};pre_norm=[None];active_mask=[None]
 def norm_hook(module,args):
  x=args[0].float();mask=active_mask[0].to(x.device);pre_norm[0]=((x*mask).sum(1)/mask.sum(1)).cpu().numpy()[0]
 hook=m.get_base_model().norm.register_forward_pre_hook(norm_hook)
 def encode(i,mode,layers=False):
  key=(mode,hashlib.sha256(texts[i].encode()).hexdigest(),layers)
  if key in cache:return cache[key]
  t=tok(texts[i],return_tensors='pt').to('cuda');active_mask[0]=t.attention_mask.unsqueeze(-1);out=m(**t,use_cache=False,output_hidden_states=layers)
  def pool(x):
   mask=active_mask[0].to(x.device);return ((x.float()*mask).sum(1)/mask.sum(1)).cpu().numpy()[0]
  v=np.stack([pool(v) for v in out.hidden_states[:-1]]+[pre_norm[0],pool(out.hidden_states[-1])]) if layers else pool(out.last_hidden_state);cache[key]=v;return v
 def headtrace(i,embedding):
  z=torch.tensor(embedding)[None];out=[]
  for part in h['project']:
   z=part(z);out.append(z.numpy()[0])
  z=torch.cat([torch.tensor(f[i:i+1]),z],-1);out.append(z.numpy()[0])
  for part in h['predict']:
   z=part(z);out.append(z.numpy()[0])
  pred=np.maximum(np.expm1(np.clip(np.log1p(base[i])+z.numpy()[0],0,25)),0);return out,pred
 traces={};numlayers=m.config.num_hidden_layers;llmlabels=['embedding']+['block_'+str(i) for i in range(1,numlayers)]+['last_block_pre_norm','final_norm'];headlabels=['project_norm','project_linear32','project_gelu','numeric_text_concat','readout_linear1','readout_gelu1','readout_dropout_eval','readout_linear2','readout_gelu2','log_residual7']
 with torch.inference_mode():
  for mode in ['adapter_on','adapter_disabled']:
   context=m.disable_adapter() if mode=='adapter_disabled' else __import__('contextlib').nullcontext()
   with context:
    for k,i in enumerate(ids):encode(i,mode,True)
    vals=[]
    for pair in pairs:
     i,j=pair['current'],pair['old'];a,b=encode(i,mode,True),encode(j,mode,True);ha,pa=headtrace(i,a[-1]);hb,pb=headtrace(i,b[-1]);vals.append({**pair,'LLM_layers':{name:distance(x,z) for name,x,z in zip(llmlabels,a,b)},'head_layers':{name:distance(x,z) for name,x,z in zip(headlabels,ha,hb)},'forecast_current':pa.tolist(),'forecast_old':pb.tolist(),'current_WAPE':float(error(y[i:i+1],pa[None])[0]),'old_WAPE':float(error(y[i:i+1],pb[None])[0])})
    traces[mode]=vals;print('LAYER TRACE COMPLETE',config,mode,len(vals),flush=True)
  # This is a fixed-weight diagnostic, not a retrained ablation or test-set selection.
  vp=[v for v in pairs if v['split']=='val' and v['changed']];mods={n:q for n,q in m.named_modules() if hasattr(q,'lora_A') and hasattr(q,'scaling')};original={n:dict(q.scaling) for n,q in mods.items()};knockouts={}
  for group in ['none','q_proj','k_proj','v_proj','o_proj']:
   changed=[]
   for n,q in mods.items():
    q.scaling.update(original[n])
    if group!='none' and n.endswith(group):
     for a in q.scaling:q.scaling[a]=0.
     changed.append(n)
   vals=[]
   for pair in vp:
    i,j=pair['current'],pair['old'];_,pa=headtrace(i,encode(i,'knockout_'+group,False));_,pb=headtrace(i,encode(j,'knockout_'+group,False));vals.append({**pair,'current_WAPE':float(error(y[i:i+1],pa[None])[0]),'old_WAPE':float(error(y[i:i+1],pb[None])[0])})
   knockouts[group]={'zeroed_modules':len(changed),'n':len(vals),'current_WAPE':float(np.mean([v['current_WAPE'] for v in vals])),'old_WAPE':float(np.mean([v['old_WAPE'] for v in vals])),'events':vals};print('KNOCKOUT',config,group,knockouts[group]['current_WAPE'],flush=True)
  for n,q in mods.items():q.scaling.update(original[n])
 hook.remove();summary={}
 for mode,vals in traces.items():
  summary[mode]={}
  for split in ['train','val']:
   vv=[v for v in vals if v['split']==split and v['changed']];summary[mode][split]={'n':len(vv),'LLM_layers':{label:{metric:float(np.mean([v['LLM_layers'][label][metric] for v in vv])) for metric in ['relative_L2','cosine_change']} for label in llmlabels},'head_layers':{label:{metric:float(np.mean([v['head_layers'][label][metric] for v in vv])) for metric in ['relative_L2','cosine_change']} for label in headlabels},'mean_update_gain_pp':float(np.mean([v['old_WAPE']-v['current_WAPE'] for v in vv]))}
 dump(A/(config+'_layer_trace.json'),{'seed':2024,'checkpoint':record,'scope':'train and validation only; all changed event lead1/7 pairs plus10 unchanged per split; no future-based case selection','adapter_off':'same adapted head, only LoRA adapters disabled; not a separately retrained frozen model','summary':summary,'knockouts':knockouts,'pair_traces':traces});print('DIAGNOSTIC COMPLETE',config,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
 try:run(a.config)
 except Exception as e:dump(A/(a.config+'_trace_failure.json'),{'error':repr(e),'traceback':traceback.format_exc()});raise
