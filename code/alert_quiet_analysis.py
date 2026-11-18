import json,numpy as np,csv
from alert_train import data,metric,R
rows,a=data();out=json.load(open(R/'quiet_policy_secondary.json'));pred=dict(np.load(R/'all_predictions.npz'));te=np.flatnonzero(a['split']=='test');groups={}
for i in te:
 if a['first'][i]>=0:groups.setdefault((rows[i]['page'],int(a['first'][i])),[]).append(i)
ref='TimeMixer';out['reference']=ref;out['paired_gains']={};out['mismatch']={};page=sorted({k[0] for k in groups})
for key in ['Qwen3','Mistral','gpt-oss-20b','deepseek-v2-lite-chat','kimi-vl-a3b-instruct']:
 parts={p:[] for p in page};thr=out['models'][key]['threshold'];rt=out['models'][ref]['threshold']
 for (p,s),ids in groups.items():
  ll=[s-rows[i]['ordinal'] for i in ids if a['quiet'][i] and pred[key][i]>=thr];rr=[s-rows[i]['ordinal'] for i in ids if a['quiet'][i] and pred[ref][i]>=rt];parts[p].append([int(bool(ll))-int(bool(rr)),max(ll,default=0)-max(rr,default=0)])
 rng=np.random.default_rng(2024);z=np.concatenate(list(parts.values()),axis=0).mean(0);bs=[np.concatenate([parts[p] for p in rng.choice(page,len(page),replace=True)],axis=0).mean(0) for _ in range(2000)];ci=np.quantile(bs,[.025,.975],axis=0);out['paired_gains'][key]={'event_recall_pp':float(100*z[0]),'recall_CI95_pp':(100*ci[:,0]).tolist(),'mean_lead_all_events_gain_days':float(z[1]),'lead_CI95_days':ci[:,1].tolist()}
 wrong=np.load(R/'controls'/(key+'_mismatch.npy'));out['mismatch'][key]=metric(rows,a,wrong,te,thr,True)
(R/'quiet_policy_secondary.json').write_text(json.dumps(out,indent=2))
with open(R/'quiet_comparison.csv','w') as f:
 w=csv.writer(f);w.writerow(['model','overall_test_AP_percent','quiet_test_FPR_percent','quiet_event_recall_percent','calendar_lead_mean_all39events_days','quiet_event_count','policy'])
 primary=json.load(open(R/'results.json'))
 for key,v in out['models'].items():w.writerow([key,100*primary['models'][key]['test']['AP'],v['FPR_percent'],v['event_recall_percent'],v['mean_lead_all_events_days'],v['event_count'],'1%validation-quiet-negative-FPR;secondary'])
print('KIMI CONDITIONAL',out['paired_gains']['kimi-vl-a3b-instruct']);print('MISMATCH',[(k,v['event_recall_percent'],v['FPR_percent']) for k,v in out['mismatch'].items()])
