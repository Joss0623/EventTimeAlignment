import os
os.environ.update(ATTENTION_PIT_DATA_ROOT='/home/home_new/qsmx/data/AttentionPIT/expanded01',ATTENTION_PIT_RUN_ROOT='/home/home_new/qsmx/runs/AttentionPIT/expanded01',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import json,datetime as dt,collections,hashlib,html
import numpy as np
from run import load,B,D,R,dump
G=D/'neighbors01';O=R/'neighbors01';assert (G/'ready.json').exists();specs=json.load(open(G/'graph_specs.json'));rows,x,y=load();requests=json.load(open(G/'requests.json'));bytitle={z['title']:z for z in requests};series=collections.defaultdict(dict);available={}
def norm(t):
 t=html.unescape(t).replace('_',' ').strip();return t[:1].upper()+t[1:]
for f in (B/'data/EventTail/pilot01/traffic').glob('*.json'):
 d=json.load(open(f));title=norm(d['title_at_request'])
 for v in d.get('items',[]):series[title][dt.datetime.strptime(v['timestamp'][:8],'%Y%m%d').date().isoformat()]=int(v['views'])
for req in requests:
 d=json.load(open(G/'traffic'/(req['key']+'.json')));available[req['title']]=d['available']
 if d['available']:
  for date,count in d['daily'].items():series[req['title']].setdefault(date,count)
counts=np.zeros((len(rows),5,28),np.float32);counts[:,0]=x;mask=np.zeros((len(rows),5),bool);mask[:,0]=True;wrong=counts.copy();wmask=mask.copy();donor_records=[];audit=[];catalogue=json.load(open(G/'title_catalogue.json'));title_to_index={t:i for i,t in enumerate(catalogue)};title_index=np.full((len(rows),5),-1,dtype=np.int32);wrong_title_index=title_index.copy()
for si,spec in enumerate(specs):
 ids=spec['row_indices'];early=next(i for i in ids if rows[i]['lead']==7);late=next(i for i in ids if rows[i]['lead']==1);start=dt.date.fromisoformat(rows[early]['origin'][:10])-dt.timedelta(days=28);end=dt.date.fromisoformat(rows[late]['origin'][:10])-dt.timedelta(days=1);dates=[(start+dt.timedelta(days=j)).isoformat() for j in range((end-start).days+1)];earlydates=dates[:28];excluded={norm(spec['title']),*spec['neighbors']};pool=[]
 for title,daily in series.items():
  if title in excluded or not all(d in daily for d in dates):continue
  h=np.array([daily[d] for d in earlydates],float)
  if h.sum()==0:continue
  scale=np.log1p(np.median(h));trend=np.log((h[-7:].mean()+1)/(h[:7].mean()+1));pool.append((title,scale,trend))
 donors=[]
 for j,title in enumerate(spec['neighbors'],1):
  for i in ids:title_index[i,j]=title_to_index[title]
  if not available.get(title,False):donors.append(None);continue
  h=np.array([series[title][d] for d in earlydates],float);scale=np.log1p(np.median(h));trend=np.log((h[-7:].mean()+1)/(h[:7].mean()+1));choices=[z for z in pool if z[0] not in donors]
  assert choices,(spec['title'],spec['graph_origin'],'no historical donor')
  donor=min(choices,key=lambda z:((z[1]-scale)**2+.25*(z[2]-trend)**2,hashlib.sha256(('2024|'+z[0]).encode()).hexdigest()))[0];donors.append(donor)
  for i in ids:
   origin=dt.date.fromisoformat(rows[i]['origin'][:10]);dd=[(origin-dt.timedelta(days=d)).isoformat() for d in range(28,0,-1)];assert max(dd)<rows[i]['origin'][:10]
   counts[i,j]=[series[title][d] for d in dd];mask[i,j]=True;wrong[i,j]=[series[donor][d] for d in dd];wmask[i,j]=True;wrong_title_index[i,j]=title_to_index[donor]
 donor_records.append({'event_id':spec['event_id'],'kind':spec['kind'],'graph_origin':spec['graph_origin'],'true_neighbors':spec['neighbors'],'nonlinked_past_matched_donors':donors,'matching':'logmedian and7daytrend from earliest7day historical input only; candidate histories cover all3origins, no target labels used'})
 if (si+1)%50==0:print('ASSEMBLE',si+1,'/',len(specs),flush=True)
assert np.array_equal(mask,wmask);assert np.array_equal(counts[:,0],x);assert np.isfinite(counts).all() and np.isfinite(wrong).all();assert (counts>=0).all()
np.savez_compressed(G/'histories.tmp.npz',counts=counts,mask=mask,wrong_counts=wrong,wrong_mask=wmask,title_index=title_index,wrong_title_index=wrong_title_index);(G/'histories.tmp.npz').replace(G/'histories.npz');dump(G/'donor_records.json',donor_records);stats={'rows':len(rows),'target_windows':len(specs),'unique_neighbor_titles':len(requests),'unavailable_neighbor_titles':sum(not v for v in available.values()),'degree_rows':{str(k):int((mask[:,1:].sum(1)==k).sum()) for k in range(5)},'all_targets_retained':True,'target_histories_identical':True,'all_neighbor_observations_before_origin':True,'graph_fixed_from_earliest_revision':True,'wrong_neighbor_mask_identical':True,'wrong_matching':'same calendar dates, initial-history-only scale/trend matching, nonlinked titles, identities fixed across3origins'};dump(O/'data_audit.json',stats);print(json.dumps(stats,indent=2))
