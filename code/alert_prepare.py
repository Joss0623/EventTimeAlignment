import json,datetime as dt,hashlib,collections
from pathlib import Path
import numpy as np
B=Path('/home/home_new/qsmx');S=B/'data/EventTail/pilot01';D=B/'data/AttentionPIT/alert01';R=B/'runs/AttentionPIT/alert01'
D.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
def norm(s):return s.replace('_',' ').strip()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def day(s):return dt.date.fromisoformat(s).toordinal()
candidates=json.load(open(S/'event_candidates.json'));page_docs=collections.defaultdict(list);docs=[]
for c in candidates:
 k=len(docs);docs.append({'id':c['id'],'available':day(c['cutoff'][:10]),'text':'Category: '+c['category']+'\nContext: '+'; '.join(c['context'])+'\nReport: '+c['claim'],'category':c['category'],'archive_revid':c['archive_revid'],'archive_revision_time':c['archive_revision_time'],'source_urls':c['source_urls']})
 for title in set(map(norm,c['candidate_titles'])):page_docs[title].append(k)
panel=collections.defaultdict(dict);conflicts=[]
for f in sorted((S/'traffic').glob('*.json')):
 d=json.load(open(f));title=norm(d['title_at_request'])
 for v in d.get('items',[]):
  t=dt.datetime.strptime(v['timestamp'][:8],'%Y%m%d').date().toordinal();count=int(v['views'])
  if t in panel[title] and panel[title][t]!=count:conflicts.append([title,t,panel[title][t],count])
  panel[title][t]=count
assert not conflicts,str(conflicts[:3])
rows=[];xx=[];ys=[];first=[];quiet=[];docindices=[];all_onsets=[];ex=collections.Counter();categories=sorted({d['category'] for d in docs});meta=[]
for title,counts in sorted(panel.items()):
 if title not in page_docs:ex['no_verified_enrollment_title']+=1;continue
 dd=sorted(page_docs[title],key=lambda i:docs[i]['available']);enroll=docs[dd[0]]['available'];days=sorted(counts);cross={};onset={}
 for s in days:
  if not all(q in counts for q in range(s-28,s)):continue
  med=float(np.median([counts[q] for q in range(s-28,s)]));cross[s]=med>=100 and counts[s]>=max(2000,3*med)
 for s in days:
  if all(q in cross for q in range(s-7,s+1)):onset[s]=cross[s] and not any(cross[q] for q in range(s-7,s))
 for s,v in onset.items():
  if v and s>=enroll:all_onsets.append({'page':title,'day':s})
 for o in days:
  if o<enroll or not(day('2024-01-01')<=o<=day('2025-12-24')):continue
  if not all(q in counts for q in range(o-35,o+7)):ex['missing_contiguous42']+=1;continue
  if not all(q in cross for q in range(o-7,o)) or any(cross[q] for q in range(o-7,o)):ex['active_or_recent_spike']+=1;continue
  target=[s for s in range(o,o+7) if onset.get(s,False)];end=o+6
  if end<=day('2024-12-31'):split='train'
  elif o>=day('2025-01-01') and end<=day('2025-06-30'):split='val'
  elif o>=day('2025-07-01') and end<=day('2025-12-31'):split='test'
  else:ex['purged_boundary']+=1;continue
  hist=np.array([counts[q] for q in range(o-28,o)],dtype=float);med=float(np.median(hist));qflag=hist[-1]<=1.5*max(med,1) and hist[-3:].mean()<=1.5*max(med,1)
  available=[i for i in dd if o-28<=docs[i]['available']<=o]
  m=[len(available),min([o-docs[i]['available'] for i in available],default=29),sum(len(docs[i]['text']) for i in available)]+[sum(docs[i]['category']==c for i in available) for c in categories]
  rows.append({'page':title,'origin':dt.date.fromordinal(o).isoformat(),'ordinal':o,'split':split,'enrolled':dt.date.fromordinal(enroll).isoformat()});xx.append(hist);ys.append(bool(target));first.append(target[0] if target else -1);quiet.append(qflag);docindices.append(available);meta.append(m)
np.savez_compressed(D/'panel.npz',x=np.array(xx,dtype=np.float32),y=np.array(ys,dtype=np.int8),first=np.array(first),quiet=np.array(quiet),meta=np.array(meta,dtype=np.float32))
dump(D/'rows.json',rows);dump(D/'documents.json',docs);dump(D/'document_indices.json',docindices);dump(D/'all_observed_onsets.json',all_onsets)
stats={'raw_files':len(list((S/'traffic').glob('*.json'))),'raw_pages':len(panel),'eligible_pages':len(set(r['page'] for r in rows)),'rows':len(rows),'documents':len(docs),'splits':{},'exclusions':dict(ex),'text_scope':'168 pre-existing sampled-day CurrentEvents historical archives, not complete news or all daily archives','panel_scope':'All raw candidate-linked page intervals; enrollment at first as-of archived mention; no selection by future visits; gaps not imputed; not all-Wikipedia random sample'}
for split in ['train','val','test']:
 ii=np.array([i for i,r in enumerate(rows) if r['split']==split]);stats['splits'][split]={'rows':len(ii),'positive_rows':int(np.array(ys)[ii].sum()),'quiet_rows':int(np.array(quiet)[ii].sum()),'quiet_positive_rows':int((np.array(ys)[ii]*np.array(quiet)[ii]).sum()),'distinct_target_events':len(set((rows[i]['page'],first[i]) for i in ii if first[i]>=0)),'text_available_rows':sum(bool(docindices[i]) for i in ii),'pages':len(set(rows[i]['page'] for i in ii))}
protocol={'seed':2024,'history_days':28,'horizon_days':7,'origin':'00UTC; counts through yesterday, archived summaries available by cutoff<=origin','label':'first crossing on o..o+6, y>=max(2000,3*prior28median),prior28median>=100,no prior7 crossing','risk_set':'no crossing in prior7 days','quiet':'last observed count and last3mean <=1.5*observed28median','split':'train2024,val2025JanJun,test2025JulDec;purge target crossing boundaries','primary':'event recall and missed-inclusive lead at validation-fixed negative-day FPR1%; actual test FPR and PR-AUC','event_warning':'event counted once; earliest alert among available origins s-6..s;lead=s-o+1days from00UTC to end of targetday','selection':'max validation average precision;min30,max150,patience20','text':'available archived claims preceding28days; same documents; recency-weighted independently frozen contextual representations','input_identity':'page name grouping only,not classifier input','cohort_sha256':hashlib.sha256((D/'rows.json').read_bytes()).hexdigest()}
dump(R/'data_summary.json',stats);dump(R/'protocol.json',protocol);print(json.dumps(stats,indent=2))

# Prospective observation eligibility, independent of any later mention or future visits.
for r,idx in zip(rows,docindices):
 r['original_split']=r['split']
 if not idx or min(r['ordinal']-docs[j]['available'] for j in idx)>21:r['split']='excluded_no_prospective_coverage'
dump(D/'rows.json',rows)
stats['stored_origins_before_prospective_filter']=len(rows);stats['rows']=sum(r['split'] in ['train','val','test'] for r in rows);stats['excluded_no_prospective_coverage']=len(rows)-stats['rows'];stats['eligible_pages']=len({r['page'] for r in rows if r['split'] in ['train','val','test']})
for split in ['train','val','test']:
 ii=[i for i,r in enumerate(rows) if r['split']==split];stats['splits'][split]={'rows':len(ii),'positive_rows':int(np.array(ys)[ii].sum()),'quiet_rows':int(np.array(quiet)[ii].sum()),'quiet_positive_rows':int((np.array(ys)[ii]*np.array(quiet)[ii]).sum()),'distinct_target_events':len(set((rows[i]['page'],first[i]) for i in ii if first[i]>=0)),'text_available_rows':sum(bool(docindices[i]) for i in ii),'pages':len({rows[i]['page'] for i in ii})}
stats['prospective_monitoring']='Each already available mention opens22 origins, availability-day through+21; future mentions never add earlier origins.';protocol['monitoring']=stats['prospective_monitoring'];protocol['cohort_sha256']=hashlib.sha256((D/'rows.json').read_bytes()).hexdigest();dump(R/'data_summary.json',stats);dump(R/'protocol.json',protocol)
