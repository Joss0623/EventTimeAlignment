import os
os.environ.update(ATTENTION_PIT_DATA_ROOT='/home/home_new/qsmx/data/AttentionPIT/expanded01',ATTENTION_PIT_RUN_ROOT='/home/home_new/qsmx/runs/AttentionPIT/expanded01',PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0')
import numpy as np,json
from collections import Counter
from run import load,read_npz,snapshot_key,D,R,plain,dump
rows,_,_=load();stats={};details=[]
for split in ['train','val','test']:
 ids=[i for i,r in enumerate(rows) if r['split']==split and r['kind']=='event' and r['lead']==1];s={'events':len(ids),'revision_changed':0,'raw_wikitext_changed':0,'clean_full_text_changed':0,'retained_input_changed':0,'clean_changed_but_input_unchanged':0};relative=[]
 for i in ids:
  r=rows[i];j=next(j for j,w in enumerate(rows) if w['event_id']==r['event_id'] and w['kind']==r['kind'] and w['lead']==7);a=json.load(open(D/'snapshots'/(snapshot_key(r)+'.json')));b=json.load(open(D/'snapshots'/(snapshot_key(rows[j])+'.json')));ra=str(read_npz(D/'representations'/(snapshot_key(r)+'.npz'))['text']);rb=str(read_npz(D/'representations'/(snapshot_key(rows[j])+'.npz'))['text']);ca=plain(a['wikitext'] or 'No historical article text was available.');cb=plain(b['wikitext'] or 'No historical article text was available.');s['revision_changed']+=a['revid']!=b['revid'];s['raw_wikitext_changed']+=a['wikitext']!=b['wikitext'];s['clean_full_text_changed']+=ca!=cb;s['retained_input_changed']+=ra!=rb;s['clean_changed_but_input_unchanged']+=ca!=cb and ra==rb
  c1=Counter(ra.lower().split());c0=Counter(rb.lower().split());lexdiff=(sum((c1-c0).values())+sum((c0-c1).values()))/max(sum(c1.values())+sum(c0.values()),1)
  if ra!=rb:relative.append(lexdiff)
  details.append({'split':split,'row':i,'old_row':j,'pageid':r['pageid'],'origin':r['origin'],'revision_changed':a['revid']!=b['revid'],'clean_full_changed':ca!=cb,'retained_changed':ra!=rb,'retained_whitespace_token_bag_L1_fraction':lexdiff,'clean_full_words':len(ca.split()),'retained_words':len(ra.split())})
 s['retained_lexical_change_among_changed']={'mean':float(np.mean(relative)),'median':float(np.median(relative))} if relative else None;stats[split]=s
dump(R/'posttrain01/mechanism_ablation01/input_audit.json',{'scope':'asof historical inputs only; no future used','comparison':'current lead1 versus lead7 for same event target','statistics':stats,'events':details});print(json.dumps(stats,indent=2))
