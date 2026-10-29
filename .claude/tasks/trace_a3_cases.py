import os,sys,json
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.stdout.reconfigure(encoding='utf-8');sys.path.insert(0,str(Path.cwd()))
import search_core as s
queries=['round basket for serving bread','long track speed skates','price of sp1 reed valve rebuild kit - 59-90154','inflatable towable tube for boating','10-in-1 air fryer toaster oven combo','rubber bag you fill with hot water to stay warm']
evalrows=[json.loads(l) for l in Path('data/eval_set.jsonl').open(encoding='utf-8') if l.strip()]
logs=[json.loads(l) for l in Path('data/rag_ablation_details.jsonl').open(encoding='utf-8') if l.strip()]
def rank(xs,pid):return next((i for i,(j,_) in enumerate(xs,1) if j==pid),None)
report=[]
for q in queries:
 item=next((r for r in evalrows if r['query'].lower()==q.lower()),None)
 if not item:item=next((r for r in evalrows if q.lower() in r['query'].lower() or r['query'].lower() in q.lower()),None)
 if item:q=item['query']; targets=item.get('target_ids',item['relevant_ids'])
 else:targets=[272]
 b=s._run_bm25(q,'text',2000); d=s._run_dense(q,'text',2000)
 f=s.rrf_fusion(b[:50],d[:50]); f30=s.rrf_fusion(b[:30],d[:30])
 row={'query':q,'targets':[],'top5':[],'rerank':{},'historical':[]}
 for pid in targets:row['targets'].append({'id':pid,'title':s._catalog[pid]['caption'],'bm25_rank':rank(b,pid),'bm25_score':next((v for i,v in b if i==pid),None),'dense_rank':rank(d,pid),'dense_score':next((v for i,v in d if i==pid),None),'rrf_rank':rank(f,pid),'rrf_score':next((v for i,v in f if i==pid),None),'rrf_pool30_rank':rank(f30,pid)})
 for pid,score in f[:5]:row['top5'].append({'id':pid,'title':s._catalog[pid]['caption'],'bm25_rank':rank(b,pid),'dense_rank':rank(d,pid),'rrf_score':score})
 for name,fn in [('minilm',s.rerank),('deberta',s.distilled_rerank)]:
  try:
   pool=[{'id':pid,'caption':s._catalog[pid]['caption'],'image_path':s._catalog[pid]['image_path'],'score':float(v)} for pid,v in f30[:30]]
   rr=fn(q,pool)
   row['rerank'][name]={'target_ranks':{str(pid):next((i for i,r in enumerate(rr,1) if r['id']==pid),None) for pid in targets},'top5':[{'id':r['id'],'title':r['caption'],'score':r['rerank_score']} for r in rr[:5]]}
  except Exception as e:row['rerank'][name]={'error':str(e)}
 for r in logs:
  if r['query']==q:row['historical'].append({k:r.get(k) for k in ['config','retrieved_ids','answer','context_recall','context_ndcg','judge_verdicts','final_query_used']})
 report.append(row)
 Path('.claude/tasks/a3-case-trace-2026-10-29.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'query':q,'targets':row['targets'],'rerank':{k:v.get('target_ranks',v) for k,v in row['rerank'].items()}},ensure_ascii=False),flush=True)
