import json, math, re, sys
from collections import Counter
from pathlib import Path
TOKEN=re.compile(r"[a-z0-9]+")
def words(text): return Counter(TOKEN.findall(text.lower()))
def search(documents, query, k=3):
    q=words(query); n=len(documents); df=Counter()
    for d in documents: df.update(words(d['text']).keys())
    def vector(counts): return {t:(1+math.log(c))*(math.log((n+1)/(df[t]+1))+1) for t,c in counts.items()}
    qv=vector(q); qnorm=math.sqrt(sum(x*x for x in qv.values()))
    scores=[]
    for d in documents:
        v=vector(words(d['text'])); norm=math.sqrt(sum(x*x for x in v.values()))
        score=sum(w*v.get(t,0) for t,w in qv.items())/(qnorm*norm) if qnorm and norm else 0
        if score>0: scores.append((score,d))
    return sorted(scores,key=lambda x:x[0],reverse=True)[:k]
if __name__=='__main__':
    index=Path('index.json')
    if sys.argv[1]=='index':
        docs=[{'source':str(p),'text':chunk.strip()} for p in Path(sys.argv[2]).glob('*.txt') for chunk in p.read_text().split('\n\n') if chunk.strip()]; index.write_text(json.dumps(docs)); print(f'Indexed {len(docs)} passages')
    elif sys.argv[1]=='ask':
        for score,d in search(json.loads(index.read_text()),sys.argv[2]): print(f"[{d['source']}, {score:.3f}] {d['text']}")
