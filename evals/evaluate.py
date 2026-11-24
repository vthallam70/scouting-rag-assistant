"""Evaluate answer keyword coverage against a user-provided JSONL file."""
import json,sys,urllib.request
items=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]; passed=0
for item in items:
 req=urllib.request.Request('http://127.0.0.1:8000/ask',data=json.dumps({'question':item['question']}).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req) as r: answer=json.load(r)['answer']
 ok=all(term.lower() in answer.lower() for term in item['expected_terms']); passed+=ok
 print(json.dumps({'question':item['question'],'pass':ok,'answer':answer}))
print(f'keyword coverage: {passed}/{len(items)}; manually review factual accuracy and citations')
