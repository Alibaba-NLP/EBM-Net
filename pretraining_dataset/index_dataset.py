__author__ = 'Qiao Jin'

import json

pmid2ctxid = {}

pmid2ctx = json.load(open('pmid2ctx.json'))
evidence = json.load(open('evidence.json'))

indexed_evidence = []
indexed_contexts = []

for entry in evidence:
	pmid = entry['pmid']
	if pmid not in pmid2ctx: continue

	if pmid not in pmid2ctxid:
		pmid2ctxid[pmid] = len(pmid2ctxid)
		indexed_contexts.append({'passage': pmid2ctx[pmid], 'ctx_id': pmid2ctxid[pmid]})
	
	entry['ctx_id'] = pmid2ctxid[pmid] 

	indexed_evidence.append(entry)

with open('indexed_evidence.json', 'w') as f:
	json.dump(indexed_evidence, f, indent=4)
with open('indexed_contexts.json', 'w') as f:
	json.dump(indexed_contexts, f, indent=4)
