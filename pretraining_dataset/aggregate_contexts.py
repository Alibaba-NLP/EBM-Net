__author__ = 'Qiao Jin'

'''
codes to aggregate the contexts
only aggregate the needed contexts
'''

import glob
import json

ctxs = glob.glob('evidence/contexts_*')

pmids = set(json.load(open('evidence_pmids.json')))
bad_pmids = set(json.load(open('dict/bad_pmids.json')))

pmid2ctx = {}

for ctx in ctxs:
	print('Processing %s' % ctx)
	data = json.load(open(ctx))

	for item in data:
		pmid = item['pmid']
		ctx = item['ctx']

		if pmid not in pmids or pmid in bad_pmids: continue

		pmid2ctx[pmid] = ctx

with open('pmid2ctx.json', 'w') as f:
	json.dump(pmid2ctx, f, indent=4)
