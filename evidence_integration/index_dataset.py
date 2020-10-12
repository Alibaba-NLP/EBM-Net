__author__ = 'Qiao Jin'

import json

splits = ['train', 'validation', 'test']

for split in splits:
	ori_data = json.load(open('%s.json' % split))
	picos = []
	ctxs = []
	pmcid2ctxid = {}


	for entry in ori_data:
		pico = {k: entry[k] for k in ['i_text', 'c_text', 'o_text', 'label']}
		pmcid = entry['pmcid']

		if pmcid not in pmcid2ctxid:
			pmcid2ctxid[pmcid] = len(ctxs)
			ctx = {'ctx_id': pmcid2ctxid[pmcid], 'passage': entry['passage']}
			ctxs.append(ctx)

		pico['ctx_id'] = pmcid2ctxid[pmcid]
		picos.append(pico)
	
	with open('indexed_%s_picos.json' % split, 'w') as f:
		json.dump(picos, f, indent=4)
	with open('indexed_%s_ctxs.json'% split, 'w') as f:
		json.dump(ctxs, f, indent=4)
