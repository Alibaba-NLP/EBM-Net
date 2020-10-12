__author__ = 'Qiao Jin'

import json

result2label = {'significantly decreased': 0,\
				'no significant difference': 1,\
				'significantly increased': 2}

def generate(picoids, split):
	path = '%s.json' % split

	output = []
	for picoid in picoids:
		if prompt_info[picoid]['label'] != 'invalid prompt':
			output.append({})
			output[-1]['picoid'] = picoid
			output[-1]['pmcid'] = prompt_info[picoid]['PMCID']
			output[-1]['i_text'] = prompt_info[picoid]['I']
			output[-1]['c_text'] = prompt_info[picoid]['C']
			output[-1]['o_text'] = prompt_info[picoid]['O']
			output[-1]['label'] = result2label[prompt_info[picoid]['label']]

			passage = ''
			if str(prompt_info[picoid]['PMCID']) in pmcid2content:
				content = pmcid2content[str(prompt_info[picoid]['PMCID'])]
				for secname, text in content:
					if secname[:len('ABSTRACT')] != 'ABSTRACT': continue
					if sec2label[secname2sec[secname]] == '1':
						passage += text
			
			output[-1]['passage'] = passage

	with open(path, 'w') as f:
		json.dump(output, f, indent=4)


prompt_info = json.load(open('materials/prompt_info.json'))
split2ids = json.load(open('materials/split2ids.json'))
pmcid2picoid = json.load(open('materials/pmcid2picoid.json'))
pmcid2content = json.load(open('materials/pmc_contents.json'))

secname2sec = json.load(open('materials/secname2sec.json'))
sec2label = json.load(open('materials/sec2label.json'))

for split, ids in split2ids.items():
	picoids = []

	for pmcid in ids:
		pmcid = str(pmcid)
		if pmcid in pmcid2picoid:
			picoids += pmcid2picoid[pmcid]
	
	generate(picoids, split)
