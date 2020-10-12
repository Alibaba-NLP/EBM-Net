__author__ = 'Qiao Jin' 

import json
import os

# process labels
ups = ['better', 'greater', 'higher', 'later', 'more', 'faster', 'older', 'longer', \
		'larger', 'broader', 'wider', 'stronger', 'deeper', 'more', 'commoner', 'richer', \
		'further', 'bigger']
downs = ['worse', 'smaller', 'lower', 'earlier', 'less', 'slower', 'younger', 'shorter', \
		'smaller', 'narrower', 'narrower', 'weaker', 'shallower', 'fewer', 'rarer', 'poorer', \
		'closer', 'smaller']
sims = ['nodiff', 'similar']

label_set =  list(set(downs)) + list(set(sims)) + list(set(ups))

label2idx = {label: idx for idx, label in enumerate(label_set)}
label2idx['similarly'] = label2idx['similar']
label2idx['similarity'] = label2idx['similar']
label2idx['similarities'] = label2idx['similar']
label2idx['farther'] = label2idx['further']

label2ctr = {k: 0 for k in list(label2idx)}

up2down = {label2idx[k]: label2idx[v] for k, v in zip(ups, downs)}
down2up = {label2idx[k]: label2idx[v] for k, v in zip(downs, ups)}

# start
output = []

removed = set(['CD']) 
indicators = set(['significant', 'significantly', 'statistically', 'statistic', '%'])
sims = set(['similar', 'similarly', 'similarity', 'similarities'])

pmids = set()

def reversed(words, label):
	
	all_rev = (words + ['MASK'])[::-1]

	mask_idx = [idx for idx, word in enumerate(all_rev) if word == '[MASK]']
	mask_idx = [0] + mask_idx + [len(all_rev)]

	for i, idx in enumerate(mask_idx[:-1]):
		all_rev[idx+1: mask_idx[i+1]] = all_rev[idx+1: mask_idx[i+1]][::-1]
	
	all_rev = all_rev[1:]

	if label in up2down:
		rev_label = up2down[label]
	elif label in down2up:
		rev_label = down2up[label]
	else:
		rev_label = label
	
	return all_rev, rev_label

def get_label(pos, indices, label2ctr):
	ind_words = [pos[ind][0] for ind in indices]

	if len(ind_words) == 2:
		label = ind_words[0].lower()
		if label not in label2idx:
			return False
		else:
			label2ctr[label] += 1
			return label2idx[label]	
	else:
		if ind_words[-1] == 'than':
			return False
		else:
			label2ctr['nodiff'] += 1
			return label2idx['nodiff']

for chunk_id in range(1, 1016):
	data_path = 'evidence/evidence_pos_{0:04d}.json'.format(chunk_id)
	if not os.path.exists(data_path): continue

	data = json.load(open(data_path))

	for item in data:
		# first detect parentheses
		pmid = item['pmid']
		pos = item['pos']
		indices = item['indices']

		label = get_label(pos, indices, label2ctr)

		if not label: continue # lose about ~20%

		par_stack = []
		idx_stack = []
		lefts = []
		rights = []
		for idx, info in enumerate(pos):
			if info[0] in {'(', ')'}:
				if not par_stack:
					if info[0] == ')': continue
					par_stack.append(info[0])
					idx_stack.append(idx)
				else:
					if par_stack[-1] == info[0]:
						par_stack.append(info[0])
						idx_stack.append(idx)
					else:
						par_stack = par_stack[:-1]
						lefts.append(idx_stack[-1])
						rights.append(idx)
						idx_stack = idx_stack[:-1]
		
		within_par = []
		if lefts and rights:
			for left, right in zip(lefts, rights):
				within_par += list(range(left, right+1))

		# detect irrelavent subsentences
		dot_indices = [idx for idx, info in enumerate(pos) if info[0] == ',']
		outer_idx = []
		if dot_indices:
			left, right = min(item['indices']), max(item['indices']) 
			# item['indices'] save the important indices
			dot_indices = [-1] + dot_indices + [len(pos)]
			# print(left, right, dot_indices)
			for i in range(len(dot_indices)-1):
				if dot_indices[i] <= left < dot_indices[i+1]:
					left_start = i
				if dot_indices[i] <= right < dot_indices[i+1]:
					right_start = i
			left = dot_indices[left_start]
			right = dot_indices[right_start+1]
			for i in range(len(pos)):
				if i <= left or i >= right:
					outer_idx.append(i)

		# detect irrelavent show that / suggest that	

		# RB before JJR in generally bad
		include_idx = []
		that_judged = False # only judge once
		for idx, i in enumerate(pos):
			if idx in outer_idx:
				#print(i, '----------OUT')
				pass
			elif idx+1 < len(pos) and (pos[idx+1][1] == 'JJR' or pos[idx+1][1] == 'RBR') and \
				((i[1] == 'RB' and i[0].lower() != 'not') \
				or i[0].lower() == 'times'):
				#print(i, '----------FRONT_RB')
				pass
			elif idx in item['indices']:
				#print(i, '----------DETECTED')
				pass
			elif i[1] in removed:
				#print(i, '----------TOREMOVE')
				pass
			elif i[0].lower() in indicators:
				#print(i, '----------INDICATOR')
				pass
			elif idx in within_par:
				#print(i, '----------INPAR')
				pass
			elif not that_judged and i[0].lower() == 'that':
				if idx < min(item['indices']):
					#print(i, '----------THAT')
					that_judged = True
					include_idx = []
			else:
				#print(i)
				include_idx.append(idx)

		final_evidence = []
		for idx, i in enumerate(pos):

			if idx in include_idx:
				final_evidence.append(i[0])
			else:
				if  final_evidence and final_evidence[-1] != '[MASK]':
					final_evidence.append('[MASK]')

		if not final_evidence: continue

		if final_evidence[-1] in ['.', '[MASK]']:
			final_evidence = final_evidence[:-1]

		# Make every word after [MASK] upper cased
		for idx, word in enumerate(final_evidence):
			if idx == 0 and word != '[MASK]':
				final_evidence[idx] = word[0].upper() + word[1:]
			elif word == '[MASK]' and idx + 1 < len(final_evidence) and final_evidence[idx+1]:
				final_evidence[idx+1] = final_evidence[idx+1][0].upper() + final_evidence[idx+1][1:]

		rev_evidence, rev_label = reversed(final_evidence, label)

		output.append({'pmid': pmid, 
				'pico': ' '.join(final_evidence), 'label': label,
				'rev_pico': ' '.join(rev_evidence), 'rev_label': rev_label})

		pmids.add(pmid)

	print('Processed chunk #%d. Got %d insts' % (chunk_id, len(output)))
	
with open('evidence.json', 'w') as f:
	json.dump(output, f, indent=4)

with open('evidence_pmids.json', 'w') as f:
	json.dump(list(pmids), f, indent=4)
