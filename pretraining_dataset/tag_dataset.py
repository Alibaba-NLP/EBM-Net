__author__ = 'Qiao Jin'

from collections import Counter
import json
import glob

import nltk
from nltk import tokenize
from nltk.tag import StanfordPOSTagger
from nltk.corpus import stopwords

import os
import sys

import random as rd


'''
used to pseudo label the dataset
'''

st = StanfordPOSTagger('english-left3words-distsim.tagger', java_options='-mx4g')

exclude = set(['rather than', 'other than'])
ups = set(['better', 'greater', 'higher', 'later', 'more', 'faster', 'older', 'longer', \
		'larger', 'broader', 'wider', 'stronger', 'deeper', 'more', 'commoner', 'richer', \
		'further', 'bigger'])
downs = set(['worse', 'smaller', 'lower', 'earlier', 'less', 'slower', 'younger', 'shorter', \
		'smaller', 'narrower', 'narrower', 'weaker', 'shallower', 'fewer', 'rarer', 'poorer', \
		'closer', 'smaller'])

key_words = ups.union(downs)

diffs = set(['difference', 'differences', 'different', 'differently', 'differ'])
sims = set(['similar', 'similarly', 'similarity', 'similarities'])

nos = set(['no', 'not'])
middles = set(['significant', 'significantly', 'statistic', 'statistically', 'statistical'])

nums = set(["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"])

sec2label = json.load(open('sec2label.json'))

def mask_and_label(sent):
	sent = ' ' + sent 
	lower_sent = sent.lower()

	if ' than ' in lower_sent and all([exc not in lower_sent for exc in exclude]):
		words = tokenize.word_tokenize(sent)
		lowers = [word.lower() for word in words]
		words_ctr = Counter(lowers)

		if words_ctr['than'] == 1: # more than 1 are not useful (mostly describing only quantitative relations) 
			than_idx = lowers.index('than') 
			inter = set(lowers[:than_idx]).intersection(key_words)

			if len(inter) >= 1:
				up_indices = [1 if word.lower() in ups else 0 for word in words]
				down_indices = [1 if word.lower() in downs else 0 for word in words]

				if any(up_indices) and not any(down_indices):
					if than_idx + 1 < len(lowers) and (lowers[than_idx+1].isnumeric() or lowers[than_idx+1] in nums):
						pass
					else:
						indices = [idx_ for idx_, up in enumerate(up_indices) if up == 1] + [than_idx]
						final = words
						direction = 2

				elif any(down_indices) and not any(up_indices):
					if than_idx + 1 < len(lowers) and (lowers[than_idx+1].isnumeric() or lowers[than_idx+1] in nums):
						pass
					else:
						indices = [idx_ for idx_, down in enumerate(down_indices) if down == 1] + [than_idx]
						final = words
						direction = 0

	elif ' similar' in lower_sent and ' to ' in lower_sent:
		words = tokenize.word_tokenize(sent)
		lowers = [word.lower() for word in words]
		words_ctr = Counter(lowers)

		for idx, lower in enumerate(lowers):
			if lower in sims:
				sim_idx = idx
				break
		
		if 'sim_idx' in locals():
			if 'to' in lowers[sim_idx:]:
				to_idx = sim_idx + lowers[sim_idx:].index('to')

				indices = [sim_idx] + [to_idx]
				final = words
				direction = 1
	
	elif ' no' in lower_sent and ' differ' in lower_sent and 'and' in lower_sent:
		words = tokenize.word_tokenize(sent)
		lowers = [word.lower() for word in words]
		words_ctr = Counter(lowers)

		for idx, word in enumerate(lowers):
			if word in diffs:
				diff_idx = idx
				break
			
		if 'diff_idx' in locals():
			# first find the left no, then scan the middle words
			for i in range(idx):
				word = words[idx-1-i]
				if word in nos:
					no_idx = idx-1-i
					break

			if 'no_idx' in locals():
				bet_indices = [1 if word == 'between' and idx > diff_idx else 0 for idx, word in enumerate(lowers)]
				
				if any(bet_indices):
					first_bet = bet_indices.index(1)
					if 'and' in lowers[first_bet:]:
						and_idx = first_bet + lowers[first_bet:].index('and')
						indices = list(range(no_idx, diff_idx+1)) + [idx for idx, bet in enumerate(bet_indices) if bet == 1] + [and_idx]
						final = words
						direction = 1
	
	if 'final' in locals() and 'direction' in locals():
		if type(final) == list and len(final) > 0 and final[-1] != '?':
			return [final, direction, indices]
	
	else:
		return False


def process(item):
	# an item is an article
	# also need to save the context
	# as well as save the evidence
	pmid = item['pmid']
	texts = item['texts'] 
	labels = item['sec_labels']

	evi_output = []
	ctx_output = {'pmid': pmid, 'ctx': ''}

	bg_status = True

	for text, label in zip(texts, labels):
		if label == 'TITLE': continue
		sents = tokenize.sent_tokenize(text) 
		if not label or label not in sec2label:
			for sent in sents:
				result = mask_and_label(sent)
				if result:
					bg_status = False
					evi_output.append({'pmid': pmid, 'pos': result[0], 'label': result[1], 'indices': result[2]})
				else:
					if bg_status:
						ctx_output['ctx'] += ' ' + sent
		else:
			judge = sec2label[label]
			
			if judge == '1': # all background
				ctx_output['ctx'] += ' ' + text
			else:
				bg_status = False # starting no background
				for sent in sents:
					result = mask_and_label(sent)
					if result: evi_output.append({'pmid': pmid, 'pos': result[0], 'label': result[1], 'indices': result[2]})

	return evi_output, ctx_output

chunks = list(range(1, 1016))
rd.shuffle(chunks)

total = 0

for idx, chunk_id in enumerate(chunks):
	if not os.path.exists('evidence/evidence_pos_{0:04d}.json'.format(chunk_id)):
		data_path = 'pubmed20n{0:04d}.json'.format(chunk_id)
		if not os.path.exists(data_path): continue

		evi_output = []
		ctx_output = []
		data = json.load(open(data_path))

		for item in data:
			results = process(item)
			evi_output += results[0]
			ctx_output.append(results[1])

		pos_list = st.tag_sents(o['pos'] for o in evi_output)

		for _idx in range(len(evi_output)):
			evi_output[_idx]['pos'] = pos_list[_idx]

		with open('evidence/evidence_pos_{0:04d}.json'.format(chunk_id), 'w') as f:
			json.dump(evi_output, f)
		with open('evidence/contexts_{0:04d}.json'.format(chunk_id), 'w') as f:
			json.dump(ctx_output, f)

	else:
		evi_output = json.load(open('evidence/evidence_pos_{0:04d}.json'.format(chunk_id)))
	
	total += len(evi_output)
	
	print('%d/%d; Processing %s; Number of evidence: %d; Total: %d' % (idx+1, len(chunks), chunk_id, len(evi_output), total))
