# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import json
import logging
import math
from tqdm import tqdm

from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

# Required by XLNet evaluation method to compute optimal threshold (see write_predictions_extended() method)
# from utils_squad_evaluate import find_all_best_thresh_v2, get_raw_scores, make_qid_to_has_ans


logger = logging.getLogger(__name__)

class CtxExample(object): # same with pre-training utils
	"""
	a single training/test example for the EBM-Net dataset.
	"""

	def __init__(
		self,
		ctx_id,
		passage_text
	):
		self.ctx_id = ctx_id
		self.passage_text = passage_text 
	
	def __str__(self):
		return self.__repr__()
	
	def __repr__(self):
		s = ""
		s += 'ctx_id: %s\n' % self.ctx_id
		s += "passage: %s\n" % self.passage_text

		return s


class CtxFeatures(object): # same with pre-training utils
	"""A single set of features of data."""

	def __init__(
		self,
		ctx_id,
		tokens,
		input_ids,
		input_mask,
		segment_ids
	):
		self.ctx_id = ctx_id
		self.tokens = tokens
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids


def read_ctx_examples(input_file, adversarial=False): # same with pre-training utils
	"""Read a EBM-Net json file into a list of EbmExample."""
	with open(input_file, "r", encoding="utf-8") as reader:
		input_data = json.load(reader)

	examples = []

	for entry in input_data:
		example = CtxExample(
			ctx_id=entry['ctx_id'],
			passage_text=entry['passage']
		)
		examples.append(example)

	return examples

def convert_ctxs_to_features(
	examples,
	tokenizer,
	max_passage_length,
	permutation=None,
	cls_token="[CLS]",
	sep_token="[SEP]",
	pad_token=0,
	sequence_a_segment_id=0,
	sequence_b_segment_id=1,
	cls_token_segment_id=0,
	pad_token_segment_id=0
):
	"""Loads a data file into a list of `InputBatch`s."""

	features = []
	for example in tqdm(examples):
		ctx_id = example.ctx_id
		psg_tokens = tokenizer.tokenize(example.passage_text)

		tokens = []
		segment_ids = []
		input_mask = []

		tokens += [cls_token]
		tokens += psg_tokens[:max_passage_length - 2]
		tokens += [sep_token]

		segment_ids = [sequence_a_segment_id] * len(tokens) 

		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length.
		while len(input_ids) < max_passage_length:
			input_ids.append(pad_token)
			input_mask.append(0)
			segment_ids.append(pad_token_segment_id)

		assert len(input_ids) == max_passage_length
		assert len(input_mask) == max_passage_length
		assert len(segment_ids) == max_passage_length

		if ctx_id < 20:
			logger.info("*** Example ***")
			logger.info("ctx_id: %s" % (ctx_id))
			logger.info("tokens: %s" % " ".join(tokens))
			logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

		features.append(
			CtxFeatures(
				ctx_id=ctx_id,
				tokens=tokens,
				input_ids=input_ids,
				input_mask=input_mask,
				segment_ids=segment_ids
			)
		)

	return features

# Starting pico utilities here, different between pre-training and final task

class PicoExample(object):
	"""
	a single training/test example for the EBM-Net dataset.
	"""

	def __init__(
		self,
		ctx_id,
		i_text,
		c_text,
		o_text,
		label
	):
		self.ctx_id = ctx_id
		self.i_text = i_text
		self.c_text = c_text
		self.o_text = o_text
		self.label = label
	
	def __str__(self):
		return self.__repr__()
	
	def __repr__(self):
		s = ""
		s += "ctx_id: %s\n" % self.ctx_id
		s += "i_text: %s\n" % self.i_text
		s += "c_text: %s\n" % self.c_text
		s += "o_text: %s\n" % self.o_text
		s += "label: %s\n" % self.label

		return s


class PicoFeatures(object): # unchanged from pre-training utils
	"""A single set of features of data."""

	def __init__(
		self,
		example_index,
		ctx_id,
		tokens,
		input_ids,
		input_mask,
		segment_ids,
		label
	):
		self.example_index = example_index
		self.ctx_id = ctx_id
		self.tokens = tokens
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label = label


def read_pico_examples(input_file, adversarial=False):
	"""Read a EBM-Net json file into a list of EbmExample."""
	with open(input_file, "r", encoding="utf-8") as reader:
		input_data = json.load(reader)

	examples = []

	for entry in input_data:
		example = PicoExample(
			ctx_id=entry['ctx_id'],
			i_text=entry['i_text'],
			c_text=entry['c_text'],
			o_text=entry['o_text'],
			label=entry['label']
		)
		examples.append(example)

		if adversarial:
			example = PicoExample(
				ctx_id=entry['ctx_id'],
				i_text=entry['c_text'],
				c_text=entry['i_text'],
				o_text=entry['o_text'],
				label=2-entry['label']
			)
			examples.append(example)

	return examples

def convert_picos_to_features(
	examples,
	tokenizer,
	max_pico_length,
	permutation=None,
	cls_token="[CLS]",
	sep_token="[SEP]",
	pad_token=0,
	sequence_a_segment_id=0,
	sequence_b_segment_id=1,
	cls_token_segment_id=0,
	pad_token_segment_id=0
):
	"""Loads a data file into a list of `InputBatch`s."""

	features = []
	example_index = 0

	if '-' in permutation: # shifting
		perm_list = permutation.split('-')
	else:
		perm_list = [permutation]

	for perm in perm_list:
		for (example_index, example) in enumerate(examples):
			ctx_id = example.ctx_id

			i_tokens = tokenizer.tokenize(example.i_text)
			c_tokens = tokenizer.tokenize(example.c_text)
			o_tokens = tokenizer.tokenize(example.o_text)
			ico_tokens = {'i': i_tokens,
						  'c': c_tokens,
						  'o': o_tokens}

			tokens = []
			segment_ids = []
			input_mask = []
			label = example.label

			assert set(perm).issubset({'i', 'o', 'c'})
			for element in perm:
				tokens += ico_tokens[element] + ['[MASK]'] 
			tokens[-1] = sep_token
			segment_ids = [sequence_b_segment_id] * len(tokens)

			if len(tokens) > max_pico_length:
				tokens = tokens[:max_pico_length-1] + [sep_token]
				segment_ids = segment_ids[:max_pico_length-1] + [sequence_b_segment_id]

			input_ids = tokenizer.convert_tokens_to_ids(tokens)
			input_mask = [1] * len(input_ids)

			# Zero-pad up to the sequence length.
			while len(input_ids) < max_pico_length:
				input_ids.append(pad_token)
				input_mask.append(0)
				segment_ids.append(pad_token_segment_id)

			assert len(input_ids) == max_pico_length
			assert len(input_mask) == max_pico_length
			assert len(segment_ids) == max_pico_length

			if example_index < 20:
				logger.info("*** Example ***")
				logger.info("example_index: %s" % (example_index))
				logger.info("tokens: %s" % " ".join(tokens))
				logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
				logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
				logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

			features.append(
				PicoFeatures(
					ctx_id=ctx_id,
					example_index=example_index,
					tokens=tokens,
					input_ids=input_ids,
					input_mask=input_mask,
					segment_ids=segment_ids,
					label=label
				)
			)

	return features
