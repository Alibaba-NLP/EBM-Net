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

import argparse
import glob
import json
import logging
import os
import random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
	AdamW,
	BertConfig,
	BertForSequenceClassification,
	BertTokenizer,
	get_cosine_schedule_with_warmup,
)

import models


logger = logging.getLogger(__name__)


def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
	return tensor.detach().cpu().tolist()


def train(args, train_picos, train_ctxs, model, tokenizer):
	""" Train the model """
	#tb_writer = SummaryWriter()

	args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	train_sampler = RandomSampler(train_picos)
	train_dataloader = DataLoader(train_picos, sampler=train_sampler, batch_size=args.train_batch_size)

	if args.max_steps > 0:
		t_total = args.max_steps
		args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
	else:
		t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay,
		},
		{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
	]
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
	scheduler = get_cosine_schedule_with_warmup(
		optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
	)

	# multi-gpu training 
	if args.n_gpu > 1:
		model = torch.nn.DataParallel(model)

	# Train!
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_picos))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
	logger.info(
		"  Total train batch size (w. parallel, distributed & accumulation) = %d",
		args.train_batch_size
		* args.gradient_accumulation_steps)
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)

	global_step = 0
	tr_loss, logging_loss = 0.0, 0.0
	model.zero_grad()
	train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
	set_seed(args)	# Added here for reproductibility
	for _ in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
		for step, batch in enumerate(epoch_iterator):
			model.train()

			batch = tuple(t.to(args.device) for t in batch)
			
			ctx_ids = to_list(batch[0])
			pico_token_ids = batch[1] # B x max_pico_length
			pico_token_mask = batch[2] # B x max_pico_length
			pico_segment_ids = batch[3] # B x max_pico_length
			labels = batch[4]

			ctx_batch = [train_ctxs[ctx_id] for ctx_id in ctx_ids] # B x list of ctx dataset
			ctx_batch = list(map(list, zip(*ctx_batch)))
			
			ctx_token_ids = torch.stack(ctx_batch[1]).to(args.device) # B x max_ctx_length
			ctx_token_mask = torch.stack(ctx_batch[2]).to(args.device) # B x max_ctx_length
			ctx_segment_ids = torch.stack(ctx_batch[3]).to(args.device) # B x max_ctx_length

			inputs = {
				"passage_ids": torch.cat([ctx_token_ids, pico_token_ids], dim=1),
				"passage_mask": torch.cat([ctx_token_mask, pico_token_mask], dim=1),
				"passage_segment_ids": torch.cat([ctx_segment_ids, pico_segment_ids], dim=1),
				"result_labels": labels
			}

			outputs = model(inputs)

			loss = outputs  # model outputs are always tuple in transformers (see doc)

			if args.n_gpu > 1:
				loss = loss.mean()	# mean() to average on multi-gpu parallel (not distributed) training
			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps

			loss.backward()

			tr_loss += loss.item()

			if (step + 1) % args.gradient_accumulation_steps == 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				model.zero_grad()
				global_step += 1

				if args.logging_steps > 0 and global_step % args.logging_steps == 0:
					# Log metrics
					#tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
					#tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
					#print((tr_loss - logging_loss) / args.logging_steps)
					logging_loss = tr_loss

				if args.save_steps > 0 and global_step % args.save_steps == 0:
					# Save model checkpoint
					output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
					if not os.path.exists(output_dir):
						os.makedirs(output_dir)
					model_to_save = (
						model.module if hasattr(model, "module") else model
					)  # Take care of distributed/parallel training
					model_to_save.save_pretrained(output_dir)
					tokenizer.save_pretrained(output_dir)
					torch.save(args, os.path.join(output_dir, "training_args.bin"))
					logger.info("Saving model checkpoint to %s", output_dir)

			if args.max_steps > 0 and global_step > args.max_steps:
				epoch_iterator.close()
				break
		if args.max_steps > 0 and global_step > args.max_steps:
			train_iterator.close()
			break

	#tb_writer.close()

	return global_step, tr_loss / global_step


def evaluate(args, eval_picos, eval_ctxs, model, tokenizer, prefix=""):

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
	# Note that DistributedSampler samples randomly
	eval_sampler = SequentialSampler(eval_picos)
	eval_dataloader = DataLoader(eval_picos, sampler=eval_sampler, batch_size=args.eval_batch_size)

	# Eval!
	logger.info("***** Running evaluation {} *****".format(prefix))
	logger.info("  Num examples = %d", len(eval_picos))
	logger.info("  Batch size = %d", args.eval_batch_size)

	example_ids = []
	all_labels = []
	all_preds = []
	all_logits = np.zeros((0, 3))

	for batch in tqdm(eval_dataloader, desc="Evaluating"):
		model.eval()
		batch = tuple(t.to(args.device) for t in batch)
		with torch.no_grad():
			ctx_ids = to_list(batch[0])
			pico_token_ids = batch[1] # B x max_pico_length
			pico_token_mask = batch[2] # B x max_pico_length
			pico_segment_ids = batch[3] # B x max_pico_length
			labels = batch[4]

			ctx_batch = [eval_ctxs[ctx_id] for ctx_id in ctx_ids] # B x list of ctx dataset
			ctx_batch = list(map(list, zip(*ctx_batch)))
			
			ctx_token_ids = torch.stack(ctx_batch[1]).to(args.device) # B x max_ctx_length
			ctx_token_mask = torch.stack(ctx_batch[2]).to(args.device) # B x max_ctx_length
			ctx_segment_ids = torch.stack(ctx_batch[3]).to(args.device) # B x max_ctx_length

			inputs = {
				"passage_ids": torch.cat([ctx_token_ids, pico_token_ids], dim=1),
				"passage_mask": torch.cat([ctx_token_mask, pico_token_mask], dim=1),
				"passage_segment_ids": torch.cat([ctx_segment_ids, pico_segment_ids], dim=1)
			}

			logits = model(inputs) # N x 3
			preds = torch.argmax(logits, dim=1) # N
			
			example_ids += list(batch[4].detach().cpu().numpy()) 
			all_labels += list(labels.detach().cpu().numpy())
			all_preds += list(preds.detach().cpu().numpy())
			all_logits = np.concatenate([all_logits, logits.detach().cpu().numpy()], axis=0)

	if not prefix:
		prefix = 'final'

	with open(os.path.join(args.output_dir, '%s_all_example_idx.json' % prefix), 'w') as f:
		json.dump([int(label) for label in example_ids], f)
	with open(os.path.join(args.output_dir, '%s_all_labels.json' % prefix), 'w') as f:
		json.dump([int(label) for label in all_labels], f)
	with open(os.path.join(args.output_dir, '%s_all_preds.json' % prefix), 'w') as f:
		json.dump([int(pred) for pred in all_preds], f)
	np.save(os.path.join(args.output_dir, '%s_all_logits.npy' % prefix), np.array(all_logits))
	
	results = {}
	results['f1'] = f1_score(all_labels, all_preds, average='macro') 
	results['acc'] = accuracy_score(all_labels, all_preds)

	return results


def represent(args, model, tokenizer):
	dataset = load_and_cache_examples(args, tokenizer, evaluate=True, do_repr=True)

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
	# Note that DistributedSampler samples randomly
	eval_sampler = SequentialSampler(dataset)
	eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

	# Eval!
	logger.info("***** Running Representations *****")
	logger.info("  Num examples = %d", len(dataset))
	logger.info("  Batch size = %d", args.eval_batch_size)

	example_ids = []
	all_reprs = np.zeros((0, model.bert.config.hidden_size))

	for batch in tqdm(eval_dataloader, desc="Evaluating"):
		model.eval()
		batch = tuple(t.to(args.device) for t in batch)
		with torch.no_grad():
			inputs = {
				"passage_ids": batch[0],
				"passage_mask": batch[1],
				"passage_segment_ids": batch[2],
			}

			reprs = model(inputs, get_reprs=True) # N x D
			
			example_ids += list(batch[4].detach().cpu().numpy()) 
			all_reprs = np.concatenate([all_reprs, reprs.detach().cpu().numpy()], axis=0)

	with open(os.path.join(args.output_dir, 'all_example_idx.json'), 'w') as f:
		json.dump([int(_id) for _id in example_ids], f)
	np.save(os.path.join(args.output_dir, 'all_reprs.npy'), np.array(all_reprs))


def load_and_cache_ctxs(args, tokenizer, evaluate=False, do_repr=False, pretraining=False):
	if args.pretraining:
		from utils_pretraining import (
			convert_ctxs_to_features,
			convert_picos_to_features,
			read_ctx_examples,
			read_pico_examples)
	else:
		from utils_ebmnet import (
			convert_ctxs_to_features,
			convert_picos_to_features,
			read_ctx_examples,
			read_pico_examples)

	# We need to index it

	# Load data features from cache or dataset file
	if do_repr:
		input_file = args.repr_ctx
	else:
		input_file = args.predict_ctx if evaluate else args.train_ctx

	cached_features_file = os.path.join(
		os.path.dirname(input_file),
		"cached_ctxs_adv{}_{}_{}".format(
			args.adversarial,
			"dev" if evaluate else "train",
			str(args.max_passage_length)
		),
	)

	if os.path.exists(cached_features_file) and not args.overwrite_cache:
		logger.info("Loading features from cached file %s", cached_features_file)
		features = torch.load(cached_features_file)
	else:
		logger.info("Creating features from dataset file at %s", input_file)

		examples = read_ctx_examples(input_file=input_file, adversarial=args.adversarial)

		features = convert_ctxs_to_features(
			examples=examples,
			tokenizer=tokenizer,
			max_passage_length=args.max_passage_length
		)

		logger.info("Saving features into cached file %s", cached_features_file)
		torch.save(features, cached_features_file)

	# Convert to Tensors and build dataset
	all_ctx_ids = torch.tensor([f.ctx_id for f in features], dtype=torch.long)
	all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

	dataset = TensorDataset(
		all_ctx_ids,
		all_input_ids,
		all_input_mask,
		all_segment_ids
	)

	return dataset


def load_and_cache_picos(args, tokenizer, evaluate=False, do_repr=False, pretraining=False):
	if args.pretraining:
		from utils_pretraining import (
			convert_ctxs_to_features,
			convert_picos_to_features,
			read_ctx_examples,
			read_pico_examples)
	else:
		from utils_ebmnet import (
			convert_ctxs_to_features,
			convert_picos_to_features,
			read_ctx_examples,
			read_pico_examples)
	# Dataset that we are going to use

	# Load data features from cache or dataset file
	if do_repr:
		input_file = args.repr_pico
	else:
		input_file = args.predict_pico if evaluate else args.train_pico

	cached_features_file = os.path.join(
		os.path.dirname(input_file),
		"cached_picos_adv{}_{}_{}_{}".format(
			args.adversarial,
			args.permutation,
			"dev" if evaluate else "train",
			str(args.max_pico_length)
		),
	)

	if os.path.exists(cached_features_file) and not args.overwrite_cache:
		logger.info("Loading features from cached file %s", cached_features_file)
		features = torch.load(cached_features_file)
	else:
		logger.info("Creating features from dataset file at %s", input_file)

		examples = read_pico_examples(input_file=input_file, adversarial=args.adversarial)

		features = convert_picos_to_features(
			examples=examples,
			tokenizer=tokenizer,
			max_pico_length=args.max_pico_length,
			permutation=args.permutation
		)

		logger.info("Saving features into cached file %s", cached_features_file)
		torch.save(features, cached_features_file)
		

	# Convert to Tensors and build dataset
	all_ctx_ids = torch.tensor([f.ctx_id for f in features], dtype=torch.long)
	all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

	mlm2cls = {}
	for i in range(34):
		if i < 15:
			mlm2cls[i] = 0
		elif 15 <= i < 17:
			mlm2cls[i] = 1
		else:
			mlm2cls[i] = 2

	if args.num_labels == 3 and args.pretraining: # here we have 34 labels to be processed
		all_labels = torch.tensor([mlm2cls[f.label] for f in features], dtype=torch.long)
	else:
		all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

	all_example_ids = torch.tensor([f.example_index for f in features], dtype=torch.long)

	dataset = TensorDataset(
		all_ctx_ids,
		all_input_ids,
		all_input_mask,
		all_segment_ids,
		all_labels,
		all_example_ids
	)

	return dataset


def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument(
		"--model_name_or_path",
		default=None,
		type=str,
		required=True,
		help='The path of the pre-trained model.'
	)
	parser.add_argument(
		"--output_dir",
		default=None,
		type=str,
		required=True,
		help="The output directory where the model checkpoints and predictions will be written.",
	)

	# Other parameters
	parser.add_argument(
		"--train_ctx", default=None, type=str, help="json file for training"
	)
	parser.add_argument(
		"--predict_ctx", default=None, type=str, help="json for predictions"
	)
	parser.add_argument(
		"--repr_ctx", default=None, type=str, help="json for representatins"
	)
	parser.add_argument(
		"--train_pico", default=None, type=str, help="json for training"
	)
	parser.add_argument(
		"--predict_pico", default=None, type=str, help="json for predictions"
	)
	parser.add_argument(
		"--repr_pico", default=None, type=str, help="json for representatins"
	)

	parser.add_argument(
		"--permutation",
		default="ioc",
		type=str,
		help="The sequence of intervention, comparison and outcome"
	)
	parser.add_argument(
		"--tokenizer_name",
		default="",
		type=str,
		help="Pretrained tokenizer name or path if not the same as model_name",
	)
	parser.add_argument(
		"--cache_dir",
		default="",
		type=str,
		help="Where do you want to store the pre-trained models downloaded from s3",
	)
	parser.add_argument(
		"--max_passage_length",
		default=256,
		type=int,
		help="max length of passage."
	)
	parser.add_argument(
		"--max_pico_length",
		default=128,
		type=int,
		help="max length of pico."
	)
	parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
	parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
	parser.add_argument("--do_repr", action="store_true", help="Whether to get representations")
	parser.add_argument(
		"--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
	)
	parser.add_argument(
		"--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
	)
	parser.add_argument("--per_gpu_train_batch_size", default=24, type=int, help="Batch size per GPU/CPU for training.")
	parser.add_argument(
		"--per_gpu_eval_batch_size", default=24, type=int, help="Batch size per GPU/CPU for evaluation."
	)
	parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
	parser.add_argument(
		"--gradient_accumulation_steps",
		type=int,
		default=1,
		help="Number of updates steps to accumulate before performing a backward/update pass.",
	)
	parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
	parser.add_argument(
		"--num_train_epochs", default=24.0, type=float, help="Total number of training epochs to perform."
	)
	parser.add_argument(
		"--max_steps",
		default=-1,
		type=int,
		help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
	)
	parser.add_argument("--warmup_steps", default=400, type=int, help="Linear warmup over warmup_steps.")
	parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
	parser.add_argument("--save_steps", type=int, default=25, help="Save checkpoint every X updates steps.")
	parser.add_argument(
		"--eval_all_checkpoints",
		action="store_true",
		help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
	)
	parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
	parser.add_argument(
		"--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
	)
	parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
	parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
	parser.add_argument("--pretraining", action="store_true", help='Whether to do pre-training')
	parser.add_argument("--num_labels", type=int, default=3, help='Number of labels at the last layer. Use 34 in pre-training and 3 in fine-tuning.')
	parser.add_argument("--adversarial", action="store_true", help='Whether using the adversarial setting.')
	args = parser.parse_args()

	args.overwrite_output_dir = True # always
	if (
		os.path.exists(args.output_dir)
		and os.listdir(args.output_dir)
		and args.do_train
		and not args.overwrite_output_dir
	):
		raise ValueError(
			"Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
				args.output_dir
			)
		)

	# Setup CUDA, GPU & distributed training
	device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
	args.n_gpu = torch.cuda.device_count()

	args.device = device

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
	)
	logger.warning(
		"Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
		args.local_rank,
		device,
		args.n_gpu,
		bool(args.local_rank != -1)
	)

	# Set seed
	set_seed(args)

	tokenizer = BertTokenizer.from_pretrained(
		args.model_name_or_path,
		do_lower_case=args.do_lower_case
	)

	model = models.EBM_Net(args, path=args.model_name_or_path) 
	model.to(args.device)

	logger.info("Training/evaluation parameters %s", args)

	# Save the trained model and the tokenizer
	# Create output directory if needed
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	# Training
	if args.do_train:
		train_ctxs = load_and_cache_ctxs(args, tokenizer, evaluate=False, pretraining=args.pretraining)
		train_picos = load_and_cache_picos(args, tokenizer, evaluate=False, pretraining=args.pretraining)
		global_step, tr_loss = train(args, train_picos, train_ctxs, model, tokenizer)
		logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

		logger.info("Saving model checkpoint to %s", args.output_dir)
		model.save_pretrained(args.output_dir)
		tokenizer.save_pretrained(args.output_dir)

		torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

	# Evaluation
	if args.do_eval:
		eval_ctxs = load_and_cache_ctxs(args, tokenizer, evaluate=True)
		eval_picos = load_and_cache_picos(args, tokenizer, evaluate=True)

		results = {}

		if args.do_train: # fine-tuning at least
			checkpoints = [args.output_dir]
		else: # zero-shot
			checkpoints = [args.model_name_or_path]	

		if args.eval_all_checkpoints:
			checkpoints = list(
				os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + 'full_model.bin', recursive=True))
			)

			logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

		logger.info("Evaluate the following checkpoints: %s", checkpoints)

		for checkpoint in checkpoints:
			# Reload the model
			if 'checkpoint' not in checkpoint:
				global_step = 'final'
			else:
				global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
			
			model = models.EBM_Net(args, path=checkpoint)
			model.to(args.device)
			# Evaluate

			result = evaluate(args, eval_picos, eval_ctxs, model, tokenizer, prefix=global_step)

			result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
			results.update(result)
			
			if 'checkpoint' in checkpoint and args.do_train and args.eval_all_checkpoints: # eval all setting
				os.remove(os.path.join(checkpoint, 'full_model.bin'))
				os.remove(os.path.join(checkpoint, 'pytorch_model.bin'))

		logger.info("Results: {}".format(results))
		with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
			results = {k: float(v) for k, v in results.items()}
			json.dump(results, f, indent=4)

	if args.do_repr:
		logger.info("Representing...")

		model = models.EBM_Net(args, path=args.model_name_or_path)
		model.to(args.device)

		represent(args, model, tokenizer)

if __name__ == "__main__":
	main()
