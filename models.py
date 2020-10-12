import os
import torch
from torch import nn
from transformers import BertModel, BertConfig

class EBM_Net(nn.Module):
	def __init__(self, args, path=None):
		super(EBM_Net, self).__init__()
		self.args = args
		self.config = BertConfig.from_pretrained(args.model_name_or_path)
		self.bert = BertModel.from_pretrained(args.model_name_or_path)

		num_cls = 34
		self.res_linear = nn.Linear(self.config.hidden_size, num_cls)
		if args.num_labels == 3:
			self.final_linear = nn.Linear(num_cls, args.num_labels)

		self.relu = nn.ReLU()
		self.m = nn.LogSoftmax(dim=1)
		self.loss = nn.NLLLoss()
		self.softmax = nn.Softmax(dim=1)

		pretrained_dict = torch.load(os.path.join(path, 'full_model.bin'))
		model_dict = self.state_dict()
		to_load = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

		if len(to_load) != len(model_dict):
			# initialize the final layers
			down_weights = torch.tensor([1/15] * 15 + [0] * 2 + [-1/17] * 17)
			mid_weights = torch.tensor([0] * 15 + [1/2] * 2 + [0] * 17)
			up_weights = torch.tensor([-1/15] * 15 + [0] * 2 + [1/17] * 17)
			weights = [down_weights, mid_weights, up_weights]

			# borrow the shape
			to_load['final_linear.weight'] = model_dict['final_linear.weight'] 
			to_load['final_linear.bias'] = model_dict['final_linear.bias']

			for idx in range(3):
				to_load['final_linear.weight'][idx] = weights[idx] 
				to_load['final_linear.bias'][idx] = 0

		model_dict.update(to_load)
		self.load_state_dict(model_dict)


	def forward(self, inputs, get_reprs=False):
		cls_embeds = self.bert(input_ids=inputs['passage_ids'],
							   attention_mask=inputs['passage_mask'],
							   token_type_ids=inputs['passage_segment_ids'])[0][:, 0, :] # B x D

		if get_reprs:
			return cls_embeds
		
		if self.args.num_labels == 3:
			logits = self.final_linear(self.softmax(self.res_linear(cls_embeds))) # B x 3
		else:
			logits = self.res_linear(cls_embeds) # B x 34

		if 'result_labels' in inputs:
			return self.loss(self.m(logits), inputs['result_labels'])
		else:
			return logits

	def save_pretrained(self, path):
		# first save the model
		torch.save(self.state_dict(), os.path.join(path, 'full_model.bin'))
		self.bert.save_pretrained(path)
		# then save the config (vocab saved outside)
		self.config.save_pretrained(path)
