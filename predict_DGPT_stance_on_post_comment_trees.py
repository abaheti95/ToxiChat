# We will use trained DGPT stance classifier to predict on the post-comment threads

import sys
sys.path.insert(0,'..')
sys.path.insert(0,'.')
import utils
from utils import RANDOM_SEED, log_list, print_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, save_in_json, load_from_json, format_time, plot_train_loss, log_TP_FP_FN_TN_from_binary_predictions, draw_and_save_precision_recall_curve, save_list_of_tuples_to_tsv
import pdb

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, GPT2Config, AdamW, get_linear_schedule_with_warmup,  AutoModelForCausalLM, AutoTokenizer
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import autograd
import torch
torch.manual_seed(RANDOM_SEED+1)

import random
random.seed(RANDOM_SEED)

import os
import re
import math
import time
import copy
import ast
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter
from sklearn import metrics

from sklearn.metrics import average_precision_score, precision_recall_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import functions and classes specific to SBF training and evaluation.
from SBF_utils import BertForSBF, SBF_BERT_Dataset, count_unique_posts, convert_string_label_to_binary, relabel_with_binarized_votes_and_create_BERT_instances, get_labels_dict_from_list
# Import functions and classes specific to OC_S training and evaluation.
from OC_S_utils import Conversation_Data, get_conversation_data_from_OC_S_file, get_save_lists_from_conv_data, OC_S_stance_Dataset, get_conversation_data_from_SBF_instances, log_TP_FP_FN_TN_from_conv_off_predictions, TARGET_GROUPS, TARGET_GROUPS_TO_ID, log_TP_FP_FN_TN_convs_from_stance_predictions, log_top_conv_stance_predictions

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", help="Path to the pickle file containing the post-threads on which we want to make predictions", type=str, required=True)
parser.add_argument("-sm", "--stance_model_dir", help="Path to the directory containing trained DGPT stance model and its tokenizer", type=str, required=True)
parser.add_argument("-om", "--offensive_model_dir", help="Path to the directory containing trained DGPT offensive model and its tokenizer", type=str, required=True)
parser.add_argument("-o", "--output_dir", help="Path to the output directory where we will save all the model prediction and results", type=str, required=True)
parser.add_argument("-s", "--save_file", help="Optional path to the output save file to save all the model prediction and results", type=str, default="")
parser.add_argument("-bs", "--batch_size", help="Train batch size for GPT2 model", type=int, default=32)
args = parser.parse_args()


import logging
# Ref: https://stackoverflow.com/a/49202811/4535284
for handler in logging.root.handlers[:]:
	logging.root.removeHandler(handler)
# Also add the stream handler so that it logs on STD out as well
# Ref: https://stackoverflow.com/a/46098711/4535284
make_dir_if_not_exists(args.output_dir)
logfile = os.path.join(args.output_dir, "output.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(logfile, mode='w'), logging.StreamHandler()])

# PRETRAINED_GPT2_MODEL = 'GPT2-base-cased'
PRETRAINED_GPT2_MODEL = 'microsoft/DialoGPT-medium'
# Other global constants required for the code
POSSIBLE_BATCH_SIZE = 1
MAX_SEQ_THRESH = 512

if torch.cuda.is_available():
	device = torch.device("cuda")
	logging.info(f"Using GPU{torch.cuda.get_device_name(0)} to make predictions")
else:
	device = torch.device("cpu")
	logging.info(f"Using CPU to make predictions")

def reweight(cls_num_list, beta=0.9999):
	'''
	Implement reweighting by effective numbers
	:param cls_num_list: a list containing # of samples of each class
	:param beta: hyper-parameter for reweighting, see paper for more details
	:return:
	'''
	per_cls_weights = None
	#############################################################################
	# TODO: reweight each class by effective numbers                            #
	#############################################################################
	n_is = np.array(cls_num_list)
	per_cls_weights = (1 - beta) / (1 - np.power(beta, n_is))
	per_cls_weights = torch.from_numpy(per_cls_weights)
	# per_cls_weights = per_cls_weights / per_cls_weights.sum() * 10
	#############################################################################
	#                              END OF YOUR CODE                             #
	#############################################################################
	return per_cls_weights

class FocalLoss(nn.Module):
	def __init__(self, weight=None, gamma=1.0):
		super(FocalLoss, self).__init__()
		assert gamma >= 0
		self.gamma = gamma
		self.weight = weight

	def forward(self, input, target):
		'''
		Implement forward of focal loss
		:param input: input predictions
		:param target: labels
		:return: tensor of focal loss in scalar
		'''
		loss = None
		zi = -input
		batch_size = input.size(0)
		zi[torch.arange(batch_size), target] *= -1
		pis = F.sigmoid(zi)
		first_term = (1-pis) ** self.gamma
		second_term = torch.log(pis)
		multipled = torch.einsum("bj,bj->b", (first_term, second_term))
		class_weights = self.weight[target].float().to(device)
		loss = -class_weights.dot(multipled)
		return loss


class GPT2ForOC_S_stance(GPT2LMHeadModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_off_labels = 2
		self.num_stance_labels = 3
		# logging.info(f"Number of off labels for GPT2ForOC_S_stance classifier = {self.num_off_labels}")
		# logging.info(f"Number of target labels for GPT2ForOC_S_stance classifier = {len(TARGET_GROUPS)}")
		logging.info(f"Number of stance labels for GPT2ForOC_S_stance classifier = {self.num_stance_labels}")
		self.dropout = nn.Dropout(config.embd_pdrop)
		# self.off_classifier = nn.Linear(config.hidden_size, self.num_off_labels)
		# self.target_classifier = nn.Linear(config.hidden_size, len(TARGET_GROUPS))
		self.stance_classifier = nn.Linear(config.hidden_size*4, self.num_stance_labels)
		# self.init_weights()
		config.focal_loss = True
		if config.focal_loss:
			# Instantiate using Focal loss
			weight = reweight(config.cls_num_list)
			self.stance_loss_fct = FocalLoss(weight=weight, gamma=1.0)
			logging.info(f"Using Class balanced focal loss with beta = 0.9999 and gamma = 1.0")
		else:
			# self.stance_loss_fct = nn.CrossEntropyLoss()
			# logging.info(f"Using Cross Entropy loss with no weights")
			# self.stance_loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 10.0]))
			# logging.info(f"Using Cross Entropy loss with weights [1.0, 10.0, 10.0]")
			self.stance_loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 100.0, 100.0]))
			logging.info(f"Using Cross Entropy loss with weights [1.0, 100.0, 100.0]")
		# self.target_loss_fct = nn.BCEWithLogitsLoss()
		# self.stance_loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 100.0, 100.0]))
		# self.stance_loss_multiplier = 2.0
	
	def forward(
		self,
		input_ids,
		eos_toward_token_ids=None,
		eos_response_token_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		stance_labels=None,
		# off_labels=None,
		# target_labels=None,
	):
		r"""
		labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
			Labels for computing the sequence classification/regression loss.
			Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
			If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
			If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

	Returns:
		:obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
		loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
			Classification (or regression if config.num_labels==1) loss.
		logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
			Classification (or regression if config.num_labels==1) scores (before SoftMax).
		hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
			Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
			of shape :obj:`(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the model at the output of each layer plus the initial embedding outputs.
		attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
			Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
			:obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

			Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
			heads.
		"""
		outputs = self.transformer(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)
		# Type of outputs = BaseModelOutputWithPastAndCrossAttentions
		# ref: https://huggingface.co/transformers/_modules/transformers/modeling_outputs.html#BaseModelOutputWithPastAndCrossAttentions
		GPT2_last_layer_output = outputs.last_hidden_state

		# Get the hidden representations for the EOS token ids
		eos_toward_token_representation = GPT2_last_layer_output[eos_toward_token_ids[0], eos_toward_token_ids[1], :]
		eos_response_token_representation = GPT2_last_layer_output[eos_response_token_ids[0], eos_response_token_ids[1], :]
		difference1 = eos_toward_token_representation - eos_response_token_representation
		hadamard = eos_toward_token_representation * eos_response_token_representation
		stance_classifier_input = torch.cat([eos_toward_token_representation, eos_response_token_representation, difference1, hadamard], axis=1)
		# Apply dropout
		stance_classifier_input = self.dropout(stance_classifier_input)
		# Compute stance logits from concatenated eos representations
		stance_logits = self.stance_classifier(stance_classifier_input)


		outputs = (stance_logits,) + outputs[2:]
		# If stance_labels given, compute loss from stance_logits
		
		loss = 0.0
		if stance_labels is not None:
			loss = self.stance_loss_fct(stance_logits.view(-1, self.num_stance_labels), stance_labels.view(-1))
			# print(f"input ids = {input_ids}, DGPT outputs shape = {GPT2_last_layer_output.size()} vs nan count = {torch.isnan(GPT2_last_layer_output).sum()}")
			# print(f"Off logits = {stance_logits} vs Off labels = {off_labels}")
			# if target_labels is not None:
			# 	# Some of the target_labels can still be None. We have to ignore loss for these target labels
			# 	for i, target_label in enumerate(target_labels):
			# 		if target_label is not None:
			# 			loss += self.target_loss_fct(target_logits[i], target_label.to(device))
			outputs = (loss,) + outputs

		return outputs  # (loss), logits, (hidden_states), (attentions)


def prepare_threads_for_stance_model_predictions(current_threads, tokenizer):
	all_GPT2_model_input_texts = list()
	gold_stance_u_id_pairs = list()
	per_instance_n_utterances = list()
	for i, (subreddit, post_thread) in enumerate(current_threads):
		GPT2_string = post_thread.replace(" EOS ", tokenizer.eos_token)
		all_GPT2_model_input_texts.append(GPT2_string)
		n_utterances = len([u for u in post_thread.split(" EOS ") if u])
		per_instance_n_utterances.append(n_utterances)
		# Create stance u_id_pairs
		for u_from in range(2, n_utterances+1):
			for u_to in range(1, u_from):
				gold_stance_u_id_pairs.append((i, u_to, u_from))

	# Tokenize
	all_GPT2_model_inputs_tokenized = tokenizer.batch_encode_plus(all_GPT2_model_input_texts, padding=True, add_special_tokens=False, return_tensors="pt")
	input_ids, attention_mask = all_GPT2_model_inputs_tokenized['input_ids'], all_GPT2_model_inputs_tokenized['attention_mask']
	# Extract the word_ids of CLS tokens i.e. the beginning of all the utterances
	eos_token_ids = (input_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)

	assert len(per_instance_n_utterances) == len(current_threads)
	# Convert the pad_token_ids to eos_token_ids as there is no pad token in DGPT model
	input_ids[input_ids==tokenizer.pad_token_id] = tokenizer.eos_token_id
	try:
		assert input_ids.size(1) < 512
	except AssertionError:
		logging.error(f"One of the instance has length longer than 512 tokens: {input_ids.shape}")
		logging.error(f"Skipping this batch!")
		return None

	# For stance labels create specific eos_token_ids for stance u_id pairs
	# Compute the per instance per utterance EOS ids
	per_instance_per_utterance_eos_ids = [list() for i in range(len(current_threads))]
	instance_ids = eos_token_ids[0].tolist()
	utterance_eos_ids = eos_token_ids[1].tolist()
	for instance_id, utterance_eos_id in zip(instance_ids, utterance_eos_ids):
		per_instance_per_utterance_eos_ids[instance_id].append(utterance_eos_id)
	# Using the creating list compute the eos_ids for stance u_id pairs
	stance_specific_instance_ids = list()
	eos_toward_token_ids = list()
	eos_response_token_ids = list()
	try:
		for instance_id, toward_u_id, response_u_id in gold_stance_u_id_pairs:
			stance_specific_instance_ids.append(instance_id)
			eos_toward_token_ids.append(per_instance_per_utterance_eos_ids[instance_id][toward_u_id-1])
			eos_response_token_ids.append(per_instance_per_utterance_eos_ids[instance_id][response_u_id-1])
	except IndexError:
		logging.error(f"Index error at {instance_id}, with {toward_u_id} and {response_u_id}")
		return None
	# Convert generated lists into tensors
	stance_specific_instance_ids = torch.LongTensor(stance_specific_instance_ids)
	eos_toward_token_ids = torch.LongTensor(eos_toward_token_ids)
	eos_response_token_ids = torch.LongTensor(eos_response_token_ids)
	# Convert token_ids into tuples for future processing
	eos_toward_token_ids = (stance_specific_instance_ids, eos_toward_token_ids)
	eos_response_token_ids = (stance_specific_instance_ids, eos_response_token_ids)
	return {"input_ids": input_ids, "eos_token_ids": eos_token_ids, "gold_stance_u_id_pairs": gold_stance_u_id_pairs, "eos_toward_token_ids": eos_toward_token_ids, "eos_response_token_ids": eos_response_token_ids, "input_str": all_GPT2_model_input_texts, "n_utterances": per_instance_n_utterances, "batch_threads": current_threads}

### Similar code for Offensive prediction

class GPT2ForOC_S_offensive(GPT2LMHeadModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_off_labels = 2
		self.num_stance_labels = 3
		logging.info(f"Number of off labels for GPT2ForOC_S_offensive classifier = {self.num_off_labels}")
		# logging.info(f"Number of target labels for GPT2ForOC_S_offensive classifier = {len(TARGET_GROUPS)}")
		# logging.info(f"Number of stance labels for GPT2ForOC_S_offensive classifier = {self.num_stance_labels}")
		self.dropout = nn.Dropout(config.embd_pdrop)
		self.off_classifier = nn.Linear(config.hidden_size, self.num_off_labels)
		# self.target_classifier = nn.Linear(config.hidden_size, len(TARGET_GROUPS))
		# self.stance_classifier = nn.Linear(config.hidden_size*4, self.num_stance_labels)
		# self.init_weights()
		self.loss_fct = nn.CrossEntropyLoss()
		# self.target_loss_fct = nn.BCEWithLogitsLoss()
		# self.stance_loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 100.0, 100.0]))
		# self.stance_loss_multiplier = 2.0
	
	def forward(
		self,
		input_ids,
		utterance_eos_ids,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		off_labels=None,
		# target_labels=None,
		# stance_labels=None,
		# eos_toward_token_ids=None,
		# eos_response_token_ids=None,
	):
		r"""
		labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
			Labels for computing the sequence classification/regression loss.
			Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
			If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
			If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

	Returns:
		:obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
		loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
			Classification (or regression if config.num_labels==1) loss.
		logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
			Classification (or regression if config.num_labels==1) scores (before SoftMax).
		hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
			Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
			of shape :obj:`(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the model at the output of each layer plus the initial embedding outputs.
		attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
			Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
			:obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

			Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
			heads.
		"""
		outputs = self.transformer(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)
		# Type of outputs = BaseModelOutputWithPastAndCrossAttentions
		# ref: https://huggingface.co/transformers/_modules/transformers/modeling_outputs.html#BaseModelOutputWithPastAndCrossAttentions
		GPT2_last_layer_output = outputs.last_hidden_state

		# Extract all EOS token representations from GPT2's last layer representations
		eos_token_representation = GPT2_last_layer_output[utterance_eos_ids[0], utterance_eos_ids[1], :]
		# Apply dropout on representations
		eos_token_representation = self.dropout(eos_token_representation)
		# Compute logits from cls representations
		off_logits = self.off_classifier(eos_token_representation)
		# target_logits = self.target_classifier(eos_token_representation)

		outputs = (off_logits,) + outputs[2:]
		# If off_labels given, compute loss from off_logits
		
		loss = 0.0
		if off_labels is not None:
			loss = self.loss_fct(off_logits.view(-1, self.num_off_labels), off_labels.view(-1))
			# print(f"input ids = {input_ids}, DGPT outputs shape = {GPT2_last_layer_output.size()} vs nan count = {torch.isnan(GPT2_last_layer_output).sum()}")
			# print(f"Off logits = {off_logits} vs Off labels = {off_labels}")
			# if target_labels is not None:
			# 	# Some of the target_labels can still be None. We have to ignore loss for these target labels
			# 	for i, target_label in enumerate(target_labels):
			# 		if target_label is not None:
			# 			loss += self.target_loss_fct(target_logits[i], target_label.to(device))
			outputs = (loss,) + outputs

		return outputs  # (loss), logits, (hidden_states), (attentions)

def prepare_threads_for_offensive_model_predictions(current_threads, tokenizer):
	all_GPT2_model_input_texts = list()
	per_instance_n_utterances = list()
	for i, (subreddit, post_thread) in enumerate(current_threads):
		GPT2_string = post_thread.replace(" EOS ", tokenizer.eos_token)
		all_GPT2_model_input_texts.append(GPT2_string)
		n_utterances = len([u for u in post_thread.split(" EOS ") if u])
		per_instance_n_utterances.append(n_utterances)

	# Tokenize
	all_GPT2_model_inputs_tokenized = tokenizer.batch_encode_plus(all_GPT2_model_input_texts, padding=True, add_special_tokens=False, return_tensors="pt")
	input_ids, attention_mask = all_GPT2_model_inputs_tokenized['input_ids'], all_GPT2_model_inputs_tokenized['attention_mask']
	# Extract the word_ids of CLS tokens i.e. the beginning of all the utterances
	eos_token_ids = (input_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)

	assert len(per_instance_n_utterances) == len(current_threads)
	# Convert the pad_token_ids to eos_token_ids as there is no pad token in DGPT model
	input_ids[input_ids==tokenizer.pad_token_id] = tokenizer.eos_token_id
	try:
		assert input_ids.size(1) < 512
	except AssertionError:
		logging.error(f"One of the instance has length longer than 512 tokens: {input_ids.shape}")
		logging.error(f"Skipping this batch!")
		return None

	return {"input_ids": input_ids, "eos_token_ids": eos_token_ids, "input_str": all_GPT2_model_input_texts, "n_utterances": per_instance_n_utterances, "batch_threads": current_threads}




def main():
	# Read the post-comments pickle file
	all_reddit_posts, all_reddit_post_id_to_index, all_reddit_posts_comments, all_reddit_comment_id_to_index, all_reddit_post_threads = load_from_pickle(args.input_file)

	# Iterate through post-threads and convert them into data on which the model can make prediction
	all_post_threads = list()
	count = 0
	for post_signature, comment_threads in all_reddit_post_threads.items():
		post_id, title, post, url, content_url = post_signature
		subreddit = url[3:].split("/",1)[0]
		subreddit_title_post = f"subreddit = {subreddit} Title:{title} \n {post}"
		for comment_thread in comment_threads:
			post_thread = f"{subreddit_title_post} EOS " + " EOS ".join([comment for comment_id, comment, comment_url in comment_thread]) + " EOS "
			all_post_threads.append((subreddit, post_thread))
		count += len(comment_threads)
		# MAX_CONVS = 10
		# if count > MAX_CONVS:
		# 	break
	logging.info(f"Total number of comment threads = {len(all_post_threads)}")

	# Load DGPT Stance model from a previously trained model
	logging.info(f"Loading pretrained Stance model and tokenizer from {args.stance_model_dir}...")
	stance_model = GPT2ForOC_S_stance.from_pretrained(args.stance_model_dir)
	stance_tokenizer = GPT2Tokenizer.from_pretrained(args.stance_model_dir)
	stance_model.to(device)

	# Load DGPT offensive model from a previously trained model
	logging.info(f"Loading pretrained Offensive model and tokenizer from {args.offensive_model_dir}...")
	offensive_model = GPT2ForOC_S_offensive.from_pretrained(args.offensive_model_dir)
	offensive_tokenizer = GPT2Tokenizer.from_pretrained(args.offensive_model_dir)
	offensive_model.to(device)

	# Iterate over the post-threads in batches and get their stance predictions
	stance_model.eval()
	offensive_model.eval()
	with torch.no_grad():
		final_post_threads_and_predictions = list()
		batch_size = args.batch_size
		n_post_threads = len(all_post_threads)
		pbar = tqdm(range(0, n_post_threads, batch_size))
		softmax_func = nn.Softmax(dim=1)
		for step, i in enumerate(pbar):
			start_index = i
			end_index = min(i+batch_size, n_post_threads)
			current_batch_post_threads = all_post_threads[start_index:end_index]
			current_batch_post_threads_and_predictions = [[subreddit, post_thread, {"stance":list(), "offensive":list()}] for subreddit, post_thread in current_batch_post_threads]

			# Get stance predictions for current threads
			batch_data = prepare_threads_for_stance_model_predictions(current_batch_post_threads, stance_tokenizer)
			if batch_data is None:
				continue
			input_dict = {"input_ids": batch_data["input_ids"].to(device)}
			input_dict["eos_toward_token_ids"] = batch_data["eos_toward_token_ids"]
			input_dict["eos_response_token_ids"] = batch_data["eos_response_token_ids"]
			# Forward
			stance_logits = stance_model(**input_dict)[0]
			# Apply softmax on the stance_logits			
			softmax_stance_logits = softmax_func(stance_logits).tolist()
			per_instance_n_utterances = batch_data["n_utterances"]
			# convert scores and id_pairs to per_instance scores and labels
			gold_stance_u_id_pairs = batch_data["gold_stance_u_id_pairs"]
			for index, (instance_id, to_u_id, from_u_id) in enumerate(gold_stance_u_id_pairs):
				current_batch_post_threads_and_predictions[instance_id][2]["stance"].append((to_u_id, from_u_id, softmax_stance_logits[index]))

			# Get offensive predictions for the current threads
			batch_data = prepare_threads_for_offensive_model_predictions(current_batch_post_threads, offensive_tokenizer)
			if batch_data is None:
				continue
			eos_token_ids = batch_data["eos_token_ids"]
			input_dict = {"input_ids": batch_data["input_ids"].to(device), "utterance_eos_ids": batch_data["eos_token_ids"]}
			# Forward
			off_logits = offensive_model(**input_dict)[0]
			softmax_off_logits = softmax_func(off_logits)

			assert softmax_off_logits.size(0) == eos_token_ids[0].size(0)
			softmax_off_logits = softmax_off_logits.tolist()
			# Convert eos_token_ids from tensor to list and zip
			eos_token_ids = (eos_token_ids[0].tolist(), eos_token_ids[1].tolist())
			prev_instance_id = -1
			for index, (instance_id, eos_token_id) in enumerate(zip(eos_token_ids[0], eos_token_ids[1])):
				if instance_id != prev_instance_id:
					prev_instance_id = instance_id
					u_id = 0
				else:
					u_id += 1
				current_batch_post_threads_and_predictions[instance_id][2]["offensive"].append((u_id, softmax_off_logits[index]))

			# Save both predictions in final list
			final_post_threads_and_predictions.extend(current_batch_post_threads_and_predictions)
	# Save the final predictions in a pickle file
	if args.save_file:
		save_file = args.save_file
	else:
		save_file = os.path.join(args.output_dir, "all_post_threads_stance_predictions.pkl")
	logging.info(f"Saving the stance predictions for {len(final_post_threads_and_predictions)} post-threads at {save_file}")
	save_in_pickle(final_post_threads_and_predictions, save_file)

if __name__ == '__main__':
	main()