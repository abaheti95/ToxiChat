# We will run the DGPT model on the pickle file containing test threads for CTG experiments

from utils import RANDOM_SEED, log_list, print_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, save_in_json, load_from_json, format_time, plot_train_loss, get_number_of_lines
import pdb

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, GPT2Config, AdamW, get_linear_schedule_with_warmup,  AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(RANDOM_SEED)

import random
random.seed(RANDOM_SEED)

import os
import re
import math
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter
from sklearn import metrics
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_file", help="Path to the pickle file for which we want to make DGPT predictions", type=str, required=True)
parser.add_argument("-o", "--output_file", help="Path to the pickle file where we will save DGPT outputs", type=str, required=True)
parser.add_argument("-m", "--model_dir", help="Path to the directory containing pretrained DGPT model", type=str, required=True)
parser.add_argument("-pre", "--preamble", help="Preamble string to the conversation. Will be added as an utterance before the comment thread", type=str, default="")
parser.add_argument("-p", "--prefix", help="Prefix string to the responses. Will be added before generating response. Useful in CTG experiments", type=str, default="")
parser.add_argument("-n", "--num_samples", help="Number of samples for each input", type=int, default=5)
parser.add_argument("-bs", "--batch_size", help="Specifies the number of sentences that should be predicted at once", type=int, default=32)
parser.add_argument("-sm", "--stance_model_dir", help="Path to the directory containing trained DGPT stance model and its tokenizer", type=str, required=True)
parser.add_argument("-om", "--offensive_model_dir", help="Path to the directory containing trained DGPT offensive model and its tokenizer", type=str, required=True)
args = parser.parse_args()

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if torch.cuda.is_available():
	device = torch.device("cuda")
	logging.info(f"Using GPU{torch.cuda.get_device_name(0)} to make predictions")
else:
	device = torch.device("cpu")
	logging.info(f"Using CPU to make predictions")

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
	""" Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
		Args:
			logits: logits distribution shape (vocabulary size)
			top_k >0: keep only top k tokens with highest probability (top-k filtering).
			top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
				Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
	"""
	assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear

	if top_p > 0.0:
		sorted_logits, sorted_indices = torch.sort(logits, descending=True)
		cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

		# Remove tokens with cumulative probability above the threshold
		sorted_indices_to_remove = cumulative_probs > top_p
		# Shift the indices to the right to keep also the first token above the threshold
		sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
		sorted_indices_to_remove[..., 0] = 0

		indices_to_remove = sorted_indices[sorted_indices_to_remove]
		logits[indices_to_remove] = filter_value
	return logits

def get_nucleus_sampling_generations_from_model(file, model, tokenizer, device, preamble, prefix):
	# We will implement custom batch nucleus sampling decoding while using the past variable. 
	# We will start generating with the smallest sequence and finish updating when all the sequences generate EOS tokens.

	all_input_generations = list()
	threads = load_from_pickle(file)
	# Create tqdm progressbar
	n_lines = len(threads)
	pbar = tqdm(threads, total=n_lines)
	# Setting model to eval for predictions
	# NOTE: assuming that model is already in the given device
	model.eval()
	
	with torch.no_grad():
		current_batch = list()
		for idx, line in enumerate(pbar):
			# # TEMP: cutting off predictions early for debugging purposes
			# if idx > 40:
			# 	break
			line = line.strip()
			# Accumulate lines to populate current batch
			# First add preamble to the line if preamble present
			if preamble:
				line = f"{preamble.strip()} EOS {line}"
			if not prefix:
				prefix = ""
			line = f"{line} {prefix}"
			current_batch.append(line.replace(" EOS ", tokenizer.eos_token))
			if len(current_batch) == args.batch_size or idx == (n_lines-1):
				# Make predictions and save them
				current_batch_saved_generations = [[line.replace(tokenizer.eos_token, " EOS "), list()] for line in current_batch]
				for _ in range(args.num_samples):
					# Tokenize the inputs in the batch and create input_ids and attention_mask for the model
					# Ref: https://github.com/huggingface/transformers/issues/3021
					token_ids = [tokenizer.encode(post, add_special_tokens=True) for post in current_batch]
					input_lengths = [len(s) for s in token_ids]
					max_seq_len = max(input_lengths)
					min_seq_len = min(input_lengths)
					input_lengths = torch.tensor(input_lengths).long().to(device)
					# logging.info(f"max_seq_len = {max_seq_len}, min_seq_len = {min_seq_len}")
					encodings_dict = tokenizer.batch_encode_plus(current_batch, max_length=max_seq_len, padding=True)
					# ideally we should be able to just input the following two variables to the function model.generate() ... => to be implemented soon!  # noqa: E501
					input_ids = torch.tensor(encodings_dict['input_ids']).to(device)
					
					attn_mask = torch.tensor(encodings_dict['attention_mask']).to(device)
					pad_token_id = tokenizer.eos_token_id
					eos_token_id = tokenizer.eos_token_id
					eos_not_in_sents = torch.ones(input_ids.shape[0]).long().to(device)
					decode_flag_sents = torch.zeros(input_ids.shape[0]).long().to(device)
					
					# we need to get the token ids of the last non-padded value
					last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1
					start_idx = inp_idx = (last_non_masked_idx).view(-1, 1).repeat(1, tokenizer.vocab_size).unsqueeze(1)
					past = None

					# Decode until all EOS found
					step = min_seq_len
					current_input_ids = input_ids[:, :min_seq_len]
					generation_ids = current_input_ids.clone()
					while eos_not_in_sents.float().sum().item() != 0.0 and step < 500:
						outputs = model(current_input_ids, past_key_values=past)
						next_token_logits = outputs[0][:, -1, :]
						past = outputs[1]

						# Intead of simple greedy decoding we will use nucleus sampling
						# next_tokens = torch.argmax(next_token_logits, dim=-1)
						next_tokens = list()
						for i in range(next_token_logits.size(0)):

							current_sent_next_token_logits = next_token_logits[i]
							# Check if this is the current thread's first token
							"""
							if input_lengths[i] == step:
								# Add high penalty for eos_token_id so that first token is never eos token
								current_sent_next_token_logits[eos_token_id] = -1e9
							"""
							top_p_next_token_logits = top_k_top_p_filtering(current_sent_next_token_logits, top_p=0.9)
							probabilities = F.softmax(top_p_next_token_logits, dim=-1)
							try:
								next_token = torch.multinomial(probabilities, 1)
							except RuntimeError as e:
								logging.error(e)
								pdb.set_trace()
							"""
							if input_lengths[i] == step:
								if next_token.item() == eos_token_id:
									while next_token.item() == eos_token_id:
										# keep resampling
										logging.error(f"Can't end with empty string for candidate {i}. EOS token prob = {probabilities[eos_token_id]}. Resampling.")
										next_token = torch.multinomial(probabilities, 1)
							"""
							next_tokens.append(next_token)
						next_tokens = torch.tensor(next_tokens).long().to(device)

						# Compute flags to indicate whether to decode or copy from input_ids
						copy_or_decode_flag = (input_lengths > step).long()
						if step < max_seq_len:
							next_input_tokens = input_ids[:, step]
						else:
							next_input_tokens = pad_token_id

						# this updates which sentences have not seen an <EOS> token so far
						# if one <EOS> token was seen the sentence is finished
						# Only update if decoding
						eos_not_in_sents.mul_(next_tokens.ne(eos_token_id).long() * (1-copy_or_decode_flag) + copy_or_decode_flag)

						# either pick the next token from input_ids or decode
						# if decoding, append a padding token here if <EOS> has been seen or append next token
						tokens_to_add = next_input_tokens * (copy_or_decode_flag) + (1 - copy_or_decode_flag) * (next_tokens * (eos_not_in_sents) + pad_token_id * (1 - eos_not_in_sents))

						# Update next inputs and all generations
						generation_ids = torch.cat([generation_ids, tokens_to_add.unsqueeze(-1)], dim=-1).to(device)
						current_input_ids = tokens_to_add.unsqueeze(-1).to(device)
						step += 1
					
					flag = False
					if eos_not_in_sents.float().sum().item() != 0.0:
						logging.warning(f"Some of the posts in current batch didn't finish properly. eos_not_in_sents = {eos_not_in_sents}")
						flag = True
					full_generations = [tokenizer.decode(output, skip_special_tokens=False).replace("\n", " ") for output in generation_ids]
					# log_list(full_generations)
					full_generations = [[e for e in s.split("<|endoftext|>") if e.strip()] for s in full_generations]
					# log_list(full_generations)
					try:
						generations = [e[-1] if len(e) > 0 else "" for e in full_generations]
						if flag:
							# TEMP: manually checking the unfinished generations
							unfinished_gens = [(i, gen) for i, (gen, eos_flag) in enumerate(zip(generations, eos_not_in_sents.tolist())) if eos_flag]
							
					except IndexError:
						# NOTE: There was an empty string in SRC which was causing this Index error
						logging.error("Some generation has not completed properly")
						log_list(full_generations)
						pdb.set_trace()
					# Update current_batch saved generations with new samples
					for i, generation in enumerate(generations):
						if generation.strip() == "[NO-STANCE]":
							# NOTE: this was used for debugging. The DAPT No-Stance model was predicting empty strings.
							pdb.set_trace()
						current_batch_saved_generations[i][1].append(generation)
				# Save current batch_generation in final list

				if prefix:
					# remove prefix from generations and threads before saving
					clean_current_batch_saved_generations = list()
					for thread, generations in current_batch_saved_generations:
						assert prefix in thread
						clean_thread = thread.replace(prefix, "")
						clean_gens = list()
						for gen in generations:
							assert prefix in gen, pdb.set_trace()
							clean_gen = gen.replace(prefix, "")
							clean_gens.append(clean_gen)
						clean_current_batch_saved_generations.append([clean_thread, clean_gens])
					current_batch_saved_generations = clean_current_batch_saved_generations
				all_input_generations.extend(current_batch_saved_generations)

				# Reset current batch
				current_batch = list()
	return all_input_generations


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
	for i, post_thread in enumerate(current_threads):
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
		pdb.set_trace()
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
	for i, post_thread in enumerate(current_threads):
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

def get_offensive_and_stance_predictions(threads, offensive_model, offensive_tokenizer, stance_model, stance_tokenizer):
	final_post_threads_and_predictions = list()
	with torch.no_grad():
		batch_size = args.batch_size
		n_threads = len(threads)
		pbar = tqdm(range(0, n_threads, batch_size))
		softmax_func = nn.Softmax(dim=1)
		for step, i in enumerate(pbar):
			start_index = i
			end_index = min(i+batch_size, n_threads)
			current_batch_post_threads = threads[start_index:end_index]
			current_batch_post_threads_and_predictions = [[post_thread, {"stance":list(), "offensive":list()}] for post_thread in current_batch_post_threads]

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
				current_batch_post_threads_and_predictions[instance_id][1]["stance"].append((to_u_id, from_u_id, softmax_stance_logits[index]))

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
				current_batch_post_threads_and_predictions[instance_id][1]["offensive"].append((u_id, softmax_off_logits[index]))

			# Save both predictions in final list
			final_post_threads_and_predictions.extend(current_batch_post_threads_and_predictions)
	return final_post_threads_and_predictions

def prepare_threads_for_DGPT_large_PPL(current_threads, tokenizer):
	all_GPT2_model_input_texts = list()
	for i, thread in enumerate(current_threads):
		GPT2_string = thread.replace(" EOS ", tokenizer.eos_token)
		all_GPT2_model_input_texts.append(GPT2_string)

	# Tokenize
	all_GPT2_model_inputs_tokenized = tokenizer.batch_encode_plus(all_GPT2_model_input_texts, padding=True, add_special_tokens=False, return_tensors="pt")
	input_ids, attention_mask = all_GPT2_model_inputs_tokenized['input_ids'], all_GPT2_model_inputs_tokenized['attention_mask']
	
	try:
		assert input_ids.size(1) < 512
	except AssertionError:
		logging.error(f"One of the instance has length longer than 512 tokens: {input_ids.shape}")
		log_list(all_GPT2_model_input_texts)
		logging.error(f"Truncating the input to 512 tokens")
		input_ids = input_ids[:, :512]
	
	return {"input_ids": input_ids, "attention_mask": attention_mask}

def last_response_DGPT_large_PPL(threads_with_generations, model, tokenizer):
	# Compute the PPL of generations using DGPT large
	MAX_SEQ_THRESH = 512
	log_softmax_fct = nn.LogSoftmax(dim=2)
	with torch.no_grad():
		batch_size = 5
		current_batch_threads = list()
		i = 0
		total_batches = 0.0
		total_loss = 0.0
		for thread_and_gen in tqdm(threads_with_generations):
			i += 1
			current_batch_threads.append(thread_and_gen)
			if len(current_batch_threads) == batch_size or i == len(thread_and_gen):
				total_batches += 1
				batch = prepare_threads_for_DGPT_large_PPL(current_batch_threads, tokenizer)
				# Process this batch and get losses
				if batch["input_ids"].size(1) >= MAX_SEQ_THRESH:
					# Skip this batch
					logging.info(f"Skipping this batch with input_ids shape = {batch['input_ids'].shape} as our GPU doesn't allow to train with sequences that long.")
					continue
				input_dict = {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}
				# input_dict["labels"] = batch["input_ids"].to(device)
				# Forward
				output = model(**input_dict)

				logits = output[0]

				shift_logits = logits[..., :-1, :].contiguous()
				log_probs = log_softmax_fct(shift_logits)

				attn_mask = batch["attention_mask"].to(device)
				labels = batch["input_ids"].to(device)

				# Here we want the mask on only the last reply
				eos_token_mask = attn_mask * (labels == tokenizer.eos_token_id)
				eos_token_positions = torch.nonzero(eos_token_mask).tolist()
				instance_eos_ids = [list() for i in range(eos_token_mask.size(0))]
				for instance_id, eos_position in eos_token_positions:
					instance_eos_ids[instance_id].append(eos_position)
				second_last_eos_ids = torch.tensor([ids[-2]+1 for ids in instance_eos_ids])
				max_len = attn_mask.size(1)
				mask_to_second_last = torch.BoolTensor(torch.arange(max_len).expand(len(second_last_eos_ids), max_len) < second_last_eos_ids.unsqueeze(1)).to(device)
				final_mask = attn_mask * (~mask_to_second_last)
				test = labels * final_mask

				shift_mask = final_mask[..., 1:].contiguous()
				shift_labels = labels[..., 1:].contiguous()

				log_probs_flat = log_probs.view(-1, log_probs.size(-1))
				target_flat = shift_labels.view(-1, 1)
				losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
				losses = losses_flat.view(shift_labels.size(0), shift_labels.size(1))
				loss = (losses * shift_mask).sum() / shift_mask.sum()

				total_loss += loss
				current_batch_threads = list()
		avg_loss = total_loss / total_batches
		perplexity = torch.exp(avg_loss)
		logging.info(f"DGPT Large Perplexity = {perplexity:3f}")
		return perplexity

def main():
	# Load the model from saved directory
	logging.info(f"Loading DialoGPT-medium model from {args.model_dir} ...")
	tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
	model = AutoModelForCausalLM.from_pretrained(args.model_dir)
	model.to(device)
	logging.info(f"Model loaded to device:{device}")
	tokenizer.pad_token = tokenizer.eos_token
	test_generations = get_nucleus_sampling_generations_from_model(args.data_file, model, tokenizer, device, args.preamble, args.prefix)
	model = model.cpu()
	torch.cuda.empty_cache()

	# Get offensive and stance prediction for every generation
	threads_with_generations = list()
	for test_thread, generations in test_generations:
		for gen in generations:
			thread_with_gen = test_thread + gen + " EOS "
			threads_with_generations.append(thread_with_gen)

	# Load DGPT large model
	DGPT_LARGE_MODEL = 'microsoft/DialoGPT-large'
	logging.info(f"Loading {DGPT_LARGE_MODEL} ...")
	dgpt_large_tokenizer = GPT2Tokenizer.from_pretrained(DGPT_LARGE_MODEL)
	dgpt_large_tokenizer.pad_token = dgpt_large_tokenizer.eos_token
	dgpt_large_model = GPT2LMHeadModel.from_pretrained(DGPT_LARGE_MODEL)
	dgpt_large_model.to(device)
	perplexity = last_response_DGPT_large_PPL(threads_with_generations, dgpt_large_model, dgpt_large_tokenizer)
	dgpt_large_model = dgpt_large_model.cpu()
	torch.cuda.empty_cache()

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

	off_stance_predictions = get_offensive_and_stance_predictions(threads_with_generations, offensive_model, offensive_tokenizer, stance_model, stance_tokenizer)

	# Save everything in pickle file
	save_in_pickle((off_stance_predictions, perplexity), args.output_file)

if __name__ == '__main__':
	main()
