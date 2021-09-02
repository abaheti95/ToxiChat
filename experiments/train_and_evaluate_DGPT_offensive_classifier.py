# We will train offensive classifier on top of the DGPT model
# We will provide a dictionary of tasks to be trained upon in the arguments
# The model training and testing code will be implemented using the transformers library with pytorch backend

import sys
sys.path.insert(0,'..')
sys.path.insert(0,'.')
import utils
from utils import RANDOM_SEED, log_list, print_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, save_in_json, load_from_json, format_time, plot_train_loss, log_TP_FP_FN_TN_from_binary_predictions, draw_and_save_precision_recall_curve, save_list_of_tuples_to_tsv
import pdb

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, GPT2Config, AdamW, get_linear_schedule_with_warmup,  AutoModelForCausalLM, AutoTokenizer
from torch import nn
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
from OC_S_utils import Conversation_Data, get_conversation_data_from_OC_S_file, get_save_lists_from_conv_data, OC_S_offensive_Dataset, get_conversation_data_from_SBF_instances, log_TP_FP_FN_TN_from_conv_off_predictions, TARGET_GROUPS, TARGET_GROUPS_TO_ID, log_TP_FP_FN_TN_convs_from_off_predictions

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-td", "--tasks_dict", help="String version of dictionary that contains all tasks and flags", type=str)
parser.add_argument("-s", "--save_dir", help="Path to the directory where we will save model and the tokenizer", type=str, required=True)
parser.add_argument("-o", "--output_dir", help="Path to the output directory where we will save all the model prediction and results", type=str, required=True)
parser.add_argument("-t", "--train", help="Flag that will indicate if the model needs to be trained", action="store_true")
parser.add_argument("-dv", "--dev_log_frequency", help="How many times should we evaluate in each epoch", type=int, default=2)
parser.add_argument("-f", "--flat_OC_S", help="Flag that will indicate if we should train OC_S data as flat", action="store_true")
parser.add_argument("-bs", "--batch_size", help="Train batch size for GPT2 model", type=int, default=32)
parser.add_argument("-d_bs", "--dev_batch_size", help="Dev and Test batch size for GPT2 model", type=int, default=8)
parser.add_argument("-e", "--n_epochs", help="Number of epochs", type=int, default=8)
parser.add_argument("-lr", "--learning_rate", help="Number of epochs", type=float, default=2e-5)
args = parser.parse_args()
args.tasks_dict = ast.literal_eval(args.tasks_dict)


import logging
# Ref: https://stackoverflow.com/a/49202811/4535284
for handler in logging.root.handlers[:]:
	logging.root.removeHandler(handler)
# Also add the stream handler so that it logs on STD out as well
# Ref: https://stackoverflow.com/a/46098711/4535284
make_dir_if_not_exists(args.output_dir)
make_dir_if_not_exists(args.save_dir)
if args.train:
	logfile = os.path.join(args.output_dir, "train_output.log")
else:
	logfile = os.path.join(args.output_dir, "output.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(logfile, mode='w'), logging.StreamHandler()])

# PRETRAINED_GPT2_MODEL = 'GPT2-base-cased'
PRETRAINED_GPT2_MODEL = 'microsoft/DialoGPT-medium'
# Other global constants required for the code
POSSIBLE_BATCH_SIZE = 1
MAX_SEQ_THRESH = 512

if torch.cuda.is_available():
	device = torch.device("cuda")
	logging.info(f"Using GPU{torch.cuda.get_device_name(0)} to train")
else:
	device = torch.device("cpu")
	logging.info(f"Using CPU to train")

def get_convs_from_OC_S_dataset(data_dir):
	#1.0 Read the OC_S train, dev and test data into conversation data
	oc_s_train_file = os.path.join(data_dir, "OC_S_train.csv")
	oc_s_dev_file = os.path.join(data_dir, "OC_S_dev.csv")
	oc_s_test_file = os.path.join(data_dir, "OC_S_test.csv")

	oc_s_train_convs, header = get_conversation_data_from_OC_S_file(oc_s_train_file, args.flat_OC_S)
	oc_s_dev_convs, header = get_conversation_data_from_OC_S_file(oc_s_dev_file, args.flat_OC_S)
	oc_s_test_convs, header = get_conversation_data_from_OC_S_file(oc_s_test_file, args.flat_OC_S)

	return oc_s_train_convs, oc_s_dev_convs, oc_s_test_convs

def get_convs_from_SBF_dataset(data_dir):
	#1.2 Read the SBF train dev and test data into pandas tables
	sbf_train_file = os.path.join(data_dir, "SBFv2.trn.csv")
	sbf_dev_file = os.path.join(data_dir, "SBFv2.dev.csv")
	sbf_test_file = os.path.join(data_dir, "SBFv2.tst.csv")

	train_df = pd.read_csv(sbf_train_file)
	dev_df = pd.read_csv(sbf_dev_file)
	test_df = pd.read_csv(sbf_test_file)
	
	#1.3 Find the number of unique posts and labels in the pandas dataframe
	n_unique_posts, df_shape = count_unique_posts(train_df)
	logging.info(f"Number of unique posts in SBF Train = {n_unique_posts} with total instances = {df_shape}")
	n_unique_posts, df_shape = count_unique_posts(dev_df)
	logging.info(f"Number of unique posts in SBF Dev = {n_unique_posts} with total instances = {df_shape}")
	n_unique_posts, df_shape = count_unique_posts(test_df)
	logging.info(f"Number of unique posts in SBF Test = {n_unique_posts} with total instances = {df_shape}")
	
	#1.4 Binarize labels for each classification task for every post and get instances
	train_instances, train_offend_labels, train_intend_labels, train_lewd_labels, train_group_labels, train_in_group_labels = relabel_with_binarized_votes_and_create_BERT_instances(train_df)
	dev_instances, dev_offend_labels, dev_intend_labels, dev_lewd_labels, dev_group_labels, dev_in_group_labels = relabel_with_binarized_votes_and_create_BERT_instances(dev_df)
	test_instances, test_offend_labels, test_intend_labels, test_lewd_labels, test_group_labels, test_in_group_labels = relabel_with_binarized_votes_and_create_BERT_instances(test_df)

	#1.5 Log data statistics
	logging.info("Train label statistics:")
	logging.info("\t\toffend\tintend\tlewd\tgroup\tin-group")
	logging.info(f"1:\t{train_offend_labels[1]}\t{train_intend_labels[1]}\t{train_lewd_labels[1]}\t{train_group_labels[1]}\t{train_in_group_labels[1]}")
	logging.info(f"0:\t{train_offend_labels[0]}\t{train_intend_labels[0]}\t{train_lewd_labels[0]}\t{train_group_labels[0]}\t{train_in_group_labels[0]}")
	logging.info("Dev label statistics:")
	logging.info("\t\toffend\tintend\tlewd\tgroup\tin-group")
	logging.info(f"1:\t{dev_offend_labels[1]}\t{dev_intend_labels[1]}\t{dev_lewd_labels[1]}\t{dev_group_labels[1]}\t{dev_in_group_labels[1]}")
	logging.info(f"0:\t{dev_offend_labels[0]}\t{dev_intend_labels[0]}\t{dev_lewd_labels[0]}\t{dev_group_labels[0]}\t{dev_in_group_labels[0]}")
	logging.info("Test label statistics:")
	logging.info("\t\toffend\tintend\tlewd\tgroup\tin-group")
	logging.info(f"1:\t{test_offend_labels[1]}\t{test_intend_labels[1]}\t{test_lewd_labels[1]}\t{test_group_labels[1]}\t{test_in_group_labels[1]}")
	logging.info(f"0:\t{test_offend_labels[0]}\t{test_intend_labels[0]}\t{test_lewd_labels[0]}\t{test_group_labels[0]}\t{test_in_group_labels[0]}")

	#1.6 Convert SBF instances into conversation data compatible with OC_S
	sbf_train_convs = get_conversation_data_from_SBF_instances(train_instances)
	sbf_dev_convs = get_conversation_data_from_SBF_instances(dev_instances)
	sbf_test_convs = get_conversation_data_from_SBF_instances(test_instances)

	return sbf_train_convs, sbf_dev_convs, sbf_test_convs

def get_GPT2_string_from_utterances(utterances):
	# We will append EOS token after each utterance
	return ' EOS '.join([u.strip() for u in utterances]) + " EOS "

class OC_S_DGPT_TokenizeCollator():
	def __init__(self, tokenizer):
		self.tokenizer = tokenizer

	def __call__(self, batch):
		all_GPT2_model_input_texts = list()
		all_convs = list()
		all_resp_types = list()
		gold_off_labels = list()
		total_off_labels_count = 0
		per_instance_n_utterances = list()
		for i, data_dict in enumerate(batch):
			GPT2_string = get_GPT2_string_from_utterances(data_dict["utterances"]).replace(" EOS ", self.tokenizer.eos_token)
			off_labels = data_dict["off_labels"]
			all_convs.append(data_dict["conv"])
			all_resp_types.append(data_dict["resp_type"])
			all_GPT2_model_input_texts.append(GPT2_string)
			gold_off_labels.extend(off_labels)
			per_instance_n_utterances.append(len(off_labels))

		# Tokenize
		all_GPT2_model_inputs_tokenized = self.tokenizer.batch_encode_plus(all_GPT2_model_input_texts, padding=True, add_special_tokens=False, return_tensors="pt")
		input_ids, attention_mask = all_GPT2_model_inputs_tokenized['input_ids'], all_GPT2_model_inputs_tokenized['attention_mask']
		try:
			assert input_ids.size(1) < 512
		except AssertionError:
			logging.error(f"One of the instance has length longer than 512 tokens: {input_ids.shape}")
			log_list(all_GPT2_model_input_texts)
			logging.error(f"Truncating the input to 512 tokens")
			input_ids = input_ids[:, :512]
			input_ids[:, 511][input_ids[:, 511] != self.tokenizer.pad_token_id] = self.tokenizer.eos_token_id

		# Extract the word_ids of CLS tokens i.e. the beginning of all the utterances
		eos_token_ids = (input_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)

		assert len(gold_off_labels) == eos_token_ids[0].size(0)
		assert len(per_instance_n_utterances) == len(batch)
		# Convert the pad_token_ids to eos_token_ids as there is no pad token in DGPT model
		input_ids[input_ids==self.tokenizer.pad_token_id] = self.tokenizer.eos_token_id
		
		
		return {"input_ids": input_ids, "eos_token_ids": eos_token_ids, "gold_off_labels": torch.LongTensor(gold_off_labels), "input_str": all_GPT2_model_input_texts, "input_convs": all_convs, "input_resp_types": all_resp_types, "n_utterances": per_instance_n_utterances, "batch_data": batch}


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

def make_predictions_on_offensive_dataset(dataloader, model, tokenizer, device, segment_name, dev_flag = False, threshold=0.5):
	# Create tqdm progressbar
	if not dev_flag:
		logging.info(f"Predicting for offensive label on the {segment_name} segment at threshold = {threshold}")
		pbar = tqdm(dataloader)
	else:
		pbar = dataloader
	# Setting model to eval for predictions
	# NOTE: assuming that model is already in the given device
	model.eval()
	all_convs_str = list()
	all_convs = list()
	all_resp_types = list()
	all_off_predictions = list()
	all_off_prediction_scores = list()
	all_off_labels = list()
	softmax_func = nn.Softmax(dim=1)
	with torch.no_grad():
		for step, batch in enumerate(pbar):
			all_convs_str.extend(batch["input_str"])
			all_convs.extend(batch["input_convs"])
			all_resp_types.extend(batch["input_resp_types"])
			# Create testing instance for model
			input_dict = {"input_ids": batch["input_ids"].to(device), "utterance_eos_ids": batch["eos_token_ids"]}
			off_labels = batch["gold_off_labels"]
			logits = model(**input_dict)[0]

			off_logits = logits

			# Apply softmax on the off_logits			
			softmax_off_logits = softmax_func(off_logits)
			per_instance_n_utterances = batch["n_utterances"]
			# print(f"Softmax_off_logits = {softmax_off_logits.size()}")
			
			_, predicted_off_labels = softmax_off_logits.max(dim=1)
			prediction_scores = softmax_off_logits[:, 1].cpu().tolist()
			predicted_off_labels = [1 if score >=threshold else 0 for score in prediction_scores]
			
			# Split the prediction scores and off_labels based on per_instance_n_utterances
			per_instance_prediction_off_scores = list()
			per_instance_predicted_off_labels = list()
			per_instance_true_off_labels = list()

			prev_n_utterances = 0
			for i, n_utterances in enumerate(per_instance_n_utterances):
				per_instance_prediction_off_scores.append([prediction_scores[prev_n_utterances + j] for j in range(n_utterances)])
				per_instance_predicted_off_labels.append([predicted_off_labels[prev_n_utterances + j] for j in range(n_utterances)])
				per_instance_true_off_labels.append([off_labels[prev_n_utterances + j].cpu().item() for j in range(n_utterances)])
				prev_n_utterances += n_utterances

			# Save all the predictions and off_labels and targets in lists
			all_off_predictions.extend(per_instance_predicted_off_labels)
			all_off_prediction_scores.extend(per_instance_prediction_off_scores)
			all_off_labels.extend(per_instance_true_off_labels)
			
	return all_convs_str, all_convs, all_resp_types, all_off_predictions, all_off_prediction_scores, all_off_labels

def add_offensive_prediction_to_conv(resp_type, prediction, score, conv):
	# Add the offensive predictions and scores in the conv given the resp_type
	assert len(conv.utterance_data) + 1 == len(prediction)
	for i, u_data in enumerate(conv.utterance_data):
		# Add prediction
		u_data.setdefault("off_prediction", list())
		u_data["off_prediction"].append(prediction[i])
		# Add score
		u_data.setdefault("off_prediction_score", list())
		u_data["off_prediction_score"].append(score[i])
	if resp_type == "dgpt":
		assert "off_prediction" not in conv.dgpt_resp_data
		conv.dgpt_resp_data["off_prediction"] = prediction[-1]
		assert "off_prediction_score" not in conv.dgpt_resp_data
		conv.dgpt_resp_data["off_prediction_score"] = score[-1]
	if resp_type == "gpt3":
		assert "off_prediction" not in conv.gpt3_resp_data
		conv.gpt3_resp_data["off_prediction"] = prediction[-1]
		assert "off_prediction_score" not in conv.gpt3_resp_data
		conv.gpt3_resp_data["off_prediction_score"] = score[-1]

def evaluate_OC_S_GPT2_predictions(convs, resp_types, predictions, scores, print_key="Default"):
	# First align the predictions with convs
	try:
		id_to_conv = {(conv.subset, conv.thread_id, conv.sample_type, conv.subreddit, conv.last_off_score):copy.deepcopy(conv) for conv in convs}
	except AttributeError:
		pdb.set_trace()
	# update predictions in id_to_conv
	for conv, resp_type, prediction, score in zip(convs, resp_types, predictions, scores):
		key = (conv.subset, conv.thread_id, conv.sample_type, conv.subreddit, conv.last_off_score)
		add_offensive_prediction_to_conv(resp_type, prediction, score, id_to_conv[key])

	for key, conv in id_to_conv.items():
		for u_data in conv.utterance_data:
			off_predictions = u_data["off_prediction"]
			assert len(off_predictions) == 2
			assert off_predictions[0] == off_predictions[1]
			u_data["off_prediction"] = off_predictions[0]
			off_scores = u_data["off_prediction_score"]
			assert len(off_scores) == 2
			assert off_scores[0] == off_scores[1]
			u_data["off_prediction_score"] = off_scores[0]
	
	# Get off f1 and cm given list of u_ids
	def get_f1_and_cm_for_given_u_ids(id_to_conv, u_ids):
		labels = list()
		predictions = list()
		scores = list()
		for key, conv in id_to_conv.items():
			for u_id in u_ids:
				label = conv.get_off_label(u_id)
				prediction = conv.get_off_prediction(u_id)
				score = conv.get_off_prediction_score(u_id)
				if label is not None and prediction is not None:
					# keep this label and prediction
					labels.append(label)
					predictions.append(prediction)
					scores.append(score)

		off_f1 = metrics.f1_score(labels, predictions)
		off_cm = metrics.confusion_matrix(labels, predictions)
		return off_f1, off_cm, predictions, scores, labels

	u_ids = [1,2,3,"dgpt","gpt3"]
	all_results = get_f1_and_cm_for_given_u_ids(id_to_conv, u_ids)
	u_ids = [1]
	first_results = get_f1_and_cm_for_given_u_ids(id_to_conv, u_ids)
	u_ids = [2,3,"dgpt","gpt3"]
	reply_results = get_f1_and_cm_for_given_u_ids(id_to_conv, u_ids)
	u_ids = ["dgpt"]
	dgpt_results = get_f1_and_cm_for_given_u_ids(id_to_conv, u_ids)
	u_ids = ["gpt3"]
	gpt3_results = get_f1_and_cm_for_given_u_ids(id_to_conv, u_ids)

	# Log all computed statistics
	logging.info(f"Offensive label classification statistics for {print_key}:")
	tn, fp, fn, tp = all_results[1].ravel()
	logging.info(f"F1 = {all_results[0]:.4f}\t\t\t\tTP={tp}\tFP={fp}\tFN={fn}\tTN={tn}")
	tn, fp, fn, tp = first_results[1].ravel()
	logging.info(f"U1 F1 = {first_results[0]:.4f}\t\t\t\tTP={tp}\tFP={fp}\tFN={fn}\tTN={tn}")
	tn, fp, fn, tp = reply_results[1].ravel()
	logging.info(f"all reply F1 = {reply_results[0]:.4f}\t\t\tTP={tp}\tFP={fp}\tFN={fn}\tTN={tn}")
	tn, fp, fn, tp = dgpt_results[1].ravel()
	logging.info(f"DGPT response F1 = {dgpt_results[0]:.4f}\t\tTP={tp}\tFP={fp}\tFN={fn}\tTN={tn}")
	tn, fp, fn, tp = gpt3_results[1].ravel()
	logging.info(f"GPT3 response F1 = {gpt3_results[0]:.4f}\t\tTP={tp}\tFP={fp}\tFN={fn}\tTN={tn}")
	logging.info("")

	return id_to_conv, {"all":all_results, "first":first_results, "reply":reply_results, "dgpt":dgpt_results, "gpt3":gpt3_results}  

def get_first_and_reply_f1_for_OS_C_flat_dataset(dataset, str_convs, predictions, scores, labels):
	first_predictions = list()
	first_labels = list()
	reply_predictions = list()
	reply_labels = list()
	for oc_s_flat_datapoint, str_conv, prediction, score, label in zip(dataset, str_convs, predictions, scores, labels):
		assert oc_s_flat_datapoint["utterances"][0] in str_conv
		if oc_s_flat_datapoint['id'][0] == 1:
			# Append the label and prediction in first_labels and first_predictions
			first_predictions.append(prediction[0])
			first_labels.append(label[0])
		else:
			# Append the label and prediction in reply_labels and reply_predictions
			reply_predictions.append(prediction[0])
			reply_labels.append(label[0])

	# Calculate F1 scores and confusion matrix for first and reply predictions
	first_off_f1 = metrics.f1_score(first_labels, first_predictions)
	first_off_cm = metrics.confusion_matrix(first_labels, first_predictions)
	reply_off_f1 = metrics.f1_score(reply_labels, reply_predictions)
	reply_off_cm = metrics.confusion_matrix(reply_labels, reply_predictions)
	return first_off_f1, first_off_cm, reply_off_f1, reply_off_cm

def get_classification_metrics_from_scores_and_labels(scores, labels, THRESHOLD=0.5):
	# Expects list of scores (all between 0.0 and 1.0) and list of binary labels (0 or 1)
	predictions = [1 if e >= THRESHOLD else 0 for e in scores]
	p, r, f1, support = metrics.precision_recall_fscore_support(labels, predictions, average="binary")
	cm = metrics.confusion_matrix(labels, predictions)
	tn, fp, fn, tp = cm.ravel()
	return p.item(), r.item(), f1.item(), tn.item(), fp.item(), fn.item(), tp.item(), cm.tolist()

def main():
	#1.0 Read and prepare tasks datasets and dataloaders for provided tasks
	task_convs = dict()
	merged_train_convs = list()
	for task, data_dir in args.tasks_dict.items():
		logging.info(f"############## Loading {task} data from {data_dir} ...")
		if task == "OC_S":
			# Preprocess OC_S data
			task_convs[task] = get_convs_from_OC_S_dataset(data_dir)
		elif task == "SBF":
			# Preprocess SBF data
			task_convs[task] = get_convs_from_SBF_dataset(data_dir)
		# elif task == "CONAN":
		# 	# Preprocess CONAN data
		# 	task_convs[task] = get_convs_from_CONAN_dataset(data_dir)
		else:
			logging.error(f"Unrecognized taskname = {task}. Skipping this task!")
			continue
		train_convs, dev_convs, test_convs = task_convs[task]
		#1.1 Log the train, dev and test statistics
		logging.info(f"{task} Train conversations = {len(train_convs)}")
		logging.info(f"{task} Dev conversations = {len(dev_convs)}")
		logging.info(f"{task} Test conversations = {len(test_convs)}")

		# Add the train_convs to merged_train_convs
		if "SBF" in args.tasks_dict and task == "OC_S":
			# SPECIAL NOTE: When SBF data given duplicate OC_S data more than once
			train_convs = train_convs * 3
		merged_train_convs.extend(train_convs)

	#2.0 Create Dataset and DataLoader from lists of Conversation_Data

	#2.1 Initialize the collator with GPT2 tokenizer
	tokenizer = GPT2Tokenizer.from_pretrained(PRETRAINED_GPT2_MODEL)
	tokenizer.add_special_tokens({'pad_token': '[PAD]'})
	# tokenizer.pad_token = '[PAD]'
	# tokenizer.eos_token = '<|endoftext|>'
	# tokenizer.pad_token_id = 50257
	# tokenizer.eos_token_id = 50256
	tokenize_collator = OC_S_DGPT_TokenizeCollator(tokenizer)

	#2.2 Create merged train and keep the dev and test separate
	random.shuffle(merged_train_convs)
	combined_train_dataset = OC_S_offensive_Dataset(merged_train_convs)
	logging.info(f"Combined Train dataset size = {len(combined_train_dataset)}")
	train_dataloader = DataLoader(combined_train_dataset, batch_size=POSSIBLE_BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=tokenize_collator)

	task_datasets_and_loaders = dict()
	logging.info(f"Creating datasets and dataloaders for the given tasks {task_convs.keys()} ...")
	for task, (train_convs, dev_convs, test_convs) in task_convs.items():
		#2.2.2 Create datasets for dev and test convs
		dev_dataset = OC_S_offensive_Dataset(dev_convs)
		test_dataset = OC_S_offensive_Dataset(test_convs)
		
		#2.2.3 Log the Dataset Statistics
		logging.info(f"{task} Dev dataset size = {len(dev_dataset)}")
		logging.info(f"{task} Test dataset size = {len(test_dataset)}")

		#2.2.4 Create dataloaders from datasets
		dev_dataloader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=False, num_workers=0, collate_fn=tokenize_collator)
		test_dataloader = DataLoader(test_dataset, batch_size=args.dev_batch_size, shuffle=False, num_workers=0, collate_fn=tokenize_collator)

		#2.2.5 Save datasets and dataloaders in dictionary
		task_datasets_and_loaders[task] = (dev_dataset, test_dataset, dev_dataloader, test_dataloader)

	#2.3 Load the model and tokenizer
	if args.train:
		# Create new model from scratch
		config = GPT2Config.from_pretrained(PRETRAINED_GPT2_MODEL)
		model = GPT2ForOC_S_offensive.from_pretrained(PRETRAINED_GPT2_MODEL, config=config)
	else:
		# Load from a previously trained model
		logging.info(f"Loading pretrained model and tokenizer from {args.save_dir}...")
		model = GPT2ForOC_S_offensive.from_pretrained(args.save_dir)
		tokenizer = GPT2Tokenizer.from_pretrained(args.save_dir)
	model.to(device)

	if args.train:
		# Trying to find out the callable methods from the model object
		# Ref: https://stackoverflow.com/a/34452/4535284
		# object_methods = [method_name for method_name in dir(model) if callable(getattr(model, method_name))]
		# print(object_methods)
		# exit()

		# Start training
		epochs = args.n_epochs
		total_steps = len(train_dataloader) * epochs
		
		# Create optimizer
		optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
		scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
		logging.info("Created model optimizer")

		# Create the learning rate scheduler.
		# NOTE: num_warmup_steps = 0 is the Default value in run_glue.py
		# We'll store a number of quantities such as training and validation loss, 
		# validation accuracy, and timings.
		training_stats = []

		logging.info(f"Initiating training loop for {args.n_epochs} epochs...")
		# Measure the total training time for the whole run.
		total_start_time = time.time()

		# Find the accumulation steps
		accumulation_steps = args.batch_size/POSSIBLE_BATCH_SIZE

		# Loss trajectory for epochs
		epoch_train_loss = list()
		best_off_f1 = 0.0
		# Dev validation trajectory
		for epoch in range(epochs):
			pbar = tqdm(train_dataloader)
			logging.info(f"Initiating Epoch {epoch+1}:")
			# Reset the total loss for each epoch.
			total_train_loss = 0
			train_loss_trajectory = list()

			# Reset timer for each epoch
			start_time = time.time()
			model.train()
			model.zero_grad()

			dev_log_frequency = args.dev_log_frequency
			n_steps = len(train_dataloader)
			dev_steps = int(n_steps / dev_log_frequency)
			for step, batch in enumerate(pbar):
				if batch["input_ids"].size(1) >= MAX_SEQ_THRESH:
					# Skip this batch
					logging.info(f"Skipping this batch with input_ids shape = {batch['input_ids'].shape} as our GPU doesn't allow to train with sequences that long.")
					continue
				# Tokenize the inputs in the batch and create input_ids and attention_mask for the model
				# Ref: https://github.com/huggingface/transformers/issues/3021

				input_dict = {"input_ids": batch["input_ids"].to(device), "utterance_eos_ids": batch["eos_token_ids"], "off_labels": batch["gold_off_labels"].to(device)}
				# Forward
				loss, logits = model(**input_dict)
									
				# loss = loss / accumulation_steps
				# Accumulate loss
				total_train_loss += loss.item()

				# Backward: compute gradients
				loss.backward()
				
				if (step + 1) % accumulation_steps == 0:
					
					# Calculate elapsed time in minutes and print loss on the tqdm bar
					elapsed = format_time(time.time() - start_time)
					avg_train_loss = total_train_loss/(step+1)
					# keep track of changing avg_train_loss
					train_loss_trajectory.append(avg_train_loss)
					pbar.set_description(f"Epoch:{epoch+1}|Batch:{step}/{len(train_dataloader)}|Time:{elapsed}|Avg. Loss:{avg_train_loss:.4f}|Loss:{loss.item():.4f}")

					# Clip the norm of the gradients to 1.0.
					# This is to help prevent the "exploding gradients" problem.
					torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

					# Update parameters
					optimizer.step()

					# Clean the model's previous gradients
					model.zero_grad()                           # Reset gradients tensors

					# Update the learning rate.
					scheduler.step()
					pbar.update()
				if (step + 1) % dev_steps == 0:
					# Perform validation on all given datasets with the model and log the performance
					logging.info(f"############## Running Validation on {task_datasets_and_loaders.keys()} ...")
					# Put the model in evaluation mode--the dropout layers behave differently
					# during evaluation.
					model.eval()

					for task, (dev_dataset, test_dataset, dev_dataloader, test_dataloader) in task_datasets_and_loaders.items():
						# Evaluate on OC_S
						dev_str_convs, dev_convs, dev_resp_types, dev_predictions, dev_prediction_scores, dev_labels = make_predictions_on_offensive_dataset(dev_dataloader, model, tokenizer, device, "dev", True)
						if task == "OC_S":
							# Evaluate and calculate F1s
							dev_ids_to_conv_predictions, every_f1_and_cm = evaluate_OC_S_GPT2_predictions(dev_convs, dev_resp_types, dev_predictions, dev_prediction_scores, print_key=f"Final {task} Dev")
							oc_s_off_f1 = every_f1_and_cm["all"][0]
						else:
							# Evaluate and calculate F1s and CM
							predictions = [prediction[0] for prediction in dev_predictions]
							labels = [label[0] for label in dev_labels]
							off_f1 = metrics.f1_score(labels, predictions)
							off_cm = metrics.confusion_matrix(labels, predictions)
							tn, fp, fn, tp = off_cm.ravel()
							logging.info(f"{task} F1 = {off_f1:.4f}\t\tTP={tp}\tFP={fp}\tFN={fn}\tTN={tn}")

					if best_off_f1 < oc_s_off_f1:
						# Keep the copy of current model
						logging.info(f"New best dev Off F1 = {oc_s_off_f1} achieved at epoch {epoch+1}")
						best_model = copy.deepcopy(model)
						best_off_f1 = oc_s_off_f1
						best_off_epoch = epoch+1
					# Put the model back in train setting
					model.train()

			# Calculate the average loss over all of the batches.
			avg_train_loss = total_train_loss / len(train_dataloader)
			
			training_time = format_time(time.time() - start_time)

			# Record all statistics from this epoch.
			training_stats.append({
					'epoch': epoch + 1,
					'Training Loss': avg_train_loss,
					'Training Time': training_time})

			# Save the loss trajectory
			epoch_train_loss.append(train_loss_trajectory)
		logging.info(f"Training complete with total Train time:{format_time(time.time()- total_start_time)}")
		log_list(training_stats)
		
		# Log the best model stats and save it
		logging.info(f"Best Dev Off F1 = {best_off_f1} at epoch {best_off_epoch}.")
		model = best_model
		# Save the model and the Tokenizer here:
		logging.info(f"Saving the model and tokenizer in {args.save_dir}")
		model.save_pretrained(args.save_dir)
		tokenizer.save_pretrained(args.save_dir)

		# Plot the train loss trajectory in a plot
		train_loss_trajectory_plot_file = os.path.join(args.output_dir, "train_loss_trajectory.png")
		logging.info(f"Saving the Train loss trajectory at {train_loss_trajectory_plot_file}")
		plot_train_loss(epoch_train_loss, train_loss_trajectory_plot_file)

		# TODO: Plot the validation performance
		# Save dev_validation_statistics
	else:
		logging.info("No training needed. Directly going to evaluation!")

	# Put the model in evaluation mode. The dropout layers behave differently during evaluation.
	model.eval()
	# Dev set evaluation

	for task, (dev_dataset, test_dataset, dev_dataloader, test_dataloader) in task_datasets_and_loaders.items():
		#TEMP: for debugging. Remove Later
		if task == "SBF":
			break

		logging.info(f"Final evaluation on {task} Dev Set")
		# Evaluate on OC_S
		dev_str_convs, dev_convs, dev_resp_types, dev_predictions, dev_prediction_scores, dev_labels = make_predictions_on_offensive_dataset(dev_dataloader, model, tokenizer, device, "dev", True)

		if task == "OC_S":
			# Evaluate and calculate F1s
			dev_ids_to_conv_predictions, every_f1_and_cm = evaluate_OC_S_GPT2_predictions(dev_convs, dev_resp_types, dev_predictions, dev_prediction_scores, print_key=f"Final {task} Dev")

			# Draw precision recall curve for Offend classification task
			logging.info("Drawing and saving precision recall curve for Offend classification task on OC_S dev set")
			off_pr_curve_save_file = os.path.join(args.output_dir, "offend_dev_pr_cruve.png")
			all_off_scores = every_f1_and_cm["all"][3]
			all_off_labels = every_f1_and_cm["all"][4]
			# PR curve for offensive label
			p, r, t = draw_and_save_precision_recall_curve(all_off_scores, all_off_labels, f"{task} Dev Offend PR curve for DGPT Offensive classifier", "DGPT Off", off_pr_curve_save_file)
			off_pr_values_save_file = os.path.join(args.output_dir, "off_pr_curve_values.tsv")
			save_list_of_tuples_to_tsv(zip(p,r,t), off_pr_values_save_file, header=['precision', 'recall', 'threshold'], delimiter='\t')

			# PR curve for safe label
			all_safe_scores = [1.0-e for e in all_off_scores]
			safe_pr_curve_save_file = os.path.join(args.output_dir, "safe_dev_pr_cruve.png")
			p, r, t = draw_and_save_precision_recall_curve(all_safe_scores, all_off_labels, f"{task} Dev Safe PR curve for DGPT Offensive classifier", "DGPT Safe", safe_pr_curve_save_file, pos_label = 0)
			safe_pr_values_save_file = os.path.join(args.output_dir, "safe_pr_curve_values.tsv")
			save_list_of_tuples_to_tsv(zip(p,r,t), safe_pr_values_save_file, header=['precision', 'recall', 'threshold'], delimiter='\t')

			results = dict()
			# Find the best threshold for Off classification task and save those thresholds in json file
			thresholds = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
			
			results["Off_task"] = dict()
			# Find best threshold for each task
			logging.info(f"Finding the best threshold for Off_task classification")
			# We will store all the metrics associated with a threshold in a list of tuples.
			# Sorting the list based on specific columns will give us best thresholds for a particular metric
			threshold_metrics = list()
			for t in thresholds:
				# compute metrics for each threshold
				task_p, task_r, task_f1, task_tn, task_fp, task_fn, task_tp, task_cm = get_classification_metrics_from_scores_and_labels(all_off_scores, all_off_labels, t)
				logging.info(f"Threshold:{t}||\tP:{task_p:.3f}\tR:{task_r:.3f}\tF1:{task_f1:.3f}\tTP:{task_tp}\tFP:{task_fp}\tFN:{task_fn}\tTN:{task_tn}")
				threshold_metrics.append((t, task_p, task_r, task_f1, task_tn, task_fp, task_fn, task_tp, task_cm))
			results["Off_task"]["dev_threshold_metrics"] = threshold_metrics
			# Sort based on f1 column i.e. 3
			best_threshold_and_metrics = sorted(threshold_metrics, key=lambda e: e[3], reverse=True)[0]
			logging.info(f"Best threshold and metrics for Off_task = {best_threshold_and_metrics}")
			results["Off_task"]["best_dev_threshold"] = best_threshold_and_metrics[0]
			results["Off_task"]["best_dev_threshold_metrics"] = best_threshold_and_metrics[1:]

			# Log the results
			logging.info(f"Offend F1: {best_threshold_and_metrics[3]}\n CM: {best_threshold_and_metrics[-1]}")

			# Log the examples of TP FP FN and TN
			logging.info(f"OC_S Dev Offend label Conv samples:")
			log_TP_FP_FN_TN_convs_from_off_predictions(dev_ids_to_conv_predictions)
		elif task == "SBF":
			# Evaluate and calculate F1s and CM
			predictions = [prediction[0] for prediction in dev_predictions]
			labels = [label[0] for label in dev_labels]
			off_f1 = metrics.f1_score(labels, predictions)
			off_cm = metrics.confusion_matrix(labels, predictions)
			tn, fp, fn, tp = off_cm.ravel()
			logging.info(f"{task} F1 = {off_f1:.4f}\t\tTP={tp}\tFP={fp}\tFN={fn}\tTN={tn}")

	# Test set evaluation
	logging.info("Final evaluation on OC_S Test Set")
	threshold = results["Off_task"]["best_dev_threshold"]
	oc_s_dev_dataset, oc_s_test_dataset, oc_s_dev_dataloader, oc_s_test_dataloader = task_datasets_and_loaders["OC_S"]

	test_str_convs, test_convs, test_resp_types, test_predictions, test_prediction_scores, test_labels = make_predictions_on_offensive_dataset(oc_s_test_dataloader, model, tokenizer, device, "test")

	# Evaluate and calculate F1s
	test_ids_to_conv_predictions, every_f1_and_cm = evaluate_OC_S_GPT2_predictions(test_convs, test_resp_types, test_predictions, test_prediction_scores, print_key=f"Final {task} Test")
	
	# Draw precision recall curve for Offend classification task
	logging.info("Drawing and saving precision recall curve for Offend classification task on OC_S test set")
	off_pr_curve_save_file = os.path.join(args.output_dir, "offend_test_pr_cruve.png")
	all_off_scores = every_f1_and_cm["all"][3]
	all_off_labels = every_f1_and_cm["all"][4]
	# PR curve for offensive label
	p, r, t = draw_and_save_precision_recall_curve(all_off_scores, all_off_labels, f"{task} test Offend PR curve for DGPT Offensive classifier", "DGPT Off", off_pr_curve_save_file)
	off_pr_values_save_file = os.path.join(args.output_dir, "test_off_pr_curve_values.tsv")
	save_list_of_tuples_to_tsv(zip(p,r,t), off_pr_values_save_file, header=['precision', 'recall', 'threshold'], delimiter='\t')

	# PR curve for safe label
	all_safe_scores = [1.0-e for e in all_off_scores]
	safe_pr_curve_save_file = os.path.join(args.output_dir, "safe_test_pr_cruve.png")
	p, r, t = draw_and_save_precision_recall_curve(all_safe_scores, all_off_labels, f"{task} test Safe PR curve for DGPT Offensive classifier", "DGPT Safe", safe_pr_curve_save_file, pos_label = 0)
	safe_pr_values_save_file = os.path.join(args.output_dir, "test_safe_pr_curve_values.tsv")
	save_list_of_tuples_to_tsv(zip(p,r,t), safe_pr_values_save_file, header=['precision', 'recall', 'threshold'], delimiter='\t')

	# Log the examples of TP FP FN and TN
	logging.info(f"OC_S Test Offend label Conv samples:")
	log_TP_FP_FN_TN_convs_from_off_predictions(test_ids_to_conv_predictions)

	# Evaluate and calculate F1s
	off_f1, off_cm, off_predictions, off_labels, remaining_out = evaluate_OC_S_GPT2_predictions(test_predictions, test_labels, no_reply=args.flat_OC_S)
	flat_off_prediction_scores = [score for instance_scores in test_prediction_scores for score in instance_scores]
	# Log the results
	task_p, task_r, task_f1, task_tn, task_fp, task_fn, task_tp, task_cm = get_classification_metrics_from_scores_and_labels(flat_off_prediction_scores, off_labels, threshold)
	logging.info(f"Off_task - Threshold: {threshold}\tP:{task_p:.3f}\tR:{task_r:.3f}\tF1:{task_f1:.3f}\tTP:{task_tp}\tFP:{task_fp}\tFN:{task_fn}\tTN:{task_tn}")

	# Also log the results with 0.5 threshold
	# threshold = 0.5
	# task_p, task_r, task_f1, task_tn, task_fp, task_fn, task_tp, task_cm = get_classification_metrics_from_scores_and_labels(flat_off_prediction_scores, off_labels, threshold)
	# logging.info(f"Off_task - Threshold: {threshold}\tP:{task_p:.3f}\tR:{task_r:.3f}\tF1:{task_f1:.3f}\tTP:{task_tp}\tFP:{task_fp}\tFN:{task_fn}\tTN:{task_tn}")

	# log separate first and reply
	if not args.flat_OC_S:
		first_off_f1, first_off_cm, reply_off_f1, reply_off_cm = remaining_out
	else:
		# if the OC_S dataset is flat then we have to find the first_off_f1 and reply_off_f1 in a different way
		# Use the oc_s_test_dataset and test_predictions to find out F1
		first_off_f1, first_off_cm, reply_off_f1, reply_off_cm = get_first_and_reply_f1_for_OS_C_flat_dataset(oc_s_test_dataset, test_str_convs, test_predictions, test_prediction_scores, test_labels)
	logging.info(f"OC_S Test first Offend F1: {first_off_f1}\n CM: {first_off_cm}")
	logging.info(f"OC_S Test reply Offend F1: {reply_off_f1}\n CM: {reply_off_cm}")
	if args.include_stance:
		# Get the stance predictions
		test_stance_pairs, test_stance_predictions, test_stance_prediction_scores, test_stance_labels = make_stance_predictions_on_dataset(oc_s_test_dataloader, model, tokenizer, device, "test", True)
		stance_p, stance_r, stance_f1, stance_support, stance_cm = evaluate_OC_S_GPT2_stance_predictions(test_stance_predictions, test_stance_labels)
		# Print the p,r f1 and support for each label
		logging.info(f"Stance prediction stats on OC_S Test set ...")
		logging.info(f"Label:\tsupport\tprecision\trecall\tf1")
		for i in range(3):
			logging.info(f"{i}:\t{stance_support[i]}\t{stance_p[i]:.3f}\t{stance_r[i]:.3f}\t{stance_f1[i]:.3f}")
		logging.info(f"OC_S Test stance CM: \n{stance_cm}")
		

	# Log the examples of TP FP FN and TN
	logging.info(f"OC_S Test Offend label Conv samples:")
	log_TP_FP_FN_TN_from_conv_off_predictions(test_predictions, test_prediction_scores, test_labels, test_str_convs)

	# Save results
	results_file = os.path.join(args.output_dir, "results.json")
	logging.info(f"Saving results in {results_file}")
	save_in_json(results, results_file)

if __name__ == '__main__':
	main()