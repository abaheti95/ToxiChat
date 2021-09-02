# We will train pairwise stance classifier on top of the BERT Large model
# We will provide a dictionary of tasks to be trained upon in the arguments (but probably we will only send OC_S stance data here)
# The model training and testing code will be implemented using the transformers library with pytorch backend

import sys
sys.path.insert(0,'..')
sys.path.insert(0,'.')
import utils
from utils import RANDOM_SEED, log_list, print_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, save_in_json, load_from_json, format_time, plot_train_loss, log_TP_FP_FN_TN_from_binary_predictions, draw_and_save_precision_recall_curve, save_list_of_tuples_to_tsv
import pdb

from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup,  AutoModelForCausalLM, AutoTokenizer
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
# from SBF_utils import BertForSBF, SBF_Bert_Dataset, count_unique_posts, convert_string_label_to_binary, relabel_with_binarized_votes_and_create_Bert_instances, get_labels_dict_from_list
# Import functions and classes specific to OC_S training and evaluation.
from OC_S_utils import Conversation_Data, get_conversation_data_from_OC_S_file, get_save_lists_from_conv_data, OC_S_pairwise_stance_Dataset, get_conversation_data_from_SBF_instances, log_TP_FP_FN_TN_from_conv_off_predictions, TARGET_GROUPS, TARGET_GROUPS_TO_ID, log_TP_FP_FN_TN_convs_from_stance_predictions, log_top_conv_stance_predictions

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-td", "--tasks_dict", help="String version of dictionary that contains all tasks and flags", type=str)
parser.add_argument("-s", "--save_dir", help="Path to the directory where we will save model and the tokenizer", type=str, required=True)
parser.add_argument("-o", "--output_dir", help="Path to the output directory where we will save all the model prediction and results", type=str, required=True)
parser.add_argument("-t", "--train", help="Flag that will indicate if the model needs to be trained", action="store_true")
parser.add_argument("-dv", "--dev_log_frequency", help="How many times should we evaluate in each epoch", type=int, default=2)
parser.add_argument("-f", "--flat_OC_S", help="Flag that will indicate if we should train OC_S data as flat", action="store_true")
parser.add_argument("-foc", "--focal_loss", help="Flag that will indicate if we should train with Class balanced Focal Loss", action="store_true")
parser.add_argument("-ao", "--adjacent_only", help="Flag that will indicate if we should train only on the adjacent stance pairs", action="store_true")
parser.add_argument("-bs", "--batch_size", help="Train batch size for Bert model", type=int, default=32)
parser.add_argument("-d_bs", "--dev_batch_size", help="Dev and Test batch size for Bert model", type=int, default=8)
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

# PRETRAINED_BERT_MODEL = 'Bert-base-cased'
PRETRAINED_BERT_MODEL = 'bert-large-cased'
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


class OC_S_Bert_pairwise_stance_TokenizeCollator():
	def __init__(self, tokenizer):
		self.tokenizer = tokenizer

	def __call__(self, batch):
		all_bert_model_input_texts = list()
		all_convs = list()
		gold_stance_labels = list()
		gold_stance_u_id_pairs = list()
		per_instance_n_utterances = list()
		for i, data_dict in enumerate(batch):
			# Merge the from_u and to_u with [SEP] token
			bert_string = f"[CLS] {data_dict['to_u']} [SEP] {data_dict['from_u']}"
			all_convs.append(data_dict["conv"])
			gold_stance_u_id_pairs.append((data_dict["to_id"], data_dict["from_id"]))
			all_bert_model_input_texts.append(bert_string)
			per_instance_n_utterances.append(len(data_dict["conv"].utterance_data) + 1)
			gold_stance_labels.append(data_dict["stance_label"])

		# Tokenize
		all_Bert_model_inputs_tokenized = self.tokenizer.batch_encode_plus(all_bert_model_input_texts, padding=True, add_special_tokens=False, return_tensors="pt")
		input_ids, attention_mask = all_Bert_model_inputs_tokenized['input_ids'], all_Bert_model_inputs_tokenized['attention_mask']

		assert len(per_instance_n_utterances) == len(batch) == len(gold_stance_labels), f"{len(per_instance_n_utterances)} == {len(batch)} == {len(gold_stance_labels)}"
		try:
			assert input_ids.size(1) < 512
		except AssertionError:
			logging.error(f"One of the instance has length longer than 512 tokens: {input_ids.shape}")
			log_list(all_bert_model_input_texts)
			logging.error(f"Truncating the input to 512 tokens")
			input_ids = input_ids[:, :512]
			attention_mask = attention_mask[:, :512]
		
		# Convert token_ids into tuples for future processing
		return {"input_ids": input_ids, "gold_stance_labels": torch.LongTensor(gold_stance_labels), "gold_stance_u_id_pairs": gold_stance_u_id_pairs, "input_str": all_bert_model_input_texts, "input_convs": all_convs, "n_utterances": per_instance_n_utterances, "batch_data": batch}


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


class BertForOC_S_stance(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_stance_labels = 3
		logging.info(f"Number of stance labels for BertForOC_S_stance classifier = {self.num_stance_labels}")
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.stance_classifier = nn.Linear(config.hidden_size, self.num_stance_labels)
		if config.focal_loss:
			# Instantiate using Focal loss
			weight = reweight(config.cls_num_list)
			self.stance_loss_fct = FocalLoss(weight=weight, gamma=1.0)
			logging.info(f"Using Class balanced focal loss with beta = 0.9999 and gamma = 1.0")
		else:
			# self.stance_loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 10.0]))
			# All Stance Pairs results
			# New best dev avg Stance agree and disagree F1 = 0.1548235294117647 achieved at epoch 7
			# OC_S agree_stance_f1 = 0.192
			# OC_S disagree_stance_f1 = 0.1176470588235294

			# self.stance_loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 30.0]))
			# Adjacent Stance Pairs result
			# New best dev avg Stance agree and disagree F1 = 0.22987341772151898 achieved at epoch 7
			# OC_S agree_stance_f1 = 0.37974683544303794
			# OC_S disagree_stance_f1 = 0.08

			# NOTE: Adding weights doesn't change anything
			self.stance_loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 100.0, 100.0]))
			logging.info(f"Using Cross entropy loss with weights [1.0, 100.0, 100.0]")
			# Adjacent Stance Pairs result
			# New best dev avg Stance agree and disagree F1 = 0.22987341772151898 achieved at epoch 7
			# OC_S agree_stance_f1 = 0.37974683544303794
			# OC_S disagree_stance_f1 = 0.08

	def forward(
		self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
		stance_labels=None,
	):
		r"""
		labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
			Labels for computing the sequence classification/regression loss.
			Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
			If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
			If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

	Returns:
		:obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
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
		outputs = self.bert(
			input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
		)
		# Type of outputs = BaseModelOutputWithPastAndCrossAttentions
		# ref: https://huggingface.co/transformers/_modules/transformers/modeling_outputs.html#BaseModelOutputWithPastAndCrossAttentions
		# ref2: https://huggingface.co/transformers/_modules/transformers/modeling_bert.html in class BertForSequenceClassification(BertPreTrainedModel):
		cls_token_representation = outputs[1]
		# Apply dropout
		stance_classifier_input = self.dropout(cls_token_representation)
		# Compute stance logits from concatenated eos representations
		stance_logits = self.stance_classifier(stance_classifier_input)


		outputs = (stance_logits,) + outputs[2:]
		# If stance_labels given, compute loss from stance_logits
		
		loss = 0.0
		if stance_labels is not None:
			loss = self.stance_loss_fct(stance_logits.view(-1, self.num_stance_labels), stance_labels.view(-1))
			# print(f"input ids = {input_ids}, DGPT outputs shape = {bert_last_layer_output.size()} vs nan count = {torch.isnan(bert_last_layer_output).sum()}")
			# print(f"Off logits = {stance_logits} vs Off labels = {off_labels}")
			# if target_labels is not None:
			# 	# Some of the target_labels can still be None. We have to ignore loss for these target labels
			# 	for i, target_label in enumerate(target_labels):
			# 		if target_label is not None:
			# 			loss += self.target_loss_fct(target_logits[i], target_label.to(device))
			outputs = (loss,) + outputs

		return outputs  # (loss), logits, (hidden_states), (attentions)

def make_predictions_on_pairwise_stance_dataset(dataloader, model, tokenizer, device, segment_name, dev_flag = False, threshold=0.5):
	# Create tqdm progressbar
	if not dev_flag:
		logging.info(f"Predicting for stance label on the {segment_name} segment at threshold = {threshold}")
		pbar = tqdm(dataloader)
	else:
		pbar = dataloader
	# Setting model to eval for predictions
	# NOTE: assuming that model is already in the given device
	model.eval()
	all_convs_str = list()
	all_convs = list()
	all_u_id_pairs = list()
	all_stance_predictions = list()
	all_stance_prediction_scores = list()
	all_stance_labels = list()
	softmax_func = nn.Softmax(dim=1)
	with torch.no_grad():
		for step, batch in enumerate(pbar):
			all_convs_str.extend(batch["input_str"])
			all_convs.extend(batch["input_convs"])
			gold_stance_u_id_pairs = batch["gold_stance_u_id_pairs"]
			all_u_id_pairs.extend(gold_stance_u_id_pairs)
			# Create testing instance for model
			input_dict = {"input_ids": batch["input_ids"].to(device)}
			stance_labels = batch["gold_stance_labels"].tolist()
			logits = model(**input_dict)[0]

			stance_logits = logits

			# Apply softmax on the stance_logits			
			softmax_stance_logits = softmax_func(stance_logits)
			per_instance_n_utterances = batch["n_utterances"]

			_, predicted_stance_labels = softmax_stance_logits.max(dim=1)
			prediction_scores = softmax_stance_logits.cpu().tolist()
			predicted_stance_labels = predicted_stance_labels.cpu().tolist()
			
			assert len(predicted_stance_labels) == len(gold_stance_u_id_pairs)

			# Save all the predictions and stance_labels and targets in lists
			all_stance_predictions.extend(predicted_stance_labels)
			all_stance_prediction_scores.extend(prediction_scores)
			all_stance_labels.extend(stance_labels)
			
	return all_convs_str, all_convs, all_u_id_pairs, all_stance_predictions, all_stance_prediction_scores, all_stance_labels

def add_stance_prediction_to_conv(u_id_pair, prediction, score, conv):
	# Add a single stance prediction and score in the conv given the u_id_pair
	# Add prediction
	u_to, u_from = u_id_pair
	conv.set_stance_prediction_and_score(u_from, u_to, prediction, score)

def evaluate_OC_S_Bert_pairwise_stance_predictions(convs, u_id_pairs, predictions, scores, print_key="Default", adjacent_only=False):
	# First align the predictions with convs
	id_to_conv = {(conv.subset, conv.thread_id, conv.sample_type, conv.subreddit, conv.last_off_score):copy.deepcopy(conv) for conv in convs}
	# update predictions in id_to_conv
	for conv, u_id_pair, prediction, score in zip(convs, u_id_pairs, predictions, scores):
		key = (conv.subset, conv.thread_id, conv.sample_type, conv.subreddit, conv.last_off_score)
		add_stance_prediction_to_conv(u_id_pair, prediction, score, id_to_conv[key])

	# Check if the stance_predictions for u_ids is correct
	for key, conv in id_to_conv.items():
		for u_data in conv.utterance_data:
			u_id = u_data["id"]
			if u_id < 2:
				continue
			for j in range(1, u_id):
				if f"{j}stance_prediction" not in u_data:
					continue
				stance_predictions = u_data[f"{j}stance_prediction"]
				assert len(stance_predictions) == 1
				u_data[f"{j}stance_prediction"] = stance_predictions[0]
				stance_scores = u_data[f"{j}stance_prediction_score"]
				assert len(stance_scores) == 1
				u_data[f"{j}stance_prediction_score"] = stance_scores[0]
		# Check for DGPT and GPT3 replies
		n_utterances = len(conv.utterance_data)
		for u_data in [conv.dgpt_resp_data, conv.gpt3_resp_data]:
			for j in range(1, n_utterances+1):
				if f"{j}stance_prediction" not in u_data:
					continue
				stance_predictions = u_data[f"{j}stance_prediction"]
				assert len(stance_predictions) == 1
				u_data[f"{j}stance_prediction"] = stance_predictions[0]
				stance_scores = u_data[f"{j}stance_prediction_score"]
				assert len(stance_scores) == 1
				u_data[f"{j}stance_prediction_score"] = stance_scores[0]
	
	# Get off f1 and cm given list of u_ids
	def get_f1_and_cm_for_given_from_u_ids(id_to_conv, from_u_ids, adjacent_only=False):
		labels = list()
		predictions = list()
		scores = list()
		for key, conv in id_to_conv.items():
			for from_u_id in from_u_ids:
				prediction, score, label = conv.get_stance_predictions_scores_and_labels_for_u_id(from_u_id, adjacent_only)
				if label is not None and prediction is not None:
					# keep this label and prediction
					labels.extend(label)
					predictions.extend(prediction)
					scores.extend(score)
		stance_cm = metrics.confusion_matrix(labels, predictions)
		stance_p, stance_r, stance_f1, stance_support = metrics.precision_recall_fscore_support(labels, predictions)
		return stance_cm, stance_p, stance_r, stance_f1, stance_support, predictions, scores, labels

	if not adjacent_only:
		# print the adjacent evaluation first then print the full evaluation
		from_u_ids = [2,3,"dgpt","gpt3"]
		adj_results = get_f1_and_cm_for_given_from_u_ids(id_to_conv, from_u_ids, True)
		from_u_ids = ["dgpt"]
		dgpt_results = get_f1_and_cm_for_given_from_u_ids(id_to_conv, from_u_ids, True)
		from_u_ids = ["gpt3"]
		gpt3_results = get_f1_and_cm_for_given_from_u_ids(id_to_conv, from_u_ids, True)

		# Log all computed statistics
		logging.info(f"Adjacent Stance label classification statistics for {print_key}:")
		logging.info(f"Adjacent stance predictions - ")
		stance_cm, stance_p, stance_r, stance_f1, stance_support, predictions, scores, labels = adj_results
		logging.info(f"Label:\tsupport\tprecision\trecall\tf1")
		macro_f1 = 0.0
		for i in range(3):
			logging.info(f"{i}:\t{stance_support[i]}\t{stance_p[i]:.3f}\t{stance_r[i]:.3f}\t{stance_f1[i]:.3f}")
			macro_f1 += stance_f1[i]
		macro_f1 /= 3
		logging.info(f"Macro-average F1: {macro_f1:.3f}")
		logging.info(f"Adjacent stance CM: \n{stance_cm}")
		logging.info(f"Adjacent DGPT stance predictions - ")
		stance_cm, stance_p, stance_r, stance_f1, stance_support, predictions, scores, labels = dgpt_results
		logging.info(f"Label:\tsupport\tprecision\trecall\tf1")
		for i in range(3):
			logging.info(f"{i}:\t{stance_support[i]}\t{stance_p[i]:.3f}\t{stance_r[i]:.3f}\t{stance_f1[i]:.3f}")
		logging.info(f"Adjacent DGPT stance CM: \n{stance_cm}")
		logging.info(f"Adjacent GPT3 stance predictions - ")
		stance_cm, stance_p, stance_r, stance_f1, stance_support, predictions, scores, labels = gpt3_results
		logging.info(f"Label:\tsupport\tprecision\trecall\tf1")
		for i in range(3):
			logging.info(f"{i}:\t{stance_support[i]}\t{stance_p[i]:.3f}\t{stance_r[i]:.3f}\t{stance_f1[i]:.3f}")
		logging.info(f"Adjacent GPT3 stance CM: \n{stance_cm}")
		logging.info("")

	from_u_ids = [2,3,"dgpt","gpt3"]
	all_results = get_f1_and_cm_for_given_from_u_ids(id_to_conv, from_u_ids)
	# Getting the adjacent only F1 on reddit comments.
	from_u_ids = [2,3]
	adj_reddit_results = get_f1_and_cm_for_given_from_u_ids(id_to_conv, from_u_ids, True)
	from_u_ids = ["dgpt"]
	dgpt_results = get_f1_and_cm_for_given_from_u_ids(id_to_conv, from_u_ids)
	from_u_ids = ["gpt3"]
	gpt3_results = get_f1_and_cm_for_given_from_u_ids(id_to_conv, from_u_ids)

	# Log all computed statistics
	logging.info(f"Stance label classification statistics for {print_key}:")
	logging.info(f"All stance predictions - ")
	stance_cm, stance_p, stance_r, stance_f1, stance_support, predictions, scores, labels = all_results
	logging.info(f"Label:\tsupport\tprecision\trecall\tf1")
	macro_f1 = 0.0
	for i in range(3):
		logging.info(f"{i}:\t{stance_support[i]}\t{stance_p[i]:.3f}\t{stance_r[i]:.3f}\t{stance_f1[i]:.3f}")
		macro_f1 += stance_f1[i]
	macro_f1 /= 3
	logging.info(f"Macro-average F1: {macro_f1:.3f}")
	logging.info(f"All stance CM: \n{stance_cm}")
	logging.info(f"DGPT stance predictions - ")
	stance_cm, stance_p, stance_r, stance_f1, stance_support, predictions, scores, labels = dgpt_results
	logging.info(f"Label:\tsupport\tprecision\trecall\tf1")
	for i in range(3):
		logging.info(f"{i}:\t{stance_support[i]}\t{stance_p[i]:.3f}\t{stance_r[i]:.3f}\t{stance_f1[i]:.3f}")
	logging.info(f"DGPT stance CM: \n{stance_cm}")
	logging.info(f"GPT3 stance predictions - ")
	stance_cm, stance_p, stance_r, stance_f1, stance_support, predictions, scores, labels = gpt3_results
	logging.info(f"Label:\tsupport\tprecision\trecall\tf1")
	for i in range(3):
		logging.info(f"{i}:\t{stance_support[i]}\t{stance_p[i]:.3f}\t{stance_r[i]:.3f}\t{stance_f1[i]:.3f}")
	logging.info(f"GPT3 stance CM: \n{stance_cm}")
	logging.info("")
	logging.info(f"Adjacent Reddit stance predictions - ")
	stance_cm, stance_p, stance_r, stance_f1, stance_support, predictions, scores, labels = adj_reddit_results
	logging.info(f"Label:\tsupport\tprecision\trecall\tf1")
	macro_f1 = 0.0
	for i in range(3):
		logging.info(f"{i}:\t{stance_support[i]}\t{stance_p[i]:.3f}\t{stance_r[i]:.3f}\t{stance_f1[i]:.3f}")
		macro_f1 += stance_f1[i]
	macro_f1 /= 3
	logging.info(f"Macro-average F1: {macro_f1:.3f}")
	logging.info(f"Adjacent Reddit stance CM: \n{stance_cm}")

	return id_to_conv, {"adj": adj_reddit_results, "all":all_results, "dgpt":dgpt_results, "gpt3":gpt3_results}
	
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
		else:
			logging.error(f"Unrecognized taskname = {task}. Skipping this task!")
			continue
		train_convs, dev_convs, test_convs = task_convs[task]
		#1.1 Log the train, dev and test statistics
		logging.info(f"{task} Train conversations = {len(train_convs)}")
		logging.info(f"{task} Dev conversations = {len(dev_convs)}")
		logging.info(f"{task} Test conversations = {len(test_convs)}")

		#1.2 Add the train_convs to merged_train_convs
		merged_train_convs.extend(train_convs)

	#2.1 Initialize the collator with Bert tokenizer
	tokenizer = BertTokenizer.from_pretrained(PRETRAINED_BERT_MODEL)
	tokenize_collator = OC_S_Bert_pairwise_stance_TokenizeCollator(tokenizer)

	#2.2 Create merged train and keep the dev and test separate
	random.shuffle(merged_train_convs)
	combined_train_dataset = OC_S_pairwise_stance_Dataset(merged_train_convs, args.adjacent_only)
	logging.info(f"Combined Train dataset size = {len(combined_train_dataset)}")
	train_dataloader = DataLoader(combined_train_dataset, batch_size=POSSIBLE_BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=tokenize_collator)

	#2.2.1 get per class number of instances from train dataloader
	all_train_stance_labels = list()
	for batch in train_dataloader:
		all_train_stance_labels.extend(batch["gold_stance_labels"].tolist())
	stace_label_counts = Counter(all_train_stance_labels)
	sc = stace_label_counts
	logging.info(f"Train stance label counts = {sc}")
	per_class_counts = [sc[0], sc[1], sc[2]]

	task_datasets_and_loaders = dict()
	logging.info(f"Creating datasets and dataloaders for the given tasks {task_convs.keys()} ...")
	for task, (train_convs, dev_convs, test_convs) in task_convs.items():
		#2.2.2 Create datasets for dev and test convs
		dev_dataset = OC_S_pairwise_stance_Dataset(dev_convs, args.adjacent_only)
		test_dataset = OC_S_pairwise_stance_Dataset(test_convs, args.adjacent_only)
		
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
		config = BertConfig.from_pretrained(PRETRAINED_BERT_MODEL)
		config.focal_loss = args.focal_loss
		config.cls_num_list = per_class_counts
		model = BertForOC_S_stance.from_pretrained(PRETRAINED_BERT_MODEL, config=config)
	else:
		# Load from a previously trained model
		logging.info(f"Loading pretrained model and tokenizer from {args.save_dir}...")
		model = BertForOC_S_stance.from_pretrained(args.save_dir)
		tokenizer = BertTokenizer.from_pretrained(args.save_dir)
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
		optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
		scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
		logging.info(f"Created model optimizer with learning rate = {args.learning_rate}")

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
		best_stance_f1 = 0.0
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
				input_dict = {"input_ids": batch["input_ids"].to(device)}
				input_dict["stance_labels"] = batch["gold_stance_labels"].to(device)
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
						dev_str_convs, dev_convs, dev_stance_u_id_pairs, dev_predictions, dev_prediction_scores, dev_labels = make_predictions_on_pairwise_stance_dataset(dev_dataloader, model, tokenizer, device, "dev", True)
						if task == "OC_S":
							# Evaluate and calculate F1s
							dev_ids_to_conv_predictions, every_f1_and_cm = evaluate_OC_S_Bert_pairwise_stance_predictions(dev_convs, dev_stance_u_id_pairs, dev_predictions, dev_prediction_scores, f"Intermediate Training {task} Dev", args.adjacent_only)
							oc_s_no_stance_f1 = every_f1_and_cm["all"][3][0]
							oc_s_agree_stance_f1 = every_f1_and_cm["all"][3][1]
							oc_s_disagree_stance_f1 = every_f1_and_cm["all"][3][2]
							oc_s_avg_stance_f1 = (oc_s_no_stance_f1 + oc_s_agree_stance_f1 + oc_s_disagree_stance_f1)/3
						else:
							# No else here
							logging.error(f"Unknown Stance task {task}. Terminating")
							exit()

					if best_stance_f1 < oc_s_avg_stance_f1:
						# Keep the copy of current model
						logging.info(f"New best dev avg Stance agree and disagree F1 = {oc_s_avg_stance_f1} achieved at epoch {epoch+1}")
						logging.info(f"OC_S agree_stance_f1 = {oc_s_agree_stance_f1}")
						logging.info(f"OC_S disagree_stance_f1 = {oc_s_disagree_stance_f1}")
						best_model = copy.deepcopy(model)
						best_stance_f1 = oc_s_avg_stance_f1
						best_stance_epoch = epoch+1
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
		logging.info(f"Best Dev Stance Macro F1 = {best_stance_f1} at epoch {best_stance_epoch}.")
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
		if task == "OC_S":
			logging.info(f"Final evaluation on {task} Stance Dev Set")
			# Evaluate on OC_S
			dev_str_convs, dev_convs, dev_stance_u_id_pairs, dev_predictions, dev_prediction_scores, dev_labels = make_predictions_on_pairwise_stance_dataset(dev_dataloader, model, tokenizer, device, "dev", True)
			# Evaluate and calculate F1s
			dev_ids_to_conv_predictions, every_f1_and_cm = evaluate_OC_S_Bert_pairwise_stance_predictions(dev_convs, dev_stance_u_id_pairs, dev_predictions, dev_prediction_scores, f"Final {task} Dev", args.adjacent_only)
			# Results are in the format stance_cm, stance_p, stance_r, stance_f1, stance_support, predictions, scores, labels
			oc_s_no_stance_f1 = every_f1_and_cm["adj"][3][0]
			oc_s_agree_stance_f1 = every_f1_and_cm["adj"][3][1]
			oc_s_disagree_stance_f1 = every_f1_and_cm["adj"][3][2]
			oc_s_predictions, oc_s_scores, oc_s_labels = every_f1_and_cm["adj"][5:8]
			oc_s_avg_stance_f1 = (oc_s_no_stance_f1 + oc_s_agree_stance_f1 + oc_s_disagree_stance_f1)/3
			# Log the results and plot PR curves
			no_stance_pr_curve_save_file = os.path.join(args.output_dir, "no_stance_dev_pr_cruve.png")
			pos_stance_pr_curve_save_file = os.path.join(args.output_dir, "positive_stance_dev_pr_cruve.png")
			neg_stance_pr_curve_save_file = os.path.join(args.output_dir, "negative_stance_dev_pr_cruve.png")
			logging.info(f"Plotting the positive and negative stance PR curves and saving them at {pos_stance_pr_curve_save_file} and {neg_stance_pr_curve_save_file} respectively")
			no_stance_scores = [e[0] for e in oc_s_scores]
			pos_stance_scores = [e[1] for e in oc_s_scores]
			neg_stance_scores = [e[2] for e in oc_s_scores]
			no_stance_labels = [e == 0 for e in oc_s_labels]
			pos_stance_labels = [e == 1 for e in oc_s_labels]
			neg_stance_labels = [e == 2 for e in oc_s_labels]
			p, r, t = draw_and_save_precision_recall_curve(no_stance_scores, no_stance_labels, f"{task} Dev No Stance PR curve for BERT Stance classifier (Adjacent)", "BERT No Stance", no_stance_pr_curve_save_file)
			no_stance_pr_values_save_file = os.path.join(args.output_dir, "no_stance_pr_curve_values.tsv")
			save_list_of_tuples_to_tsv(zip(p,r,t), no_stance_pr_values_save_file, header=['precision', 'recall', 'threshold'], delimiter='\t')
			p, r, t = draw_and_save_precision_recall_curve(pos_stance_scores, pos_stance_labels, f"{task} Dev Positive Stance PR curve for BERT Stance classifier (Adjacent)", "BERT Pos Stance", pos_stance_pr_curve_save_file)
			pos_stance_pr_values_save_file = os.path.join(args.output_dir, "positive_stance_pr_curve_values.tsv")
			save_list_of_tuples_to_tsv(zip(p,r,t), pos_stance_pr_values_save_file, header=['precision', 'recall', 'threshold'], delimiter='\t')
			p, r, t = draw_and_save_precision_recall_curve(neg_stance_scores, neg_stance_labels, f"{task} Dev Negative Stance PR curve for BERT Stance classifier (Adjacent)", "BERT Neg Stance", neg_stance_pr_curve_save_file)
			neg_stance_pr_values_save_file = os.path.join(args.output_dir, "negative_stance_pr_curve_values.tsv")
			save_list_of_tuples_to_tsv(zip(p,r,t), neg_stance_pr_values_save_file, header=['precision', 'recall', 'threshold'], delimiter='\t')


			# Log the examples of TP FP FN and TN
			analysis_csv_rows = list()
			logging.info(f"OC_S Dev Stance label Conv samples:")
			analysis_csv_rows.append(["OC_S Dev Stance label Conv samples:", "Utterances", "My comments"])
			logging.info(f"Positive Stance label:")
			analysis_csv_rows.append(["Positive Stance label:"])
			pos_stance_rows = log_TP_FP_FN_TN_convs_from_stance_predictions(dev_ids_to_conv_predictions, 1)
			analysis_csv_rows.extend(pos_stance_rows)
			analysis_csv_rows.append([])
			logging.info(f"Negative Stance label:")
			analysis_csv_rows.append(["Negative Stance label:"])
			neg_stance_rows = log_TP_FP_FN_TN_convs_from_stance_predictions(dev_ids_to_conv_predictions, 2)
			analysis_csv_rows.extend(neg_stance_rows)
			# Save the csv for further analysis
			save_list_of_tuples_to_tsv(analysis_csv_rows, os.path.join(args.output_dir, "pos_neg_stance_TP_FP_FN_analysis_rows.csv"), header=None, delimiter=',')
			# Log top scoring examples for positive and negative stance
			analysis_csv_rows = list()
			logging.info(f"Top Positive and Negative Stance prediction convs.")
			analysis_csv_rows.append(["OC_S Dev Stance label Conv samples:", "Utterances", "My comments"])
			logging.info(f"Positive Stance Top convs:")
			analysis_csv_rows.append(["Positive Stance Top convs:"])
			pos_stance_rows = log_top_conv_stance_predictions(dev_ids_to_conv_predictions, 1, K=10)
			analysis_csv_rows.extend(pos_stance_rows)
			analysis_csv_rows.append([])
			logging.info(f"Negative Stance Top convs:")
			analysis_csv_rows.append(["Negative Stance Top convs:"])
			neg_stance_rows = log_top_conv_stance_predictions(dev_ids_to_conv_predictions, 2, K=10)
			analysis_csv_rows.extend(neg_stance_rows)
			# Save the csv for further analysis
			save_list_of_tuples_to_tsv(analysis_csv_rows, os.path.join(args.output_dir, "pos_neg_stance_top_convs_analysis_rows.csv"), header=None, delimiter=',')

			


			# Test set evaluation
			logging.info(f"Final evaluation on {task} Stance Test Set")
			# Evaluate on OC_S
			test_str_convs, test_convs, test_stance_u_id_pairs, test_predictions, test_prediction_scores, test_labels = make_predictions_on_pairwise_stance_dataset(test_dataloader, model, tokenizer, device, "test")
			# Evaluate and calculate F1s
			test_ids_to_conv_predictions, every_f1_and_cm = evaluate_OC_S_Bert_pairwise_stance_predictions(test_convs, test_stance_u_id_pairs, test_predictions, test_prediction_scores, f"Final {task} Test", args.adjacent_only)
			# Results are in the format stance_cm, stance_p, stance_r, stance_f1, stance_support, predictions, scores, labels
			oc_s_no_stance_f1 = every_f1_and_cm["adj"][3][0]
			oc_s_agree_stance_f1 = every_f1_and_cm["adj"][3][1]
			oc_s_disagree_stance_f1 = every_f1_and_cm["adj"][3][2]
			oc_s_predictions, oc_s_scores, oc_s_labels = every_f1_and_cm["adj"][5:8]
			oc_s_avg_stance_f1 = (oc_s_no_stance_f1 + oc_s_agree_stance_f1 + oc_s_disagree_stance_f1)/3
			# Log the results and plot PR curves
			no_stance_pr_curve_save_file = os.path.join(args.output_dir, "test_no_stance_test_pr_cruve.png")
			pos_stance_pr_curve_save_file = os.path.join(args.output_dir, "test_positive_stance_test_pr_cruve.png")
			neg_stance_pr_curve_save_file = os.path.join(args.output_dir, "test_negative_stance_test_pr_cruve.png")
			logging.info(f"Plotting the positive and negative stance PR curves and saving them at {pos_stance_pr_curve_save_file} and {neg_stance_pr_curve_save_file} respectively")
			no_stance_scores = [e[0] for e in oc_s_scores]
			pos_stance_scores = [e[1] for e in oc_s_scores]
			neg_stance_scores = [e[2] for e in oc_s_scores]
			no_stance_labels = [e == 0 for e in oc_s_labels]
			pos_stance_labels = [e == 1 for e in oc_s_labels]
			neg_stance_labels = [e == 2 for e in oc_s_labels]
			p, r, t = draw_and_save_precision_recall_curve(no_stance_scores, no_stance_labels, f"{task} Test No Stance PR curve for DGPT Stance classifier (Adjacent)", "DGPT No Stance", no_stance_pr_curve_save_file)
			no_stance_pr_values_save_file = os.path.join(args.output_dir, "test_no_stance_pr_curve_values.tsv")
			save_list_of_tuples_to_tsv(zip(p,r,t), no_stance_pr_values_save_file, header=['precision', 'recall', 'threshold'], delimiter='\t')
			p, r, t = draw_and_save_precision_recall_curve(pos_stance_scores, pos_stance_labels, f"{task} Test Positive Stance PR curve for DGPT Stance classifier (Adjacent)", "DGPT Pos Stance", pos_stance_pr_curve_save_file)
			pos_stance_pr_values_save_file = os.path.join(args.output_dir, "test_positive_stance_pr_curve_values.tsv")
			save_list_of_tuples_to_tsv(zip(p,r,t), pos_stance_pr_values_save_file, header=['precision', 'recall', 'threshold'], delimiter='\t')
			p, r, t = draw_and_save_precision_recall_curve(neg_stance_scores, neg_stance_labels, f"{task} Test Negative Stance PR curve for DGPT Stance classifier (Adjacent)", "DGPT Neg Stance", neg_stance_pr_curve_save_file)
			neg_stance_pr_values_save_file = os.path.join(args.output_dir, "test_negative_stance_pr_curve_values.tsv")
			save_list_of_tuples_to_tsv(zip(p,r,t), neg_stance_pr_values_save_file, header=['precision', 'recall', 'threshold'], delimiter='\t')


			# Log the examples of TP FP FN and TN
			analysis_csv_rows = list()
			logging.info(f"OC_S Test Stance label Conv samples:")
			analysis_csv_rows.append(["OC_S Test Stance label Conv samples:", "Utterances", "My comments"])
			logging.info(f"Positive Stance label:")
			analysis_csv_rows.append(["Positive Stance label:"])
			pos_stance_rows = log_TP_FP_FN_TN_convs_from_stance_predictions(test_ids_to_conv_predictions, 1)
			analysis_csv_rows.extend(pos_stance_rows)
			analysis_csv_rows.append([])
			logging.info(f"Negative Stance label:")
			analysis_csv_rows.append(["Negative Stance label:"])
			neg_stance_rows = log_TP_FP_FN_TN_convs_from_stance_predictions(test_ids_to_conv_predictions, 2)
			analysis_csv_rows.extend(neg_stance_rows)
			# Save the csv for further analysis
			save_list_of_tuples_to_tsv(analysis_csv_rows, os.path.join(args.output_dir, "test_pos_neg_stance_TP_FP_FN_analysis_rows.csv"), header=None, delimiter=',')
			# Log top scoring examples for positive and negative stance
			analysis_csv_rows = list()
			logging.info(f"Top Positive and Negative Stance prediction convs.")
			analysis_csv_rows.append(["OC_S Test Stance label Conv samples:", "Utterances", "My comments"])
			logging.info(f"Positive Stance Top convs:")
			analysis_csv_rows.append(["Positive Stance Top convs:"])
			pos_stance_rows = log_top_conv_stance_predictions(test_ids_to_conv_predictions, 1, K=10)
			analysis_csv_rows.extend(pos_stance_rows)
			analysis_csv_rows.append([])
			logging.info(f"Negative Stance Top convs:")
			analysis_csv_rows.append(["Negative Stance Top convs:"])
			neg_stance_rows = log_top_conv_stance_predictions(test_ids_to_conv_predictions, 2, K=10)
			analysis_csv_rows.extend(neg_stance_rows)
			# Save the csv for further analysis
			save_list_of_tuples_to_tsv(analysis_csv_rows, os.path.join(args.output_dir, "test_pos_neg_stance_top_convs_analysis_rows.csv"), header=None, delimiter=',')
		else:
			# No else here
			logging.error(f"Unknown Stance task {task}. Terminating")
			exit()
		exit()
		

if __name__ == '__main__':
	main()