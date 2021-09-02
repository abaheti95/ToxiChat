# We will fine-tune DGPT model on label controlled conversations
# We will provide a train file and a test file containing the threads with control label
# The model training and testing code will be implemented using the transformers library with pytorch backend

import sys
sys.path.insert(0,'..')
sys.path.insert(0,'.')
import utils
from utils import RANDOM_SEED, log_list, print_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, save_in_json, load_from_json, format_time, plot_train_loss, log_TP_FP_FN_TN_from_binary_predictions, draw_and_save_precision_recall_curve, save_list_of_tuples_to_tsv, load_from_txt
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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train_file", help="Path to the train threads with control labels", type=str, required=True)
parser.add_argument("-so", "--subset_only", help="Specific control label on whose responses we should fine-tune", type=str, default="")
parser.add_argument("-d", "--dev_file", help="Path to the dev threads with control labels", type=str, required=True)
parser.add_argument("-s", "--save_dir", help="Path to the directory where we will save model and the tokenizer", type=str, required=True)
parser.add_argument("-o", "--output_dir", help="Path to the output directory where we will save all the model prediction and results", type=str, required=True)
parser.add_argument("-dv", "--dev_log_frequency", help="How many times should we evaluate in each epoch", type=int, default=4)
parser.add_argument("-bs", "--batch_size", help="Train batch size for GPT2 model", type=int, default=32)
parser.add_argument("-d_bs", "--dev_batch_size", help="Train batch size for GPT2 model", type=int, default=4)
parser.add_argument("-e", "--n_epochs", help="Number of epochs", type=int, default=4)
parser.add_argument("-lr", "--learning_rate", help="Number of epochs", type=float, default=2e-5)
args = parser.parse_args()

import logging
# Ref: https://stackoverflow.com/a/49202811/4535284
for handler in logging.root.handlers[:]:
	logging.root.removeHandler(handler)
# Also add the stream handler so that it logs on STD out as well
# Ref: https://stackoverflow.com/a/46098711/4535284
make_dir_if_not_exists(args.output_dir)
make_dir_if_not_exists(args.save_dir)
logfile = os.path.join(args.output_dir, "output.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(logfile, mode='w'), logging.StreamHandler()])

OFF_LABEL = "[OFF]"
SAFE_LABEL = "[SAFE]"
POS_STANCE_LABEL = "[AGREE]"
NO_STANCE_LABEL = "[NO-STANCE]"

# PRETRAINED_GPT2_MODEL = 'GPT2-base-cased'
# PRETRAINED_GPT2_MODEL = 'microsoft/DialoGPT-small'
# Other global constants required for the code
# POSSIBLE_BATCH_SIZE = 2


PRETRAINED_GPT2_MODEL = 'microsoft/DialoGPT-medium'
# # Other global constants required for the code
POSSIBLE_BATCH_SIZE = 1

MAX_SEQ_THRESH = 512

if torch.cuda.is_available():
	device = torch.device("cuda")
	logging.info(f"Using GPU{torch.cuda.get_device_name(0)} to train")
else:
	device = torch.device("cpu")
	logging.info(f"Using CPU to train")

class CTG_TokenizeCollator():
	def __init__(self, tokenizer):
		self.tokenizer = tokenizer

	def __call__(self, batch):
		all_GPT2_model_input_texts = list()
		for i, thread in enumerate(batch):
			GPT2_string = thread.replace(" EOS ", self.tokenizer.eos_token)
			all_GPT2_model_input_texts.append(GPT2_string)

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
		
		return {"input_ids": input_ids, "attention_mask": attention_mask, "batch_data": batch}

class CTG_threads_Dataset(Dataset):
	"""CTG_threads_Dataset stores CTG thread strings as instances. It takes list of threads as input."""
	def __init__(self, threads, subset_only=None):
		super(CTG_threads_Dataset, self).__init__()
		if subset_only:
			logging.info(f"Keeping only the subset of the threads that contain {subset_only} marker ...")
			# We will only consider the threads that are in the given subset
			# For example, for offensive control CTG we will consider only the threads that are either [OFF] or [SAFE]
			new_threads = list()
			n_discarded = 0.0
			n_kept = 0.0
			n_total = 0.0
			for thread in threads:
				utterances = [e for e in thread.split(" EOS ") if e]
				if utterances[-1].startswith(subset_only):
					# Keep this thread
					n_kept += 1.0
					# Remove the label identifier and create a new thread
					utterances[-1] = utterances[-1][len(subset_only):]
					new_threads.append(" EOS ".join(utterances) + " EOS ")
				else:
					# discard this thread
					n_discarded += 1.0
				n_total += 1
			logging.info(f"Out of total {n_total} threads, kept vs discarded = {n_kept} vs {n_discarded}")
			threads = new_threads
		self.instances = threads
		self.nsamples = len(self.instances)

	def __getitem__(self, index):
		return self.instances[index]

	def __len__(self):
		return self.nsamples

def main():
	#1. Read the train and dev conversations
	logging.info(f"Loading train and dev threads from {args.train_file} and {args.dev_file}")
	train_threads = load_from_pickle(args.train_file)
	dev_threads = load_from_pickle(args.dev_file)
	logging.info(f"Total train threads = {len(train_threads)} vs Total dev threads = {len(dev_threads)}")

	#2.1 Create dataset and dataloaders from train and dev threads
	train_dataset = CTG_threads_Dataset(train_threads, args.subset_only)
	dev_dataset = CTG_threads_Dataset(dev_threads, args.subset_only)

	#2.2 Get the DGPT small tokenizer and model
	logging.info(f"Loading {PRETRAINED_GPT2_MODEL} pretrained model and tokenizer ...")
	tokenizer = GPT2Tokenizer.from_pretrained(PRETRAINED_GPT2_MODEL)
	model = GPT2LMHeadModel.from_pretrained(PRETRAINED_GPT2_MODEL)

	#2.3 Add our special tokens in the vocabulary
	num_added_toks = tokenizer.add_tokens([OFF_LABEL, SAFE_LABEL, POS_STANCE_LABEL, NO_STANCE_LABEL])
	logging.info(f"Current vocabulary size = {len(tokenizer)}")
	logging.info(f"Added {num_added_toks} tokens to DGPT vocabulary")
	tokenizer.pad_token = tokenizer.eos_token
	# Note: resize_token_embeddings expects to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
	model.resize_token_embeddings(len(tokenizer))
	model.to(device)
	logging.info(f"New vocabulary size = {len(tokenizer)}")

	#2.4 Initialize the collator with GPT2 tokenizer
	tokenize_collator = CTG_TokenizeCollator(tokenizer)

	#2.5 Initialize train and dev dataloaders
	train_dataloader = DataLoader(train_dataset, batch_size=POSSIBLE_BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=tokenize_collator)
	dev_dataloader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=False, num_workers=0, collate_fn=tokenize_collator)
	
	
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
	best_dev_ppl = 100000000.0
	# Dev validation trajectory
	log_softmax_fct = nn.LogSoftmax(dim=2)
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
			# test = labels * final_mask
			# pdb.set_trace()

			shift_mask = final_mask[..., 1:].contiguous()
			shift_labels = labels[..., 1:].contiguous()

			log_probs_flat = log_probs.view(-1, log_probs.size(-1))
			target_flat = shift_labels.view(-1, 1)
			losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
			losses = losses_flat.view(shift_labels.size(0), shift_labels.size(1))
			loss = (losses * shift_mask).sum()/shift_mask.sum()
								
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
				logging.info(f"############## Running Validation on the dev set ...")
				# Put the model in evaluation mode--the dropout layers behave differently
				# during evaluation.
				model.eval()

				# Validate on dev set by calculating perplexity or F1
				total_loss = 0.0
				with torch.no_grad():
					for batch in tqdm(dev_dataloader):
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
				avg_dev_loss = total_loss / len(dev_dataloader)
				perplexity = torch.exp(avg_dev_loss)
				logging.info(f"Validation Perplexity = {perplexity:3f}")
				if best_dev_ppl > perplexity:
					# Keep the copy of current model in cpu to avoid out of memory issues
					logging.info(f"New best dev Off F1 = {perplexity} achieved at epoch {epoch+1}")
					best_model = copy.deepcopy(model).cpu()
					best_model.save_pretrained(args.save_dir)
					torch.cuda.empty_cache()
					# Command to check if the model is on cuda
					# print(next(best_model.parameters()).is_cuda)
					best_dev_ppl = perplexity
					best_dev_epoch = epoch+1
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
	logging.info(f"Best Dev PPL = {best_dev_ppl} at epoch {best_dev_epoch}.")
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

if __name__ == '__main__':
	main()