# We will get the fine-tuning data for Controllable text generation experiments

from utils import RANDOM_SEED, log_list, print_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, save_in_json, load_from_json, \
					format_time, get_number_of_lines, write_list_to_file, save_list_of_tuples_to_tsv, get_ngrams_from_sentence, \
					get_ngram_freq_from_corpus, normalize_vocab, get_num_of_word_in_corpus, save_in_jsonl, load_from_jsonl, load_from_tsv_to_list_of_list, save_list_of_tuples_to_tsv, save_in_txt
import os
import subprocess
import ast
import pdb
import random
from collections import Counter
random.seed(RANDOM_SEED)

from OC_S_utils import Conversation_Data, get_conversation_data_from_OC_S_file, get_save_lists_from_conv_data

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", help="Pickle file containing conversations with stance and offensive predictions", type=str, required=True)
parser.add_argument("-o", "--output_dir", help="Directory where the various fine-tuning splits of this program will be saved", type=str, required=True)
args = parser.parse_args()

import logging
# Ref: https://stackoverflow.com/a/49202811/4535284
for handler in logging.root.handlers[:]:
	logging.root.removeHandler(handler)

make_dir_if_not_exists(args.output_dir)
logfile = os.path.join(args.output_dir, "output.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(logfile, mode='w'), logging.StreamHandler()])

OFF_LABEL = "[OFF]"
SAFE_LABEL = "[SAFE]"
POS_STANCE_LABEL = "[AGREE]"
NO_STANCE_LABEL = "[NO-STANCE]"

def get_posts_from_convs(convs):
	return [conv.utterance_data[0]["comment"][2:] for conv in convs]

def sample_threads_with_unique_posts_from_threads(threads, SAMPLE_SIZE=300, previous_threads=None):
	unique_posts_to_threads = dict()
	for element in threads:
		subreddit, thread, prediction_dict = element
		post_with_subreddit = thread.split(" EOS ")[0]
		post = post_with_subreddit[post_with_subreddit.find("Title"):]
		unique_posts_to_threads.setdefault(post, list())
		unique_posts_to_threads[post].append(element)

	# If previous threads are given then remove those posts from unique_posts_to_threads
	if previous_threads:
		for element in previous_threads:
			subreddit, thread, prediction_dict = element
			post_with_subreddit = thread.split(" EOS ")[0]
			post = post_with_subreddit[post_with_subreddit.find("Title"):]
			if post in unique_posts_to_threads:
				del unique_posts_to_threads[post]

	# Sample one thread from each unique post
	if len(unique_posts_to_threads) <= SAMPLE_SIZE:
		return [random.choice(threads) for post, threads in unique_posts_to_threads.items()]
	else:
		unique_posts = list(unique_posts_to_threads.keys())
		sampled_posts = random.sample(unique_posts, SAMPLE_SIZE)
		return [random.choice(unique_posts_to_threads[post]) for post in sampled_posts]

def get_posts_from_threads(threads):
	posts = list()
	for subreddit, thread, prediction_dict in threads:
		post_with_subreddit = thread.split(" EOS ")[0]
		post = post_with_subreddit[post_with_subreddit.find("Title"):]
		posts.append(post)
	return posts

def get_thread_strings_from_thread_predictions(threads):
	all_threads = list()
	for subreddit, thread, prediction_dict in threads:
		# Remove the subreddit marker from the thread
		thread = thread[thread.find("Title:"):]
		all_threads.append(thread)
	return all_threads

def get_last_response_from_threads(threads):
	last_responses = list()
	for subreddit, thread, prediction_dict in threads:
		utterances = [e for e in thread.split(" EOS ") if e]
		last_responses.append(utterances[-1])
	return last_responses

def filter_threads_based_on_posts(threads, posts):
	filtered_threads = list()
	for element in threads:
		subreddit, thread, prediction_dict = element
		post_with_subreddit = thread.split(" EOS ")[0]
		post = post_with_subreddit[post_with_subreddit.find("Title"):]
		if post in posts:
			# Filter this thread
			continue
		# Else keep this thread
		filtered_threads.append(element)
	return filtered_threads

def get_stance_threads(threads, pos_stance_threshold, neg_stance_threshold, no_stance_threshold):
	no_stance_threads = list()
	pos_stance_threads = list()
	neg_stance_threads = list()
	for element in threads:
		subreddit, thread, prediction_dict = element
		stance_predictions = prediction_dict["stance"]
		if stance_predictions[-1][2][0] >= no_stance_threshold:
			no_stance_threads.append(element)
		elif stance_predictions[-1][2][1] >= pos_stance_threshold:
			pos_stance_threads.append(element)
		elif stance_predictions[-1][2][2] >= neg_stance_threshold:
			neg_stance_threads.append(element)
	return pos_stance_threads, neg_stance_threads, no_stance_threads

def log_threads(threads, K=10):
	count = 0
	log_rows = list()
	for element in threads:
		subreddit, thread, prediction_dict = element
		# logging.info(f"{count+1}\t{thread}")
		# logging.info(prediction_dict)
		log_rows.append((count+1, thread, prediction_dict["stance"][-1], prediction_dict["offensive"]))
		count += 1
		if count == K:
			return log_rows

def get_offensive_control_corpus_from_threads(threads):
	# Based on the last offensive score assign off-labels to the last utterance
	threads_with_indicators = list()
	n_off_threads = 0
	for element in threads:
		subreddit, thread, prediction_dict = element
		# Remove the subreddit marker from the thread
		thread = thread[thread.find("Title:"):]
		# Get the utterances
		utterances = [e for e in thread.split(" EOS ") if e]
		# Remove the last reply
		thread_without_last_reply = " EOS ".join(utterances[:-1])
		# Check last reply offensive prediction
		off_preds = prediction_dict["offensive"]
		OFF_INDICATOR = OFF_LABEL if off_preds[-1][1][0] < off_preds[-1][1][1] else SAFE_LABEL
		if OFF_INDICATOR == OFF_LABEL:
			n_off_threads += 1
		# Prepare the final thread to be saved
		thread_with_indicator = thread_without_last_reply + " EOS " + OFF_INDICATOR + utterances[-1] + " EOS "
		threads_with_indicators.append(thread_with_indicator)
	logging.info(f"TOTAL OFF THREADs = {n_off_threads}")
	return threads_with_indicators

def get_off_and_safe_control_thread_counts(threads_with_indicators):
	off_count = 0.0
	safe_count = 0.0
	total_count = 0.0
	for thread in threads_with_indicators:
		if OFF_LABEL in thread:
			off_count += 1.0
		elif SAFE_LABEL in thread:
			safe_count += 1.0
		total_count += 1.0
	return off_count, safe_count, total_count

def get_stance_control_corpus_from_threads(threads):
	# Based on the last stance score assign stance-labels to the last utterance
	threads_with_indicators = list()
	for element in threads:
		subreddit, thread, prediction_dict = element
		# Remove the subreddit marker from the thread
		thread = thread[thread.find("Title:"):]
		# Get the utterances
		utterances = [e for e in thread.split(" EOS ") if e]
		# Remove the last reply
		thread_without_last_reply = " EOS ".join(utterances[:-1])
		# Check last reply stance prediction
		stance_preds = prediction_dict["stance"]
		STANCE_INDICATOR = POS_STANCE_LABEL if stance_preds[-1][2][0] < stance_preds[-1][2][1] else NO_STANCE_LABEL
		# Prepare the final thread to be saved
		thread_with_indicator = thread_without_last_reply + " EOS " + STANCE_INDICATOR + utterances[-1] + " EOS "
		threads_with_indicators.append(thread_with_indicator)
	return threads_with_indicators

def get_both_control_corpus_from_threads(threads):
	# Based on the last stance score assign both off and stance-labels to the last utterance
	threads_with_indicators = list()
	for element in threads:
		subreddit, thread, prediction_dict = element
		# Remove the subreddit marker from the thread
		thread = thread[thread.find("Title:"):]
		# Get the utterances
		utterances = [e for e in thread.split(" EOS ") if e]
		# Remove the last reply
		thread_without_last_reply = " EOS ".join(utterances[:-1])
		# Check last reply offensive prediction
		off_preds = prediction_dict["offensive"]
		OFF_INDICATOR = OFF_LABEL if off_preds[-1][1][0] < off_preds[-1][1][1] else SAFE_LABEL
		# Check last reply stance prediction
		stance_preds = prediction_dict["stance"]
		STANCE_INDICATOR = POS_STANCE_LABEL if stance_preds[-1][2][0] < stance_preds[-1][2][1] else NO_STANCE_LABEL
		# Prepare the final thread to be saved
		thread_with_indicator = thread_without_last_reply + " EOS " + OFF_INDICATOR + STANCE_INDICATOR + utterances[-1] + " EOS "
		threads_with_indicators.append(thread_with_indicator)
	return threads_with_indicators

def split_threads_into_dev_and_train_based_on_posts(threads_with_indicators, dev_split_percent=0.05):
	posts = [thread.split(" EOS ")[0] for thread in threads_with_indicators]
	unique_posts = list(set(posts))
	dev_posts_size = int(dev_split_percent * len(unique_posts))
	dev_posts = set(random.sample(unique_posts, dev_posts_size))
	# Split the threads into dev threads and train threads
	dev_threads = list()
	train_threads = list()
	for thread in threads_with_indicators:
		post = thread.split(" EOS ")[0]
		if post in dev_posts:
			dev_threads.append(thread)
		else:
			train_threads.append(thread)
	return train_threads, dev_threads

def save_train_and_dev_threads_in_pkl_files(train_threads, dev_threads, save_file_prefix):
	train_save_file = save_file_prefix + "_train.pkl"
	save_in_pickle(train_threads, train_save_file)
	dev_save_file = save_file_prefix + "_dev.pkl"
	save_in_pickle(dev_threads, dev_save_file)

def main():
	#1. Read the conversations and offensive and stance predictions from input pickle file
	final_post_threads_and_predictions = load_from_pickle(args.input_file)
	logging.info(f"Total number of threads = {len(final_post_threads_and_predictions)}")
	pos_stance_threshold = 0.90
	no_stance_threshold = 0.9996
	neg_stance_threshold = 0.876
	offensive_threshold = 0.877
	# 0.8012048192771084	0.44039735099337746	0.8947051763534546
	# 0.7987421383647799	0.4703703703703704	0.8771452307701111
	safe_threshold = 0.993
	# 0.9843444227005871	0.5	0.9932619389146566
	# 0.988527724665392	0.5	0.9951793178915977
	
	#2. We want to create multiple different subsets
	# First we will create test conversations
	# 500 offensive threads with last utterance as offensive
	# 500 offensive threads with last utterance safe but the thread is offensive
	# [No longer needed] 300 safe threads with last utterance as safe
	off_threads_with_off_reply = list()
	off_threads_with_safe_reply = list()
	safe_threads = list()
	for element in final_post_threads_and_predictions:
		subreddit, thread, prediction_dict = element
		post_with_subreddit = thread.split(" EOS ")[0]
		post = post_with_subreddit[post_with_subreddit.find("Title"):]
		stance_predictions = prediction_dict['stance']
		offensive_predictions = prediction_dict['offensive']
		
		# Check if all predictions are safe
		safe = [off_scores[0] >= safe_threshold for u_id, off_scores in offensive_predictions]
		if all(safe):
			# save this in safe threads list
			safe_threads.append(element)
			continue
		off = [off_scores[1] >= offensive_threshold for u_id, off_scores in offensive_predictions]
		if any(off):
			# check the last reply offensive label
			last_reply_off = offensive_predictions[-1][1][1] >= offensive_threshold
			last_reply_safe = offensive_predictions[-1][1][0] >= safe_threshold
			if last_reply_off:
				# save this in off threads with off reply
				off_threads_with_off_reply.append(element)
				continue
			if last_reply_safe:
				# save this in off threads with safe reply
				off_threads_with_safe_reply.append(element)
	#2.1 shuffle collected subsets
	random.shuffle(off_threads_with_off_reply)
	random.shuffle(off_threads_with_safe_reply)
	random.shuffle(safe_threads)
	#2.2 print statistics of all splits
	logging.info(f"Off threads with off reply = {len(off_threads_with_off_reply)}")
	logging.info(f"Off threads with safe reply = {len(off_threads_with_safe_reply)}")
	logging.info(f"safe threads = {len(safe_threads)}")

	#3. Extract the test set from extracted threads
	test_size = 500
	permissible_test_off_threads_with_off_reply = [element for element in off_threads_with_off_reply if len(element[2]["offensive"]) <= 3]
	test_off_threads_with_off_reply = sample_threads_with_unique_posts_from_threads(permissible_test_off_threads_with_off_reply, test_size)
	permissible_test_off_threads_with_safe_reply = [element for element in off_threads_with_safe_reply if len(element[2]["offensive"]) <= 3]
	test_off_threads_with_safe_reply = sample_threads_with_unique_posts_from_threads(permissible_test_off_threads_with_safe_reply, test_size, test_off_threads_with_off_reply)
	permissible_test_safe_threads = [element for element in safe_threads if len(element[2]["offensive"]) <= 3]
	test_safe_threads = sample_threads_with_unique_posts_from_threads(permissible_test_safe_threads, test_size)
	#3.1 Count total unique posts in test threads
	# test_posts = get_posts_from_threads(test_off_threads_with_off_reply + test_off_threads_with_safe_reply + test_safe_threads)
	# NOTE: getting rid of the test safe threads as they don't evaluate what we want to evaluate
	test_posts = get_posts_from_threads(test_off_threads_with_off_reply + test_off_threads_with_safe_reply)
	logging.info(f"Total number of unique posts in test threads = {len(set(test_posts))}/{len(test_posts)}")

	#3.2 Save test threads in pickle file
	# test_threads = get_thread_strings_from_thread_predictions(test_off_threads_with_off_reply + test_off_threads_with_safe_reply + test_safe_threads)
	# NOTE: getting rid of the test safe threads as they don't evaluate what we want to evaluate
	test_threads = get_thread_strings_from_thread_predictions(test_off_threads_with_off_reply + test_off_threads_with_safe_reply)
	test_threads_pkl_file = os.path.join(args.output_dir, "test_threads.pkl")
	save_in_pickle(test_threads, test_threads_pkl_file)

	#3.3 Filter threads from test posts from the rest of the samples
	off_threads_with_off_reply = filter_threads_based_on_posts(off_threads_with_off_reply, set(test_posts))
	off_threads_with_safe_reply = filter_threads_based_on_posts(off_threads_with_safe_reply, set(test_posts))
	safe_threads = filter_threads_based_on_posts(safe_threads, set(test_posts))

	#3.4 Log threads statistics after filtering
	logging.info(f"Logging thread counts of different subsets after removing test threads")
	logging.info(f"Off threads with off reply = {len(off_threads_with_off_reply)}")
	logging.info(f"Off threads with safe reply = {len(off_threads_with_safe_reply)}")
	logging.info(f"safe threads = {len(safe_threads)}")



	#4. Extract pos and no stance threads from all subsets
	pos_stance_off_threads_off_reply, neg_stance_off_threads_off_reply, no_stance_off_threads_off_reply = get_stance_threads(off_threads_with_off_reply, pos_stance_threshold, neg_stance_threshold, no_stance_threshold)
	pos_stance_off_threads_safe_reply, neg_stance_off_threads_safe_reply, no_stance_off_threads_safe_reply = get_stance_threads(off_threads_with_safe_reply, pos_stance_threshold, neg_stance_threshold, no_stance_threshold)
	pos_stance_safe_threads, neg_stance_safe_threads, no_stance_safe_threads = get_stance_threads(safe_threads, pos_stance_threshold, neg_stance_threshold, no_stance_threshold)

	#4.1 Log final statistics
	all_log_rows = list()
	logging.info(f"Off threads with off replies total = {len(off_threads_with_off_reply)}")
	logging.info(f"Pos stance threads = {len(pos_stance_off_threads_off_reply)} vs Neg stance threads = {len(neg_stance_off_threads_off_reply)} vs No stance threads = {len(no_stance_off_threads_off_reply)}")
	# logging.info(f"\nPos Stance Off threads with off replies:")
	all_log_rows.append(["Off threads with off replies:"])
	all_log_rows.append(["Pos Stance:"])
	all_log_rows.extend(log_threads(pos_stance_off_threads_off_reply))

	all_log_rows.append(["Neg Stance:"])
	all_log_rows.extend(log_threads(neg_stance_off_threads_off_reply))
	# logging.info(f"\nNo Stance Off threads with off replies:")
	all_log_rows.append(["No Stance:"])
	all_log_rows.extend(log_threads(no_stance_off_threads_off_reply))
	all_log_rows.append([])

	logging.info(f"Off threads with safe replies total = {len(off_threads_with_safe_reply)}")
	logging.info(f"Pos stance threads = {len(pos_stance_off_threads_safe_reply)} vs Neg stance threads = {len(neg_stance_off_threads_safe_reply)} vs No stance threads = {len(no_stance_off_threads_safe_reply)}")
	# logging.info(f"\nPos Stance Off threads with safe replies:")
	all_log_rows.append(["Off threads with safe replies:"])
	all_log_rows.append(["Pos Stance:"])
	all_log_rows.extend(log_threads(pos_stance_off_threads_safe_reply))
	all_log_rows.append(["Neg Stance:"])
	all_log_rows.extend(log_threads(neg_stance_off_threads_safe_reply))
	# logging.info(f"\nNo Stance Off threads with safe replies:")
	all_log_rows.append(["No Stance:"])
	all_log_rows.extend(log_threads(no_stance_off_threads_safe_reply))
	all_log_rows.append([])
	
	logging.info(f"Safe threads total = {len(safe_threads)}")
	logging.info(f"Pos stance threads = {len(pos_stance_safe_threads)} vs Neg stance threads = {len(neg_stance_safe_threads)} vs No stance threads = {len(no_stance_safe_threads)}")
	# logging.info(f"\nPos Stance safe threads:")
	all_log_rows.append(["Safe threads:"])
	all_log_rows.append(["Pos Stance:"])
	all_log_rows.extend(log_threads(pos_stance_safe_threads))
	all_log_rows.append(["Neg Stance:"])
	all_log_rows.extend(log_threads(neg_stance_safe_threads))
	# logging.info(f"\nNo Stance safe threads:")
	all_log_rows.append(["No Stance:"])
	all_log_rows.extend(log_threads(no_stance_safe_threads))
	all_log_rows.append([])

	threads_log_save_file = os.path.join(args.output_dir, "threads_sample_for_manual_evaluation.csv")
	save_list_of_tuples_to_tsv(all_log_rows, threads_log_save_file, header=None, delimiter=',')

	#5. Create corpus for different control

	#5.1 offensive vs safe control
	# In this we will use off_threads_with_off_reply off_threads_with_safe_reply and safe threads (subsets from all 3)
	safe_sample_size = 200000 - (len(off_threads_with_off_reply) + len(off_threads_with_safe_reply))
	off_control_threads = off_threads_with_off_reply + off_threads_with_safe_reply + random.sample(safe_threads, safe_sample_size)
	# Shuffle after creating the sample
	random.shuffle(off_control_threads)
	off_control_posts = get_posts_from_threads(off_control_threads)
	logging.info(f"Total off control threads = {len(off_control_threads)} with number of unique posts = {len(set(off_control_posts))}")
	logging.info(f"Off threads off reply = {len(off_threads_with_off_reply)}, off threads safe reply = {len(off_threads_with_safe_reply)}, safe threads = {safe_sample_size}")
	off_control_threads_with_indicators = get_offensive_control_corpus_from_threads(off_control_threads)
	#5.1.1 split the final threads into train and dev segments
	train_off_control_threads, dev_off_control_threads = split_threads_into_dev_and_train_based_on_posts(off_control_threads_with_indicators)
	logging.info(f"Total train off control threads = {len(train_off_control_threads)} vs dev off control threads = {len(dev_off_control_threads)}")
	#5.1.2 save the final splits in txt file for model fine-tuning
	off_control_save_prefix = os.path.join(args.output_dir, "off_control")
	logging.info(f"Saving the off control train and dev threads at {off_control_save_prefix} ...\n\n")
	save_train_and_dev_threads_in_pkl_files(train_off_control_threads, dev_off_control_threads, off_control_save_prefix)
	#5.2.3 save off_responses separately for negative samples
	off_responses = get_last_response_from_threads(off_threads_with_off_reply)
	off_resposnes_safe_file = os.path.join(args.output_dir, "off_responses.pkl")
	logging.info(f"Saving {len(off_responses)} offensive responses at {off_resposnes_safe_file} ...")
	save_in_pickle(off_responses, off_resposnes_safe_file)

	#5.2 Pos Stance vs No Stance control
	# We will create 3 experiments in this case. 
	# 1 - both off and safe data = all_stance
	# In this we will use off_threads_with_off_reply off_threads_with_safe_reply and safe_threads (subsets from all 3)
	all_stance_control_threads = pos_stance_off_threads_off_reply + no_stance_off_threads_off_reply + pos_stance_off_threads_safe_reply + no_stance_off_threads_safe_reply + random.sample(pos_stance_safe_threads, 70000) + random.sample(no_stance_safe_threads, 80000)
	# Shuffle after creating the sample
	random.shuffle(all_stance_control_threads)
	all_stance_control_posts = get_posts_from_threads(all_stance_control_threads)
	logging.info(f"Total all (TYPE 1) stance control threads = {len(all_stance_control_threads)} with number of unique posts = {len(set(all_stance_control_posts))}")
	all_stance_control_threads_with_indicators = get_stance_control_corpus_from_threads(all_stance_control_threads)
	#5.2.1 split the final threads into train and dev segments
	train_all_stance_control_threads, dev_all_stance_control_threads = split_threads_into_dev_and_train_based_on_posts(all_stance_control_threads_with_indicators)
	logging.info(f"Total train all (TYPE 1) stance control threads = {len(train_all_stance_control_threads)} vs dev all (TYPE 1) stance control threads = {len(dev_all_stance_control_threads)}")
	#5.2.1.1 save the final splits in txt file for model fine-tuning
	all_stance_control_save_prefix = os.path.join(args.output_dir, "all_stance_control")
	logging.info(f"Saving the off control train and dev threads at {all_stance_control_save_prefix} ...\n\n")
	save_train_and_dev_threads_in_pkl_files(train_all_stance_control_threads, dev_all_stance_control_threads, all_stance_control_save_prefix)

	# 2 - only safe replies = safe_reply_stance
	# In this we will use off_threads_with_safe_reply and safe_threads (subsets from last 2)
	safe_reply_stance_control_threads = pos_stance_off_threads_safe_reply + no_stance_off_threads_safe_reply + pos_stance_safe_threads + random.sample(no_stance_safe_threads, 100000)
	# Shuffle after creating the sample
	random.shuffle(safe_reply_stance_control_threads)
	safe_reply_stance_control_posts = get_posts_from_threads(safe_reply_stance_control_threads)
	logging.info(f"Total safe reply (TYPE 2) stance control threads = {len(safe_reply_stance_control_threads)} with number of unique posts = {len(set(safe_reply_stance_control_posts))}")
	safe_reply_stance_control_threads_with_indicators = get_stance_control_corpus_from_threads(safe_reply_stance_control_threads)
	#5.2.2 split the final threads into train and dev segments
	train_safe_reply_stance_control_threads, dev_safe_reply_stance_control_threads = split_threads_into_dev_and_train_based_on_posts(safe_reply_stance_control_threads_with_indicators)
	logging.info(f"Total train safe reply (TYPE 2) stance control threads = {len(train_safe_reply_stance_control_threads)} vs dev safe reply (TYPE 2) stance control threads = {len(dev_safe_reply_stance_control_threads)}")
	#5.2.2.1 save the final splits in txt file for model fine-tuning
	safe_reply_stance_control_save_prefix = os.path.join(args.output_dir, "safe_reply_stance_control")
	logging.info(f"Saving the off control train and dev threads at {safe_reply_stance_control_save_prefix} ...\n\n")
	save_train_and_dev_threads_in_pkl_files(train_safe_reply_stance_control_threads, dev_safe_reply_stance_control_threads, safe_reply_stance_control_save_prefix)
	
	# 3 - only safe threads = safe_stance
	# In this we will only use safe_threads
	safe_stance_control_threads = pos_stance_safe_threads + random.sample(no_stance_safe_threads, 200000)
	# Shuffle after creating the sample
	random.shuffle(safe_stance_control_threads)
	safe_stance_control_posts = get_posts_from_threads(safe_stance_control_threads)
	logging.info(f"Total safe only (TYPE 3) stance control threads = {len(safe_stance_control_threads)} with number of unique posts = {len(set(safe_stance_control_posts))}")
	safe_stance_control_threads_with_indicators = get_stance_control_corpus_from_threads(safe_stance_control_threads)
	#5.2.3 split the final threads into train and dev segments
	train_safe_stance_control_threads, dev_safe_stance_control_threads = split_threads_into_dev_and_train_based_on_posts(safe_stance_control_threads_with_indicators)
	logging.info(f"Total train safe only (TYPE 3) stance control threads = {len(train_safe_stance_control_threads)} vs dev safe only (TYPE 3) stance control threads = {len(dev_safe_stance_control_threads)}")
	#5.2.3.1 save the final splits in txt file for model fine-tuning
	safe_stance_control_save_prefix = os.path.join(args.output_dir, "safe_stance_control")
	logging.info(f"Saving the off control train and dev threads at {safe_stance_control_save_prefix} ...\n\n")
	save_train_and_dev_threads_in_pkl_files(train_safe_stance_control_threads, dev_safe_stance_control_threads, safe_stance_control_save_prefix)

	#5.3 Both Offensive and Stance control
	both_control_threads = all_stance_control_threads
	both_control_posts = get_posts_from_threads(both_control_threads)
	logging.info(f"Total both control threads = {len(both_control_threads)} with number of unique posts = {len(set(both_control_posts))}")
	both_control_threads_with_indicators = get_both_control_corpus_from_threads(both_control_threads)
	#5.3.1 split the final threads into train and dev segments
	train_both_control_threads, dev_both_control_threads = split_threads_into_dev_and_train_based_on_posts(both_control_threads_with_indicators)
	logging.info(f"Total train both control threads = {len(train_both_control_threads)} vs dev both control threads = {len(dev_both_control_threads)}")
	#5.3.2 save the final splits in txt file for model fine-tuning
	both_control_save_prefix = os.path.join(args.output_dir, "both_control")
	logging.info(f"Saving the off control train and dev threads at {both_control_save_prefix} ...\n\n")
	save_train_and_dev_threads_in_pkl_files(train_both_control_threads, dev_both_control_threads, both_control_save_prefix)
if __name__ == '__main__':
	main()