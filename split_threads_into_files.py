# We will split the threads into smaller splits

from utils import RANDOM_SEED, log_list, print_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, save_in_json, load_from_json, \
					format_time, get_number_of_lines, write_list_to_file, save_list_of_tuples_to_tsv, get_ngrams_from_sentence, \
					get_ngram_freq_from_corpus, normalize_vocab, get_num_of_word_in_corpus, save_in_jsonl, load_from_jsonl, load_from_tsv_to_list_of_list
import os
import subprocess
import ast
import pdb
import random
from collections import Counter
random.seed(RANDOM_SEED+4)

from OC_S_utils import Conversation_Data, get_conversation_data_from_OC_S_file, get_save_lists_from_conv_data

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", help="Path to the file where all the post threads are saved", type=str, required=True)
parser.add_argument("-o", "--output_dir", help="Directory where we will save the splits", type=str, required=True)
parser.add_argument("-n", "--n_splits", help="Number of splits", type=int, required=True)
args = parser.parse_args()

import logging
# Ref: https://stackoverflow.com/a/49202811/4535284
for handler in logging.root.handlers[:]:
	logging.root.removeHandler(handler)

make_dir_if_not_exists(args.output_dir)
logfile = os.path.join(args.output_dir, "output.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(logfile, mode='w'), logging.StreamHandler()])

def main():
	# Read the post-comments pickle file
	all_reddit_posts, all_reddit_post_id_to_index, all_reddit_posts_comments, all_reddit_comment_id_to_index, all_reddit_post_threads = load_from_pickle(args.input_file)
	
	total = len(all_reddit_post_threads)
	logging.info(f"Total post threads pairs in the dictionary = {total}")
	split_dicts = [dict() for i in range(args.n_splits)]
	split_size = int((total+10) / args.n_splits)
	count = 0.0
	prev_split_id = -1
	for k, v in all_reddit_post_threads.items():
		count += 1.0
		split_id = int(count // split_size)
		if split_id != prev_split_id:
			prev_split_id = split_id
			logging.info(f"Current split_id = {split_id} and count = {count}")
		# add to the split dict at split_id
		split_dicts[split_id][k] = v

	# Save all split dicts in different splits
	for i, split_dict in enumerate(split_dicts):
		# Save the current split_dict in its split pickle file
		split_save_file = os.path.join(args.output_dir, f"split_{i}.pkl")
		logging.info(f"Saving {len(split_dict)} post threads pairs at {split_save_file} ...")
		save_in_pickle((all_reddit_posts, all_reddit_post_id_to_index, all_reddit_posts_comments, all_reddit_comment_id_to_index, split_dict), split_save_file)

if __name__ == '__main__':
	main()