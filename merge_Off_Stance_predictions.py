# We will merge the stance predictions we got from the splits

from utils import RANDOM_SEED, log_list, print_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, save_in_json, load_from_json, \
					format_time, get_number_of_lines, write_list_to_file, save_list_of_tuples_to_tsv, get_ngrams_from_sentence, \
					get_ngram_freq_from_corpus, normalize_vocab, get_num_of_word_in_corpus, save_in_jsonl, load_from_jsonl, load_from_tsv_to_list_of_list

import os
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", help="Path to the directory where all the split offensive and stance predictions are saved", type=str, required=True)
parser.add_argument("-o", "--output_file", help="Final pickle file where we will save all the merged predictions", type=str, required=True)
parser.add_argument("-n", "--n_splits", help="Number of splits", type=int, required=True)
args = parser.parse_args()

def main():
	all_preds = list()
	for i in range(args.n_splits):
		split_file = os.path.join(args.input_dir, f"split_{i}_preds.pkl")
		logging.info(f"loading predictions from {split_file}")
		all_preds.extend(load_from_pickle(split_file))
	# Save everything in the single final output pickle file
	logging.info(f"Saving {len(all_preds)} threads predictions at {args.output_file}")
	save_in_pickle(all_preds, args.output_file)	
if __name__ == '__main__':
	main()