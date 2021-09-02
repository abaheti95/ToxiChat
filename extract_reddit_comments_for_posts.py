# We will extract the comment threads from all subreddits for specific posts that we have already shortlisted

from utils import RANDOM_SEED, log_list, print_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, save_in_json, \
 					load_from_json, load_from_jsonl, format_time, get_number_of_lines, write_list_to_file, \
 					save_list_of_tuples_to_tsv, get_ngrams_from_sentence, get_ngram_freq_from_corpus, normalize_vocab, \
 					get_num_of_word_in_corpus

import os
import pdb
import json
import random
random.seed(RANDOM_SEED)
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import zstandard as zstd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--files", help="List of reddit comment dump files", type=str, nargs='+')
parser.add_argument("-p", "--posts_file", help="File that contains shortlisted reddit posts", type=str, required=True)
parser.add_argument("-o", "--output_dir", help="Directory where the results of this program will be saved", type=str, required=True)
args = parser.parse_args()

import logging
# Ref: https://stackoverflow.com/a/49202811/4535284
for handler in logging.root.handlers[:]:
	logging.root.removeHandler(handler)

make_dir_if_not_exists(args.output_dir)
logfile = os.path.join(args.output_dir, "output.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(logfile, mode='w'), logging.StreamHandler()])


# We will store all the comments in a subreddit in a dictionary of dictionaries
# Upper level keys will be the subreddit names
# Each lower level key will be the comment id and the values will be details about the comment
save_file_writer = open(os.path.join(args.output_dir, f"all_subreddit_post_related_comments.jsonl"), "w")
total_saved_comments = 0

def save_comment_info_in_file(writer, comment, id, link_id, parent_id, score, author, retrieved_on, url):
	# Save each comment in a single line if possible
	# Prepare save dict
	comment_info = {"id":id,
					"parent_id":parent_id,
					"link_id":link_id,
					"score":score,
					"author":author,
					"retrieved_on":retrieved_on,
					"comment":comment,
					"url":url}
	comment_info_string = json.dumps(comment_info)
	writer.write(f"{comment_info_string}\n")

def read_reddit_comment_dump_and_save_post_related_comments(posts_ids, dump_file):
	global save_file_writer, total_saved_comments
	dctx = zstd.ZstdDecompressor()
	previous_line = ""
	chunk_index = 0
	with open(dump_file, 'rb') as fh:
		reader = dctx.stream_reader(fh)
		while True:
			chunk = reader.read(2**24)
			chunk_index += 1
			if not chunk:
				break
			# Extract string data from compressed chunk
			string_data = chunk.decode()
			lines = string_data.split("\n")
			for i, line in enumerate(lines[:-1]):
				if i == 0:
					line = previous_line + line
				comment_object = json.loads(line)
				# Extract the subreddit, comment, id, parent_id, author, score
				subreddit = comment_object["subreddit"]
				
				score = comment_object["score"]
				# NOTE: Adding a threshold on score to limit the data
				if score <= 1:
					continue
				comment = comment_object["body"]
				id = comment_object["id"]
				link_id = comment_object["link_id"]
				parent_id = comment_object["parent_id"]
				# Check if link_id is in the list of post_ids
				if link_id[3:] not in posts_ids:
					continue
				# print(link_id)
				# print(parent_id)
				# pdb.set_trace()
				author = comment_object["author"]
				retrieved_on = comment_object["retrieved_on"]
				url = comment_object['permalink']
				# Save comment information in global files
				if comment == "[deleted]" or comment == "[removed]" or "I am a bot" in comment:
					# ignore/remove this comment from the dataset
					continue
				save_comment_info_in_file(save_file_writer, comment, id, link_id, parent_id, score, author, retrieved_on, url)
				total_saved_comments += 1
			previous_line = lines[-1]
			if chunk_index % 100 == 0:
				logging.info(f"Chunk number: {chunk_index}. Total Comments:{total_saved_comments}")
				save_file_writer.flush()

def main():
	# First read all the posts
	all_posts = load_from_jsonl(args.posts_file)
	# Create set of post_ids from all_posts
	all_posts_ids = set([post["id"] for post in all_posts])
	for file in args.files:
		logging.info(f"Reading comments from file: {file}")
		read_reddit_comment_dump_and_save_post_related_comments(all_posts_ids, file)

	# Close all open files
	save_file_writer.close()

if __name__ == '__main__':
	main()

