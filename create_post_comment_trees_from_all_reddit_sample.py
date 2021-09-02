# After running extract_reddit_posts.py on all reddit posts dumps
# and running extract_reddit_comments_for_posts.py on reddit comments dumps 
# we extracted posts and comments trees of all subreddits in different jsonl files

# In this file we will create post-comment trees for all_subreddits and extract 3 or fewer turn conversations from it.
# A sample of these post-comment threads will be the input to the DGPT model

from utils import RANDOM_SEED, log_list, print_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, save_in_json, load_from_json, \
					format_time, get_number_of_lines, write_list_to_file, save_list_of_tuples_to_tsv, get_ngrams_from_sentence, \
					get_ngram_freq_from_corpus, normalize_vocab, get_num_of_word_in_corpus, save_in_jsonl, load_from_jsonl, remove_multiple_space, remove_markdown_urls, replace_urls, remove_markdown_urls, URL_TOKEN
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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-ic", "--input_comments_file", help="File where the comments of all reddit dumps are present in jsonl file", type=str, required=True)
parser.add_argument("-ip", "--input_posts_file", help="File where the posts of all reddit dumps are present in jsonl file", type=str, required=True)
parser.add_argument("-mc", "--max_comments", help="Maximum number of consecutive comments in the thread", type=int, default=2)
parser.add_argument("-o", "--output_dir", help="Directory where the results of this program will be saved", type=str, required=True)
args = parser.parse_args()

import logging
# Ref: https://stackoverflow.com/a/49202811/4535284
for handler in logging.root.handlers[:]:
	logging.root.removeHandler(handler)

make_dir_if_not_exists(args.output_dir)
logfile = os.path.join(args.output_dir, "output.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(logfile, mode='w'), logging.StreamHandler()])

def get_maximal_threads_from_start_comment(current_comment, previous_comments, comment_id_to_index, all_comments, K=3):
	children = current_comment["children"]
	all_sequences = list()
	if len(children) == 0 or K == 1:
		# Base condition. No more children to traverse
		
		# Append current comment at the end of previous_comments
		previous_comments.append((current_comment["id"], current_comment["comment"], current_comment["url"]))
		# Add the current list to all_sequences
		all_sequences.append(previous_comments)
	else:
		# Recursively traverse for each children and update the return sequences
		for child_id in children:
			# Retrive the child comment from all_comments
			child_comment = all_comments[comment_id_to_index[child_id]]
			all_sequences.extend(get_maximal_threads_from_start_comment(child_comment, previous_comments + [(current_comment["id"], current_comment["comment"], current_comment["url"])], comment_id_to_index, all_comments, K-1))
	return all_sequences

def check_if_post_or_comment_is_okay(post_or_comment):
	# we will pre-process the post or comment. 
	# First we will remove all the urls
	# Second we will remove all markdowns
	cleaned_post_or_comment, number_of_urls = replace_urls(post_or_comment, URL_TOKEN)
	cleaned_post_or_comment, n_links = remove_markdown_urls(cleaned_post_or_comment)

	if cleaned_post_or_comment == URL_TOKEN or not cleaned_post_or_comment:
		return False
	return cleaned_post_or_comment

def make_post_comment_threads_from_comments(posts, comments):
	# We will be given a 2 lists of json objects.
	# Each json post object will be a post containing "id", "title", "post", "score", "author", "retrieved_on", "url", "content_url"
	# Each json comment object will be a comment containing "id", "parent_id", "link_id", "score", "author", "retrieved_on", "comment"
	# We want to make all the links bi-directional i.e. parents should also point to children. Can be added in the comment_dict
	# Top level comments (direct reply to the posts) can be identified by a flag added in the comment_dict

	# Reddit prefixes. Ref: https://www.reddit.com/dev/api/
	# type prefixes
	# t1_	Comment
	# t2_	Account
	# t3_	Link
	# t4_	Message
	# t5_	Subreddit
	# t6_	Award

	# Filter comments based on number of words
	MAX_TOKS = 50
	logging.info(f"Filtering comments of length greater than {MAX_TOKS} tokens")
	prev_size = len(comments)
	comments = [comment for comment in comments if len(comment["comment"].split()) <= MAX_TOKS]
	logging.info(f"Previous size:{prev_size} and new size:{len(comments)}")

	post_id_to_index = dict()
	for i, post in enumerate(posts):
		clean_post = check_if_post_or_comment_is_okay(post["post"])
		post["ignore_post"] = False
		if not clean_post:
			# don't want only url posts
			post["ignore_post"] = True
			continue
		post["post"] = clean_post
		post_id_to_index[post["id"]] = i
		post.setdefault("children", set())

	# Update comment dicts with new variables and create a comment to list index lookup dictionary
	comment_id_to_index = dict()
	for i, comment in enumerate(comments):
		clean_comment = check_if_post_or_comment_is_okay(comment["comment"])
		comment["ignore_comment"] = False
		if not clean_comment:
			# don't want only url comments
			comment["ignore_comment"] = True
			continue
		comment["comment"] = clean_comment
		comment_id_to_index[comment["id"]] = i
		comment.setdefault("children", set())
		comment["parent_present"] = False

	# Now traverse the list of comments and update children
	parent_not_found = 0
	posts_children_found = 0
	for i, comment in enumerate(comments):
		if comment["ignore_comment"]:
			continue
		# Find parent and check if it is in the lookup index
		parent_id = comment["parent_id"]
		if parent_id[:3] != "t1_":
			# t3_ is the link to the post
			assert parent_id[:3] == "t3_", f"Unknown parent_id {parent_id} found when creating threads from posts and comments"

			# Check if parent_id is present in the post_id_to_index dict
			if parent_id[3:] in post_id_to_index:
				# If present the keep track of this comment
				parent_post_index = post_id_to_index[parent_id[3:]]
				posts[parent_post_index]["children"].add(comment["id"])
				posts_children_found += 1
				comment["parent_post_present"] = True
		elif parent_id[3:] in comment_id_to_index:
			# Add current comment to the parent's children list
			parent_comment_index = comment_id_to_index[parent_id[3:]]
			comments[parent_comment_index]["children"].add(comment["id"])
			# Update the flag in current comment
			comment["parent_comment_present"] = True
		else:
			# logging.info(f"{parent_id} not found in the lookup")
			parent_not_found += 1

	logging.info(f"Total comments with post as parent = {posts_children_found} and comments with no found parents = {parent_not_found}")

	# Create threads from posts by traversing its comment children
	all_post_comment_threads = dict()
	total_threads = 0
	for i, post in enumerate(posts):
		if post["ignore_post"]:
			continue
		if len(post["children"]) > 0:
			# Save the post signature in the keys of the all_post_comment_threads dict
			post_signature = (post["id"], post["title"], post["post"], post["url"], post["content_url"])
			all_post_comment_threads[post_signature] = list()
			# print(post_signature)

		for child_comment_id in post["children"]:
			# Get the child comment from the id
			comment = comments[comment_id_to_index[child_comment_id]]
			# Create threads of size 2 using recursion
			K = args.max_comments
			current_threads_with_urls = get_maximal_threads_from_start_comment(comment, list(), comment_id_to_index, comments, K)
			total_threads += len(current_threads_with_urls)
			# logging.info(f"For comment {comment['id']}, Number of threads of size {K} = {len(current_threads_with_urls)}. Total = {total_threads}")
			# print_list(current_threads_with_urls)
			# Add post's comment threads to the post_signature
			all_post_comment_threads[post_signature].extend(current_threads_with_urls) 
		if (i+1) % 10000 == 0:
			print(f"current post number = {i}/{len(posts)}")

	logging.info(f"Total comment threads found = {total_threads}")
	return all_post_comment_threads, post_id_to_index, posts, comment_id_to_index, comments, total_threads

def main():
	K = args.max_comments
	# Process subreddits one at a time
	posts_file = args.input_posts_file
	all_reddit_posts = load_from_jsonl(posts_file)
	comments_file =args.input_comments_file
	all_reddit_posts_comments = load_from_jsonl(comments_file)
	if len(all_reddit_posts) == 0 or len(all_reddit_posts_comments) == 0:
		logging.error(f"all_reddit: #posts = {len(all_reddit_posts)} vs #comments = {len(all_reddit_posts_comments)}. Skipping entire reddit LOL!")
		exit()
	all_reddit_post_threads, all_reddit_post_id_to_index, all_reddit_posts, all_reddit_comment_id_to_index, all_reddit_posts_comments, all_reddit_total_threads = make_post_comment_threads_from_comments(all_reddit_posts, all_reddit_posts_comments)
	logging.info(f"all_reddit: Number of posts = {len(all_reddit_posts)} || Number of comments = {len(all_reddit_posts_comments)} || Number of threads of size {K} or less = {all_reddit_total_threads}")
	all_reddit_post_and_comment_threads_save_file = os.path.join(args.output_dir, f"all_reddit_post_and_comments_{K}_threads.pkl")
	logging.info(f"Saving post objects, comment objects, lookup tables, and post-comment threads dict with ids and urls in pickle file: {all_reddit_post_and_comment_threads_save_file}")
	save_in_pickle((all_reddit_posts, all_reddit_post_id_to_index, all_reddit_posts_comments, all_reddit_comment_id_to_index, all_reddit_post_threads), all_reddit_post_and_comment_threads_save_file)
	all_reddit_post_comment_threads_save_file = os.path.join(args.output_dir, f"all_reddit_post_comment_{K}_threads_for_analysis.txt")
	logging.info(f"Saving the comment threads for analysis at {all_reddit_post_comment_threads_save_file}")
	with open(all_reddit_post_comment_threads_save_file, "w") as writer:
		for post_signature, thread_lists in all_reddit_post_threads.items():
			post_id, post_title, post, post_url, post_content_url = post_signature
			for thread_list in thread_lists:
				thread_string = f"Title:{remove_multiple_space(post_title)} EOS {remove_multiple_space(post)} EOS " + ' EOS '.join([remove_multiple_space(comment_string) for id, comment_string, url in thread_list])
				writer.write(f"{thread_string}\n")


if __name__ == '__main__':
	main()
