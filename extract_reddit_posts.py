# We will extract the posts from all subreddits.

from utils import RANDOM_SEED, log_dict, log_list, print_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, save_in_json, load_from_json, \
					format_time, get_number_of_lines, write_list_to_file, save_list_of_tuples_to_tsv, get_ngrams_from_sentence, get_ngram_freq_from_corpus, normalize_vocab, get_num_of_word_in_corpus, replace_urls, remove_markdown_urls

import os
import re
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
parser.add_argument("-f", "--files", help="List of reddit submissions (posts) dump files", type=str, nargs='+')
parser.add_argument("-o", "--output_dir", help="Directory where the results of this program will be saved", type=str, required=True)
parser.add_argument("-p", "--drop_prob", help="Probability for dropping the current post. ", type=float, default=0.98)
args = parser.parse_args()

import logging
# Ref: https://stackoverflow.com/a/49202811/4535284
for handler in logging.root.handlers[:]:
	logging.root.removeHandler(handler)

make_dir_if_not_exists(args.output_dir)
logfile = os.path.join(args.output_dir, "output.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(logfile, mode='w'), logging.StreamHandler()])



# We will store all the posts in a subreddit in a dictionary of dictionaries
# Upper level keys will be the subreddit names
# Each lower level key will be the post id and the values will be details about the post
save_file_writer = open(os.path.join(args.output_dir, f"all_subreddit_posts.jsonl"), "w")
# We will also save post counts by type
post_type_counts = {"[deleted]":0 ,"[removed]":0, "bot_post":0, "url_post":0, "video_post":0, "image_post":0, "text_post":0, "small_text_post":0, "title_only_post":0, "cross_post":0}
total_saved_posts = 0

DROP_PROB = args.drop_prob
def save_post_info_in_file(writer, title, post, post_type, id, score, author, created_utc, url, content_url):
	# Save each post in a single line if possible
	# Prepare save dict
	post_info = {"id":id,
					"title":title,
					"post":post,
					"post_type":post_type,
					"score":score,
					"author":author,
					"created_utc":created_utc,
					"url":url,
					"content_url":url}
	post_info_string = json.dumps(post_info)
	writer.write(f"{post_info_string}\n")

def read_reddit_post_dump_and_save_subreddit_posts(dump_file):
	global save_file_writer, total_saved_posts, post_type_counts
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
				post_object = json.loads(line)
				# Post object contains the following keys = dict_keys(['all_awardings', 'allow_live_comments', 'archived', 'author', 'author_created_utc', 'author_flair_background_color', 'author_flair_css_class', 'author_flair_richtext', 'author_flair_template_id', 'author_flair_text', 'author_flair_text_color', 'author_flair_type', 'author_fullname', 'author_patreon_flair', 'author_premium', 'awarders', 'can_gild', 'can_mod_post', 'category', 'content_categories', 'contest_mode', 'created_utc', 'discussion_type', 'distinguished', 'domain', 'edited', 'event_end', 'event_is_live', 'event_start', 'gilded', 'gildings', 'hidden', 'id', 'is_crosspostable', 'is_meta', 'is_original_content', 'is_reddit_media_domain', 'is_robot_indexable', 'is_self', 'is_video', 'link_flair_background_color', 'link_flair_css_class', 'link_flair_richtext', 'link_flair_text', 'link_flair_text_color', 'link_flair_type', 'locked', 'media', 'media_embed', 'media_only', 'no_follow', 'num_comments', 'num_crossposts', 'over_18', 'parent_whitelist_status', 'permalink', 'pinned', 'post_hint', 'preview', 'pwls', 'quarantine', 'removal_reason', 'removed_by', 'removed_by_category', 'retrieved_on', 'score', 'secure_media', 'secure_media_embed', 'selftext', 'send_replies', 'spoiler', 'stickied', 'subreddit', 'subreddit_id', 'subreddit_name_prefixed', 'subreddit_subscribers', 'subreddit_type', 'suggested_sort', 'thumbnail', 'thumbnail_height', 'thumbnail_width', 'title', 'total_awards_received', 'url', 'whitelist_status', 'wls'])
				# Extract the subreddit, post, id, author, score
				subreddit = post_object["subreddit"]
				score = post_object["score"]
				# NOTE: Adding a threshold on score to limit the data
				# TEMP: Removing this criteria
				# if score <= 1:
				# 	continue
				num_comments = post_object["num_comments"]
				if num_comments == 0:
					# Ignore this post as no reply present
					# logging.info("Removing this post with 0 comments!!!")
					continue
				title = post_object["title"]
				post = post_object["selftext"]
				id = post_object["id"]
				author = post_object["author"]
				retrieved_on = post_object["retrieved_on"]
				url = post_object['permalink']
				content_url = post_object['url']
				created_utc = post_object['created_utc']
				
				# Save post information in global files
				print_post_flag = False
				if post in ["[deleted]","[removed]"]:
					# ignore/remove this post from the dataset
					post_type_counts[post] += 1
					continue
				if "I am a bot" in post:
					# ignore/remove this post from the dataset
					post_type_counts["bot_post"] += 1
					continue
				if not post:
					# This is a video, image, url or reddit crosspost
					if content_url.startswith("https://v.redd.it"):
						post_type_counts["video_post"] += 1
						# ignore/remove these posts from the dataset
						continue
					if content_url.startswith("https://i.redd.it"):
						post_type_counts["image_post"] += 1
						post_type = "image_post"
						# ignore/remove these posts from the dataset
						continue
					if content_url.startswith("https://www.reddit.com/r/"):
						# This could be the link to self or cross-post
						if url in content_url:
							post_type_counts["title_only_post"] += 1
							post_type = "title_only_post"
							# print_post_flag = True
						else:
							post_type_counts["cross_post"] += 1
							# ignore/remove these posts from the dataset
							continue
					else:
						post_type_counts["url_post"] += 1
						post_type = "url_post"
						# TEMP: Keep ignore/remove these posts from the dataset
						# continue
				else:
					# This would be a post that contains text
					post_type_counts["text_post"] += 1
					# We want to find the posts that are relatively small
					SMALL_POST_TOKEN_THRESHOLD = 70
					if len(post.split()) <= SMALL_POST_TOKEN_THRESHOLD:
						# The post may contain URLs in it
						# Convert such urls to special URL token
						url_free_post, number_of_urls = replace_urls(post)
						# if number_of_urls > 0:
						# 	logging.info(f"post = {post}")
						# 	logging.info(f"url free post = {url_free_post}")
						# 	logging.info(f"nURLS = {number_of_urls}")


						if "[" in url_free_post:
							url_free_post, n_links = remove_markdown_urls(url_free_post)
							# if n_links > 0:
							# 	logging.info(f"post = {post}")
							# 	logging.info(f"clean_post = {url_free_post}")
							# 	logging.info(f"nLinks = {n_links}")
							# 	if post_type_counts["small_text_post"] >= 3:
							# 		exit()
						post = url_free_post
						# Check if the post contains only url and not much text
						post_type_counts["small_text_post"] += 1
						post_type = "small_text_post"
					else:
						#TEMP: For now we are removing longer posts
						continue

				if random.uniform(0,1) < DROP_PROB:
					# Randomly drop 90% of the posts
					continue

				if print_post_flag:
					logging.info(f"{id} post by {author} of title = {title}")
					logging.info(f"post = {post}")
					logging.info(f"num_comments = {num_comments}")
					logging.info(url)
					logging.info(content_url)
					logging.info("")
				# if total_saved_posts == 100:
				# 	logging.info(f"Total saved posts = {total_saved_posts}")
				# 	logging.info(post_type_counts)
				# 	exit()
				save_post_info_in_file(save_file_writer, title, post, post_type, id, score, author, created_utc, url, content_url)
				total_saved_posts += 1
			previous_line = lines[-1]
			if chunk_index % 100 == 0:
				logging.info(f"Chunk number: {chunk_index}. Total posts:{total_saved_posts}. Post distribution = {post_type_counts}")
				save_file_writer.flush()
				# log_dict(all_subreddit_posts_files, K=20)

def main():
	for file in args.files:
		logging.info(f"Reading posts from file: {file}")
		read_reddit_post_dump_and_save_subreddit_posts(file)

	# Close all open files
	save_file_writer.close()

if __name__ == '__main__':
	main()

