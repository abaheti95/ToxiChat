import os
import pickle
import numpy as np

def save_in_pickle(save_object, save_file):
	with open(save_file, "wb") as pickle_out:
		pickle.dump(save_object, pickle_out)

def read_glove_vectors(glove_file):
	glove_dict = dict()
	with open(glove_file, "r") as reader:
		for line in reader:
			line_spl = line.strip().split()
			word = line_spl[0]
			embedding = [float(e) for e in line_spl[1:]]
			assert len(embedding) == 300
			glove_dict[word] = np.array(embedding)
	return glove_dict

glove_file = "glove.6B.300d.txt"
print(f"Reading glove vectors from {glove_file}")
glove_dict = read_glove_vectors(glove_file)
glove_save_file = "glove.6B.300d.pkl"
print(f"Saving glove numpy vectors dict in {glove_save_file}")
save_in_pickle(glove_dict, glove_save_file)