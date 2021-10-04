# ToxiChat
Code and data for the EMNLP 2021 paper ["Just Say No: Analyzing the Stance of Neural Dialogue Generation in Offensive Contexts"](https://arxiv.org/pdf/2108.11830.pdf).

## Install dependencies  
`conda env create -f environment.yml`

## Data  

The train, dev, test split of the data are provided in `data/OC_S_post_thread/` folder as `.csv` files. Here is how each csv file is organized.

```
utterance = human or chatbot response
uOff = 0 - safe or 1 - offensive indicator label for each utterance
uOffTarget = a set of target groups if uOff==1
u1stance = stance of current utterance towards 1st utterance (0 - neutral, 1 - agree, -1 - disagree)
u2stance = stance of current utterance towards 2st utterance
u3stance = stance of current utterance towards 3st utterance (note: this column is left blank when 3rd utterance is not available i.e. number of reddit utterances = 2)
resp_coherence = 0 - incoherent or 1 - coherent indicator label for each chatbot response.
```
To load and use the data please use `get_conversation_data_from_OC_S_file(OC_S_file)` function from `OC_S_utils.py` file. 

## Offensive and Stance Classification models

### Single instance Offensive Classification

#### NBOW model
We will train NBOW single sentence classification model initialized with GloVe embedding  
To train NBOW model, you'd need to download and extract [GloVe vectors](https://nlp.stanford.edu/data/glove.6B.zip) into `data/GloVe/` dir and then run `python convert_glove_text_vectors_to_pkl.py` from within the directory
- Training offensive classifier on OC_S_post_thread data  
	`python experiments/train_and_evaluate_NBOW_offensive_classifier.py -g data/GloVe/glove.6B.300d.pkl -td "{'OC_S':'data/OC_S_post_thread/'}" -s saved_models/OC_S_post_thread/NBOW_OC_S_offensive_e30 -o results/OC_S_post_thread/NBOW_OC_S_offensive_e30 -e 30 -dv 1 -t`

#### BERT large cased model
- Training offensive classifier on OC_S_post_thread data  
	`python experiments/train_and_evaluate_BERT_offensive_classifier.py -e 8 -td "{'OC_S':'data/OC_S_post_thread/'}" -s saved_models/OC_S_post_thread/BERT_large_OC_S_offensive_e8 -o results/OC_S_post_thread/BERT_large_OC_S_offensive_e8 -t`

### Full Sequence Offensive Classification (DGPT)  
We will train a DGPT model offensive classifier for the entire comment thread with EOS tokens used for sentence representations.  
- Training offensive classifier on OC_S_post_thread data  
	`python experiments/train_and_evaluate_DGPT_offensive_classifier.py -e 12 -td "{'OC_S':'data/OC_S_post_thread/'}" -s saved_models/OC_S_post_thread/DGPT_medium_OC_S_offensive_e12 -o results/OC_S_post_thread/DGPT_medium_OC_S_offensive_e12 -t`  
- Training offensive classifier on OC_S_post_thread + SBF data  
	`python experiments/train_and_evaluate_DGPT_offensive_classifier.py -e 3 -td "{'OC_S':'data/OC_S_post_thread/', 'SBF':'data/SBF'}" -s saved_models/OC_S_post_thread/DGPT_medium_OC_S_and_SBF_offensive_e3 -o results/OC_S_post_thread/DGPT_medium_OC_S_and_SBF_offensive_e3 -t -dv 4`  


## Stance Classification

### Pairwise Stance Classification

#### NBOW model
We will train NBOW Sentence Pair classification model initialized with GloVe embedding  
- Training Stance classifier on OC_S_post_thread_data (cross entropy)  
	`python experiments/train_and_evaluate_NBOW_pairwise_stance_classifier.py -g data/GloVe/glove.6B.300d.pkl -td "{'OC_S':'data/OC_S_post_thread/'}" -s saved_models/OC_S_post_thread/NBOW_OC_S_pairwise_stance_e30 -o results/OC_S_post_thread/NBOW_OC_S_pairwise_stance_e30 -e 30 -dv 1 -t`  

#### BERT large cased model
We will train Bert Sentence Pair classification model  
- Training Stance classifier on OC_S_post_thread_data (cross entropy)  
	`python experiments/train_and_evaluate_BERT_pairwise_stance_classifier.py -e 8 -td "{'OC_S':'data/OC_S_post_thread/'}" -s saved_models/OC_S_post_thread/BERT_large_OC_S_pairwise_stance_e8 -o results/OC_S_post_thread/BERT_large_OC_S_pairwise_stance_e8 -t`  

### Full Sequence Stance Classification  
We will train a DGPT model stance classifier for the entire comment thread with EOS tokens used for sentence representations.  
- Training Stance classifier on OC_S_post_thread_data (cross entropy)  
	`python experiments/train_and_evaluate_DGPT_stance_classifier.py -e 12 -td "{'OC_S':'data/OC_S_post_thread/'}" -s saved_models/OC_S_post_thread/DGPT_medium_OC_S_stance_e12 -o results/OC_S_post_thread/DGPT_medium_OC_S_stance_e12 -t`  
- Training Stance classifier on OC_S_post_thread_data (Focal Loss)  
	`python experiments/train_and_evaluate_DGPT_stance_classifier.py -e 16 -td "{'OC_S':'data/OC_S_post_thread/'}" -s saved_models/OC_S_post_thread/DGPT_medium_OC_S_stance_e16_focal_lr5e_5 -o results/OC_S_post_thread/DGPT_medium_OC_S_stance_e16_focal_lr5e_5 -foc -lr 5e-5 -t`  

To download pretrained DGPT offensive and Stance (Focal) classifiers use the following [link](https://mega.nz/file/ANhEWDiA#ky-f6HNfmgM4-QVpNv_-z5cN1yf4d0Ml6PAEWHnQVCg)

# Mitigating Offensive language using Controlled Text Generation

## Dataset Preparation
We will first create a dataset of posts and comments from all of the reddit. Then we will create comment trees from these posts and comments and label them with our stance and offensive classifiers

### Downloading the reddit posts and comments dumps
1. Download the reddit comments and submissions dumps from August(08) to October(10), 2019 in the data folder   
	```
	mkdir -p data/reddit_dumps/comments_compressed
	cd data/reddit_dumps/comments_compressed
	wget -nc https://files.pushshift.io/reddit/comments/RC_2019-10.zst
	wget -nc https://files.pushshift.io/reddit/comments/RC_2019-09.zst
	wget -nc https://files.pushshift.io/reddit/comments/RC_2019-08.zst
	wget -nc https://files.pushshift.io/reddit/comments/RC_2019-07.zst
	wget -nc https://files.pushshift.io/reddit/comments/RC_2019-06.zst
	wget -nc https://files.pushshift.io/reddit/comments/RC_2019-05.zst
	cd ..
	mkdir posts_compressed
	cd posts_compressed
	wget -nc https://files.pushshift.io/reddit/submissions/RS_2019-10.zst
	wget -nc https://files.pushshift.io/reddit/submissions/RS_2019-09.zst
	wget -nc https://files.pushshift.io/reddit/submissions/RS_2019-08.zst
	wget -nc https://files.pushshift.io/reddit/submissions/RS_2019-07.zst
	wget -nc https://files.pushshift.io/reddit/submissions/RS_2019-06.zst
	wget -nc https://files.pushshift.io/reddit/submissions/RS_2019-05.zst
	cd ../../
	```
### Create posts and comments sample
- `python extract_reddit_posts.py -f data/reddit_dumps/posts_compressed/RS_2019-10.zst data/reddit_dumps/posts_compressed/RS_2019-09.zst data/reddit_dumps/posts_compressed/RS_2019-08.zst data/reddit_dumps/posts_compressed/RS_2019-07.zst data/reddit_dumps/posts_compressed/RS_2019-06.zst data/reddit_dumps/posts_compressed/RS_2019-05.zst -p 0.8 -o data/reddit_dumps/posts/all_mitigating_sample/`  
- `python extract_reddit_comments_for_posts.py -f data/reddit_dumps/comments_compressed/RC_2019-05.zst data/reddit_dumps/comments_compressed/RC_2019-06.zst data/reddit_dumps/comments_compressed/RC_2019-07.zst data/reddit_dumps/comments_compressed/RC_2019-08.zst data/reddit_dumps/comments_compressed/RC_2019-09.zst data/reddit_dumps/comments_compressed/RC_2019-10.zst -p data/reddit_dumps/posts/all_mitigating_sample/all_subreddit_posts.jsonl -o data/reddit_dumps/comments/all_mitigating_sample/`  

### Create threads from posts and comments sample  
`python create_post_comment_trees_from_all_reddit_sample.py -ip data/reddit_dumps/posts/all_mitigating_sample/all_subreddit_posts.jsonl -ic data/reddit_dumps/comments/all_mitigating_sample/all_subreddit_post_related_comments.jsonl -mc 3 -o data/reddit_dumps/post_comment_threads/all_mitigating_sample/`  

#### Split the post comment threads into 4 splits  
`python split_threads_into_files.py -i data/reddit_dumps/post_comment_threads/all_mitigating_sample/all_reddit_post_and_comments_3_threads.pkl -o data/reddit_dumps/post_comment_threads/all_mitigating_sample/splits/ -n 4`  

#### Predict separately for each split  
- `python predict_DGPT_stance_on_post_comment_trees.py -i data/reddit_dumps/post_comment_threads/all_mitigating_sample/splits/split_0.pkl -sm saved_models/OC_S_post_thread/DGPT_medium_OC_S_stance_e16_focal_lr5e_5 -om saved_models/OC_S_post_thread/DGPT_medium_OC_S_and_SBF_offensive_e2 -o data/reddit_dumps/post_comment_threads/all_mitigating_sample/splits/predictions_both/ -s data/reddit_dumps/post_comment_threads/all_mitigating_sample/splits/predictions_both/split_0_preds.pkl`  
- `python predict_DGPT_stance_on_post_comment_trees.py -i data/reddit_dumps/post_comment_threads/all_mitigating_sample/splits/split_1.pkl -sm saved_models/OC_S_post_thread/DGPT_medium_OC_S_stance_e16_focal_lr5e_5 -om saved_models/OC_S_post_thread/DGPT_medium_OC_S_and_SBF_offensive_e2 -o data/reddit_dumps/post_comment_threads/all_mitigating_sample/splits/predictions_both/ -s data/reddit_dumps/post_comment_threads/all_mitigating_sample/splits/predictions_both/split_1_preds.pkl`  
- `python predict_DGPT_stance_on_post_comment_trees.py -i data/reddit_dumps/post_comment_threads/all_mitigating_sample/splits/split_2.pkl -sm saved_models/OC_S_post_thread/DGPT_medium_OC_S_stance_e16_focal_lr5e_5 -om saved_models/OC_S_post_thread/DGPT_medium_OC_S_and_SBF_offensive_e2 -o data/reddit_dumps/post_comment_threads/all_mitigating_sample/splits/predictions_both/ -s data/reddit_dumps/post_comment_threads/all_mitigating_sample/splits/predictions_both/split_2_preds.pkl`  
- `python predict_DGPT_stance_on_post_comment_trees.py -i data/reddit_dumps/post_comment_threads/all_mitigating_sample/splits/split_3.pkl -sm saved_models/OC_S_post_thread/DGPT_medium_OC_S_stance_e16_focal_lr5e_5 -om saved_models/OC_S_post_thread/DGPT_medium_OC_S_and_SBF_offensive_e2 -o data/reddit_dumps/post_comment_threads/all_mitigating_sample/splits/predictions_both/ -s data/reddit_dumps/post_comment_threads/all_mitigating_sample/splits/predictions_both/split_3_preds.pkl`  

#### Merge predictions  
`python merge_Off_Stance_predictions.py -i data/reddit_dumps/post_comment_threads/all_mitigating_sample/splits/predictions_both/ -n 4 -o data/reddit_dumps/post_comment_threads/all_mitigating_sample/splits/predictions_both/merged_split_predictions.pkl`  

### Create CTG fine-tuning dataset from post_comment threads with stance and offensive labels  
`python get_fine_tuning_subsets_from_label_predicted_convs.py -i data/reddit_dumps/post_comment_threads/all_mitigating_sample/splits/predictions_both/merged_split_predictions.pkl -o data/reddit_dumps/post_comment_threads/CTG_experiments/all_mitigating_sample/final/`  

## Fine-tune DGPT medium model for different CTG experiments 

### DAPT  
CTG using DAPT i.e. simply training on the subset we care about  
#### 1. Off Control [SAFE] subset (DAPT - [S])  
`python experiments/CTG_DGPT_finetuner.py -so [SAFE] -t data/reddit_dumps/post_comment_threads/CTG_experiments/all_mitigating_sample/final/off_control_train.pkl -d data/reddit_dumps/post_comment_threads/CTG_experiments/all_mitigating_sample/final/off_control_dev.pkl -s saved_models/CTG/Off_control_DGPT_safe_subset -o results/CTG/Off_control_DGPT_safe_subset -e 3`  
#### 2. Safe Stance Control [NO-STANCE] subset (DAPT - [S][N])  
`python experiments/CTG_DGPT_finetuner.py -so [NO-STANCE] -t data/reddit_dumps/post_comment_threads/CTG_experiments/all_mitigating_sample/final/safe_stance_control_train.pkl -d data/reddit_dumps/post_comment_threads/CTG_experiments/all_mitigating_sample/final/safe_stance_control_dev.pkl -s saved_models/CTG/safe_stance_control_DGPT_no_stance_subset -o results/CTG/safe_stance_control_DGPT_no_stance_subset -e 3`  

### ATCON  
CTG using control labels  
#### 1. Offensive Label Control (ATCON [S])  
`python experiments/CTG_DGPT_finetuner.py -t data/reddit_dumps/post_comment_threads/CTG_experiments/all_mitigating_sample/final/off_control_train.pkl -d data/reddit_dumps/post_comment_threads/CTG_experiments/all_mitigating_sample/final/off_control_dev.pkl -s saved_models/CTG/Off_control_DGPT -o results/CTG/Off_control_DGPT -e 3 -dv 100`   
#### 2. Stance Label Control (Safe) (ATCON [N])  
`python experiments/CTG_DGPT_finetuner.py -t data/reddit_dumps/post_comment_threads/CTG_experiments/all_mitigating_sample/final/safe_stance_control_train.pkl -d data/reddit_dumps/post_comment_threads/CTG_experiments/all_mitigating_sample/final/safe_stance_control_dev.pkl -s saved_models/CTG/safe_stance_control_DGPT -o results/CTG/safe_stance_control_DGPT -e 3`  
#### 3. Both Offensive and Stance Label Control (both) (ATCON [S][N])  
`python experiments/CTG_DGPT_finetuner.py -t data/reddit_dumps/post_comment_threads/CTG_experiments/all_mitigating_sample/final/both_control_train.pkl -d data/reddit_dumps/post_comment_threads/CTG_experiments/all_mitigating_sample/final/both_control_dev.pkl -s saved_models/CTG/both_control_DGPT -o results/CTG/both_control_DGPT -e 3`  

## Generate Responses on test set using CTG models  
Control labels [OFF]/[SAFE] and [AGREE]/[NO-STANCE]  

- Baseline No Control  
	`python generate_CTG_responses_and_make_off_and_stance_predictions.py -m microsoft/DialoGPT-medium -d data/reddit_dumps/post_comment_threads/CTG_experiments/all_mitigating_sample/final/test_threads.pkl -sm saved_models/OC_S_post_thread/DGPT_medium_OC_S_stance_e16_focal_lr5e_5 -om saved_models/OC_S_post_thread/DGPT_medium_OC_S_and_SBF_offensive_e3 -n 1 -bs 10 -o results/CTG/DGPT/test_threads_replies_and_off_stance_preds.pkl`  
- DAPT Offensive Control Safe Subset (DAPT - [S])  
	`python generate_CTG_responses_and_make_off_and_stance_predictions.py -m saved_models/CTG/Off_control_DGPT_safe_subset -d data/reddit_dumps/post_comment_threads/CTG_experiments/all_mitigating_sample/final/test_threads.pkl -sm saved_models/OC_S_post_thread/DGPT_medium_OC_S_stance_e16_focal_lr5e_5 -om saved_models/OC_S_post_thread/DGPT_medium_OC_S_and_SBF_offensive_e2 -n 1 -bs 10 -o results/CTG/Off_control_DGPT/DAPT_Off_control_safe_subset_test_threads_replies_and_off_stance_preds.pkl`  
- DAPT Safe Stance Control No-Stance Subset (DAPT - [S][N])  
	`python generate_CTG_responses_and_make_off_and_stance_predictions.py -m saved_models/CTG/safe_stance_control_DGPT_no_stance_subset -d data/reddit_dumps/post_comment_threads/CTG_experiments/all_mitigating_sample/final/test_threads.pkl -sm saved_models/OC_S_post_thread/DGPT_medium_OC_S_stance_e16_focal_lr5e_5 -om saved_models/OC_S_post_thread/DGPT_medium_OC_S_and_SBF_offensive_e2 -n 1 -bs 10 -o results/CTG/safe_stance_control_DGPT/DAPT_safe_stance_control_no_stance_subset_test_threads_replies_and_off_stance_preds.pkl`  
- Offensive Control (ATCON - [S])    
	`python generate_CTG_responses_and_make_off_and_stance_predictions.py -m saved_models/CTG/Off_control_DGPT -p [SAFE] -d data/reddit_dumps/post_comment_threads/CTG_experiments/all_mitigating_sample/final/test_threads.pkl -sm saved_models/OC_S_post_thread/DGPT_medium_OC_S_stance_e16_focal_lr5e_5 -om saved_models/OC_S_post_thread/DGPT_medium_OC_S_and_SBF_offensive_e2 -n 1 -bs 10 -o results/CTG/Off_control_DGPT/Off_control_test_threads_safe_replies_and_off_stance_preds.pkl`
- Stance Control (Safe) (ATCON - [N])    
	`python generate_CTG_responses_and_make_off_and_stance_predictions.py -m saved_models/CTG/safe_stance_control_DGPT -p [NO-STANCE] -d data/reddit_dumps/post_comment_threads/CTG_experiments/all_mitigating_sample/final/test_threads.pkl -sm saved_models/OC_S_post_thread/DGPT_medium_OC_S_stance_e16_focal_lr5e_5 -om saved_models/OC_S_post_thread/DGPT_medium_OC_S_and_SBF_offensive_e2 -n 1 -bs 10 -o results/CTG/safe_stance_control_DGPT/safe_stance_control_test_threads_no_stance_replies_and_off_stance_preds.pkl`  
- Both Control (ATCON - [S][N])    
	`python generate_CTG_responses_and_make_off_and_stance_predictions.py -m saved_models/CTG/both_control_DGPT -p [SAFE][NO-STANCE] -d data/reddit_dumps/post_comment_threads/CTG_experiments/all_mitigating_sample/final/test_threads.pkl -sm saved_models/OC_S_post_thread/DGPT_medium_OC_S_stance_e16_focal_lr5e_5 -om saved_models/OC_S_post_thread/DGPT_medium_OC_S_and_SBF_offensive_e2 -n 1 -bs 10 -o results/CTG/both_control_DGPT/both_control_test_threads_safe_no_stance_replies_and_off_stance_preds.pkl`  


## Automatic evalaution of CTG test predictions  
`python automatic_evaluation_of_CTG_test_predictions.py -mg "[('DGPT medium baseline', 'results/CTG/DGPT/test_threads_replies_and_off_stance_preds.pkl'), ('ATCON - [S]', 'results/CTG/Off_control_DGPT/Off_control_test_threads_safe_replies_and_off_stance_preds.pkl'), ('ATCON [N]', 'results/CTG/safe_stance_control_DGPT/safe_stance_control_test_threads_no_stance_replies_and_off_stance_preds.pkl'), ('ATCON [N][S]', 'results/CTG/both_control_DGPT/both_control_test_threads_safe_no_stance_replies_and_off_stance_preds.pkl'), ('DAPT [S]', 'results/CTG/Off_control_DGPT/DAPT_Off_control_safe_subset_test_threads_replies_and_off_stance_preds.pkl'), ('DAPT [S][N]', 'results/CTG/safe_stance_control_DGPT/DAPT_safe_stance_control_no_stance_subset_test_threads_replies_and_off_stance_preds.pkl')]"  -o results/CTG/auto_eval/`  

# Citation
```
@article{baheti2021just,
  title={Just Say No: Analyzing the Stance of Neural Dialogue Generation in Offensive Contexts},
  author={Baheti, Ashutosh and Sap, Maarten and Ritter, Alan and Riedl, Mark},
  year={2021}
  booktitle={EMNLP},
}
```