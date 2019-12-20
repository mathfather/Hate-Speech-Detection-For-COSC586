# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)

import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
import os
import sys
sys.path.append(r"./")
from processors import *

# Loading data
def pos_classifier():
    print("Start loading dataset..")
    train_data = pd.read_csv(r'./data/olid-training-v1.0.tsv', delimiter='\t', header=0)
    test_data = pd.read_csv(r'./data/join_testset_levelb.csv', header=0)

    train_tweets_levelb = train_data.query("subtask_a == 'OFF'")[["tweet"]]
    train_labels_levelb = train_data.query("subtask_a == 'OFF'")[["subtask_b"]]
    train_tweets_levelb = train_tweets_levelb.assign(tag='train')
    test_tweets_levelb = test_data[["tweet"]]
    test_labels_levelb = test_data[["subtask_b"]]
    test_tweets_levelb = test_tweets_levelb.assign(tag='test')

    tweets = train_tweets_levelb.append(test_tweets_levelb)
    clean_tweets = copy.deepcopy(tweets)

    # Preprocessing data

    tqdm.pandas(desc="Removing noises...")
    clean_tweets['tweet'] = tweets['tweet'].progress_apply(noise_cancelation)

    tqdm.pandas(desc="Tokenizing data...")
    clean_tweets['tokens'] = clean_tweets['tweet'].progress_apply(tokenize)

    tqdm.pandas(desc="Tagging data...")
    clean_tweets['tags'] = clean_tweets['tokens'].progress_apply(partofspeech)

    tqdm.pandas(desc="lemmatizing...")
    clean_tweets['tokens'] = clean_tweets['tokens'].progress_apply(lemma)

    text_vector = clean_tweets['tokens'].tolist()
    pos_vector = clean_tweets['tags'].tolist()

    # Embedding

    vectors = tfidf(text_vector)
    p_vectors = tfidf(pos_vector, indicator='pos')
    combine_vectors = np.concatenate([vectors, p_vectors], axis=1)

    clean_tweets['vector'] = vectors.tolist()
    clean_tweets['pos_vector'] = p_vectors.tolist()
    clean_tweets['mixing'] = combine_vectors.tolist()

    train_vectors = clean_tweets.query("tag == 'train'")['mixing'].tolist()
    train_labels = train_labels_levelb.values.tolist()
    test_vectors = clean_tweets.query("tag == 'test'")['mixing'].tolist()
    test_labels = test_labels_levelb.values.tolist()

    # Classifing

    return classify(train_vectors, train_labels, test_vectors, test_labels)

if __name__ == "__main__":
    pos_classifier()