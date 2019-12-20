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

def basic_classifier():
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

    tqdm.pandas(desc="Removing stopwords...")
    clean_tweets['tokens'] = clean_tweets['tokens'].progress_apply(remove_stopwords)

    tqdm.pandas(desc="Stemming and lemmatizing...")
    clean_tweets['tokens'] = clean_tweets['tokens'].progress_apply(stem_lem)

    text_vector = clean_tweets['tokens'].tolist()

    # Embedding

    clean_tweets['vector'] = tfidf(text_vector).tolist()

    train_vectors = clean_tweets.query("tag == 'train'")['vector'].tolist()
    train_labels = train_labels_levelb.values.tolist()
    test_vectors = clean_tweets.query("tag == 'test'")['vector'].tolist()
    test_labels = test_labels_levelb.values.tolist()

    # Classifing

    return classify(train_vectors, train_labels, test_vectors, test_labels, pkl_classifier=r'basic_classifier.pkl')

if __name__ == "__main__":
    basic_classifier()