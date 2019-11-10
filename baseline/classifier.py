import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from tqdm import tqdm
import copy

import re
import nltk
nltk.download('punkt', 'stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Loading data

print("Start loading dataset..")
train_data = pd.read_csv(r'./data/olid-training-v1.0.tsv', delimiter='\t', header=0)
test_data = pd.read_csv(r'./data/join_testset_levelb.csv', header=0)

train_tweets_levelb = train_data.query("subtask_a == 'OFF'")[["tweet"]]
train_labels_levelb = train_data.query("subtask_a == 'OFF'")[["subtask_b"]]
train_tweets_levelb = train_tweets_levelb.assign(tag='train')

train_size = train_tweets_levelb.shape[0]

test_tweets_levelb = test_data[["tweet"]]
test_labels_levelb = test_data[["subtask_b"]]
test_tweets_levelb = test_tweets_levelb.assign(tag='test')

tweets = train_tweets_levelb.append(test_tweets_levelb)
clean_tweets = copy.deepcopy(tweets)

# Preprocessing data

def noise_cancelation(tweet):
    noises = ['URL', '@USER', '\'ve', 'n\'t', '\'s', '\'m']
    for noise in noises:
        tweet = tweet.replace(noise, '')
    return re.sub(r'[^a-zA-Z]', ' ', tweet)

def tokenize(tweet):
    lower_tweet = tweet.lower()
    return word_tokenize(lower_tweet)

