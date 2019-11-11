# -*- coding: utf-8 -*-

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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV


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

def remove_stopwords(tokens):
    clean_tokens = list()
    stopword_set = set(stopwords.words('english'))
    for token in tokens:
        if token not in stopword_set:
            token = token.strip()
            if token != '':
                clean_tokens.append(token)
    return clean_tokens

def stem_lem(tokens):
    clean_tokens = list()
    for token in tokens:
        token = wordnet_lemmatizer.lemmatize(token)
        token = lancaster_stemmer.stem(token)
        clean_tokens.append(token)
    return clean_tokens

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

def tfidf(text_vector):
    vectorizer = TfidfVectorizer()
    joint_data = [' '.join(tweet) for tweet in tqdm(text_vector, "Vectorizing...")]
    vectorizer = vectorizer.fit(joint_data)
    vectors = vectorizer.transform(joint_data).toarray()
    return vectors

clean_tweets['vector'] = tfidf(text_vector).tolist()

train_vectors = clean_tweets.query("tag == 'train'")['vector'].tolist()
train_labels = train_labels_levelb.values.tolist()
test_vectors = clean_tweets.query("tag == 'test'")['vector'].tolist()
test_labels = test_labels_levelb.values.tolist()

# Classifing

def classify(train_vectors, train_labels, test_vectors, test_labels):
    classifier = LogisticRegression(multi_class='auto', solver='newton-cg')
    classifier = GridSearchCV(classifier, {"C":np.logspace(-3, 3, 7), "penalty":["l2"]}, cv=3, n_jobs=-1, scoring='f1_macro')
    classifier.fit(train_vectors, train_labels)
    classifier = classifier.best_estimator_
    accuracy = f1_score(train_labels, classifier.predict(train_vectors), average='macro')
    print("Training f1 score: {}.".format(accuracy))
    test_predictions = classifier.predict(test_vectors)
    accuracy = f1_score(test_labels, test_predictions, average='macro')
    print("Testing f1 score:", accuracy)
    print("Confusion Matrix:", )
    print(confusion_matrix(test_labels, test_predictions))

print("Start training model...")
classify(train_vectors, train_labels, test_vectors, test_labels)