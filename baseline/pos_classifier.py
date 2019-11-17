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
import nltk

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.metrics import accuracy_score

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

def lemma(tokens):
    clean_tokens = list()
    for token in tokens:
        token = wordnet_lemmatizer.lemmatize(token)
        clean_tokens.append(token)
    return clean_tokens

def partofspeech(tokens):
    tags = nltk.pos_tag(tokens)
    tag_list = [entry[1] for entry in tags]
    return tag_list

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

def tfidf(text_vector, indicator='text'):
    if indicator == 'text':
        print("Start vectorizing text vectors...")
    elif indicator == 'pos':
        print("Start vectorizing part-of-speech vectors...")
    vectorizer = TfidfVectorizer()
    joint_data = [' '.join(tweet) for tweet in tqdm(text_vector, "Vectorizing...")]
    vectorizer = vectorizer.fit(joint_data)
    vectors = vectorizer.transform(joint_data).toarray()
    return vectors

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

def classify(train_vectors, train_labels, test_vectors, test_labels, type='LR'):
    pickled_classifier = os.path.join(r'baseline', r'pos_classifier.pkl')
    if not os.path.exists(pickled_classifier):
        print("Start training model...")
        if type == 'LR':
            classifier = LogisticRegression(multi_class='auto', solver='newton-cg')
            classifier = GridSearchCV(classifier, {"C":np.logspace(-3, 3, 7), "penalty":["l2"]}, cv=3, n_jobs=-1, scoring='f1_macro')
            classifier.fit(train_vectors, train_labels)
            classifier = classifier.best_estimator_
        elif type == 'SVM':
            classifier = SVC(gamma='auto')
            classifier = GridSearchCV(classifier, {'C':[0.001, 0.01, 0.1, 1, 10]}, cv=3, n_jobs=-1, scoring='f1_macro')
            classifier.fit(train_vectors, train_labels)
            classifier = classifier.best_estimator_
        try:
            joblib.dump(classifier, pickled_classifier)
        except:
            print("Some part of the model is unpicklable, now interrupt dumping...")
    else:
        print("Loading existing model...")
        classifier = joblib.load(pickled_classifier)
    print("Start predicting...")
    score = f1_score(train_labels, classifier.predict(train_vectors), average='macro')
    print("Training f1 score: {}.".format(score))
    test_predictions = classifier.predict(test_vectors)
    score = f1_score(test_labels, test_predictions, average='macro')
    print("Testing f1 score:", score)
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_predictions))
    print("Overall accuracy:")
    print(accuracy_score(test_labels, test_predictions))

classify(train_vectors, train_labels, test_vectors, test_labels)