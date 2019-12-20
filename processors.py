# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import os
import re

import nltk
# nltk.download('punkt', 'stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer()
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import spacy
spacier = spacy.load("en_core_web_lg", disable=["parser"])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.metrics import precision_score, recall_score

def oversampling(train_data):
    print(r"Oversampling UNT data...")
    length = train_data.shape[0]
    for i in range(length):
        if train_data.loc[i]["subtask_b"] == "UNT":
            seed = np.random.random_integers(0, 1)
            if seed == 1:
                train_data.loc[train_data.shape[0] + 1] = train_data.loc[i]
    return train_data.sample(frac=1).reset_index(drop=True)

def noise_cancelation(tweet):
    noises = ['URL', '@USER', '\'ve', 'n\'t', '\'s', '\'m']
    for noise in noises:
        tweet = tweet.replace(noise, '')
    return re.sub(r'[^a-zA-Z]', ' ', tweet)

def noise_replacement(tweet):
    noises = ['URL', '\'ve', 'n\'t', '\'s', '\'m']
    for noise in noises:
        tweet = tweet.replace(noise, '')
    tweet = re.sub(r"(@USER[\s]+)*@USER", 'user', tweet)
    return re.sub(r'[^a-zA-Z]', ' ', tweet)

def compression(tweet):
    return re.sub(r"[\s]+", ' ', tweet)

def tokenize(tweet):
    lower_tweet = tweet.lower()
    return word_tokenize(lower_tweet)

def tweet_tokenize(tweet):
    lower_tweet = tweet.lower()
    return tweet_tokenizer.tokenize(lower_tweet)

def remove_stopwords(tokens):
    clean_tokens = list()
    stopword_set = set(stopwords.words('english'))
    for token in tokens:
        if token not in stopword_set:
            token = token.strip()
            if token != '':
                clean_tokens.append(token)
    return clean_tokens

def lemma(tokens):
    clean_tokens = list()
    for token in tokens:
        token = wordnet_lemmatizer.lemmatize(token)
        clean_tokens.append(token)
    return clean_tokens

def stem_lem(tokens):
    clean_tokens = list()
    for token in tokens:
        token = wordnet_lemmatizer.lemmatize(token)
        token = lancaster_stemmer.stem(token)
        clean_tokens.append(token)
    return clean_tokens

def partofspeech(tokens):
    tags = nltk.pos_tag(tokens)
    tag_list = [entry[1] for entry in tags]
    return tag_list

def spacy_partofspeech(tokens):
    return [spacier(token)[0].pos_ for token in tokens]

def spacy_nameentity(tokens):
    return [spacier(token)[0].ent_type_ for token in tokens]

def tfidf(text_vector, indicator='text'):
    if indicator == 'text':
        print("Start vectorizing vectors...")
    elif indicator == 'pos':
        print("Start vectorizing part-of-speech vectors...")
    vectorizer = TfidfVectorizer()
    joint_data = [' '.join(tweet) for tweet in tqdm(text_vector, "Vectorizing...")]
    vectorizer = vectorizer.fit(joint_data)
    vectors = vectorizer.transform(joint_data).toarray()
    return vectors

def classify(train_vectors, train_labels, test_vectors, test_labels, type='LR', penalty='l2', pkl_classifier=r'pos_classifier.pkl'):
    pickled_classifier = os.path.join(r'ml', pkl_classifier)
    if not os.path.exists(pickled_classifier):
        print("Start training model...")
        if type == 'LR':
            if penalty == 'l2':
                classifier = LogisticRegression(multi_class='auto', solver='newton-cg')
                classifier = GridSearchCV(classifier, {"C":np.logspace(-3, 3, 7), "penalty":["l2"]}, cv=3, n_jobs=-1, scoring='f1_macro')
                classifier.fit(train_vectors, train_labels)
                classifier = classifier.best_estimator_
            elif penalty == 'l1':
                classifier = LogisticRegression(multi_class='auto', solver='liblinear')
                classifier = GridSearchCV(classifier, {"C":np.logspace(-3, 3, 7), "penalty":["l1"]}, cv=3, n_jobs=-1, scoring='f1_macro')
                classifier.fit(train_vectors, train_labels)
                classifier = classifier.best_estimator_
            else:
                classifier = LogisticRegression(multi_class='auto', solver='newton-cg')
                classifier = GridSearchCV(classifier, {"penalty":["none"]}, cv=3, n_jobs=-1, scoring='f1_macro')
                classifier.fit(train_vectors, train_labels)
                classifier = classifier.best_estimator_
        elif type == 'SVM':
            classifier = SVC(gamma='auto')
            classifier = GridSearchCV(classifier, {'C':[0.001, 0.01, 0.1, 1, 10, 100]}, cv=3, n_jobs=-1, scoring='f1_macro')
            classifier.fit(train_vectors, train_labels)
            classifier = classifier.best_estimator_
        elif type == 'RF':
            classifier = RandomForestClassifier(max_depth=300, min_samples_split=2, n_jobs=-1)
            params = {'n_estimators': [n for n in range(150,300,50)], 'criterion':['gini','entropy'], }
            classifier = GridSearchCV(classifier, params, cv=3, n_jobs=-1, scoring='f1_macro')
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
    print("Training f1 macro score: {}.".format(score))
    test_predictions = classifier.predict(test_vectors)
    score = f1_score(test_labels, test_predictions, average='macro')
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_predictions))
    print("Testing f1 macro score:", score)
    f1 = score
    print("Weighted precision:")
    p_score = precision_score(test_labels, test_predictions, average='weighted')
    print(p_score)
    print("Weighted recall:")
    r_score = recall_score(test_labels, test_predictions, average='weighted')
    print(r_score)
    print(test_predictions)
    return test_predictions

if __name__ == "__main__":
    pass