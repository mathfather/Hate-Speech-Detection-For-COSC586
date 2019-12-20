# -*- coding: utf-8 -*-

import pandas as pd

from basic_classifier import basic_classifier
from pos_classifier import pos_classifier
from lr_classifier import lr_classifier
from replace_classifier import replace_classifier
from middle_classifier import middle_classifier

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score

basic_results = basic_classifier()
pos_results = pos_classifier()
middle_results = middle_classifier()
replace_results = replace_classifier()
lr_results = lr_classifier()

test_data = pd.read_csv(r'./data/join_testset_levelb.csv', header=0)
test_labels_levelb = test_data[["subtask_b"]]
test_labels = test_labels_levelb.values.tolist()

final_prediction = list()

for i in range(len(basic_results)):
    voter = 0
    if basic_results[i] == 'TIN':
        voter += 1
    if pos_results[i] == 'TIN':
        voter += 1
    if middle_results[i] == 'TIN':
        voter += 1
    if replace_results[i] == 'TIN':
        voter += 1
    if lr_results[i] == 'TIN':
        voter += 1
    if voter >= 3:
        final_prediction.append('TIN')
    else:
        final_prediction.append('UNT')

score = f1_score(test_labels, final_prediction, average='macro')
print("Confusion Matrix:")
print(confusion_matrix(test_labels, final_prediction))
print("Testing f1 macro score:", score)
print("Precision:")
p_score = precision_score(test_labels, final_prediction, average='weighted')
print(p_score)
print("Recall:")
r_score = recall_score(test_labels, final_prediction, average='weighted')
print(r_score)