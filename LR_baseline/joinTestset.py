# -*- coding: utf-8 -*-

import pandas as pd
import os

if os.path.exists(r'./data/join_testset_levelb.csv'):
    os.remove(r'./data/join_testset_levelb.csv')
    print("The former joined file is deleted. Now generate a new one.")
data = pd.read_csv(r'./data/testset-levelb.tsv', delimiter='\t', header=0)
labels = pd.read_csv(r'./data/labels-levelb.csv', header=None, names=['id', 'subtask_b'])
data = data.join(labels.set_index('id'), on='id')

data.to_csv(r'./data/join_testset_levelb.csv')