# Hate Speech Detection for COSC586

This repository is for the research program Hate Speech Detection assigned by course COSC586:Text Mining from Georgetown University. 

## Desciption

[SemEval-2019 Task 6](https://arxiv.org/pdf/1903.08983.pdf) had defined three sub-tasks for hate speech detection:

- **Sub-task A:** Offensive language identification
- **Sub-task B:** Automatic categorization of offense
- **Sub-task C:** Offense target identification

For this repository, we mainly focus on the sub-task B, namely the automatic categorization of offense. In this task, our goal is to predict the type of offense. Only offensive posts are included. The two categories to be classified in sub-task B are as follow:

- Targeted Insult(TIN): Posts containing an insult/threat to an individual, group, or others
- Untargeted(UNT): Posts containing nontargeted profanity and swearing. Posts with general profanity are not targeted, but they contain non-acceptable language

## Baseline

The most basic baseline that we use for this task is built based on the this [repository](https://github.com/FTS152/NLP-Project-2-Offensive-Tweet-Classification-SemEval-2019-Task6). This baseline makes use of Tfidf vectorization and uses Logistic Regression to classify offensive langauge. As the f1 score for this baseline is only around **0.50**, it's only slightly better than randomly guessing. Also, several problems arise while processing the data:

- Exclude lots of so called noise in the text data, for example "n't", usually stands for negativity. Besides this, the exclusion also involve certain pattern like "@USER", which often indicates targets in sentences. This kind of data cleaning will definitely remove lots of information from the original data.
- Do both stemming and lemmatization to the text data, where different words might have the same stemmed form, and that might cause disambiguity.
- The vectors from TfidfVectorization are too sparse: the length of vector is too long, and most of its elements are zero. This will lead to overfitting on the training data, also hurt the performance of the model.

Overall the baseline performs very bad. The Tfidf method only captures the shallow features, and neglects most of the semantic and syntactic features. This kind of shortcoming leads to the classfier focusing only on lexical features.