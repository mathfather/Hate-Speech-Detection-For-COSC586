# Hate Speech Detection for COSC586

This repository is for the research program Hate Speech Detection assigned by course COSC586:Text Mining from Georgetown University. 

## Desciption

[SemEval-2019 Task 6](https://arxiv.org/pdf/1903.08983.pdf) had defined three sub-tasks for hate speech detection:

- **Sub-task A:** Offensive language identification
- **Sub-task B:** Automatic categorization of offense
- **Sub-task C:** Offense target identification

For this repository, we mainly focus on the sub-task B, namely the automatic categorization of offense. In this task, our goal is to predict the type of offense. Only offensive posts are included in this task. The two categories in sub-task B are as follow:

- Targeted Insult(TIN): Posts containing an insult/threat to an individual, group, or others
- Untargeted(UNT):Posts containing nontargeted profanity and swearing. Posts with general profanity are not targeted, but they contain non-acceptable language

## Baseline

The most basic baseline that we use for this task is built based on the this [repository](https://github.com/FTS152/NLP-Project-2-Offensive-Tweet-Classification-SemEval-2019-Task6). This baseline make use of Tfidf vectorization and use Logistic Regression to classify offensive langauge. As the f1 score for this baseline is only around **0.50**, it's only slightly better than randomly guessing. Also, several problems arise while processing the data:

- Exclude lots of so called noise in the text data, for example "n't", usually stands for negativity. Besides this, the exclusion also involve certain pattern like "@USER", which often indicates targets in sentences. This kind of data cleaning will definitely remove lots of information from the original data.
- Do both stemming and lemmatization to the text data, where different word might have the same stemming form, and that might cause disambiguity.
- The vectors from TfidfVectorization are too sparse: the length of vector is too long, and most of its elements are zero. This will also hurt the performance of the model.