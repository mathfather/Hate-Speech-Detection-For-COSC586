# Hate Speech Detection for COSC586

This repository is for the research program Hate Speech Detection assigned by course [**COSC586:Text Mining**](https://myaccess.georgetown.edu/pls/bninbp/bwckctlg.p_disp_course_detail?cat_term_in=201730&subj_code_in=COSC&crse_numb_in=586) from Georgetown University. Detailed report for this program(from October 2019 to December 2019) can be found [here](https://github.com/mathfather/Hate-Speech-Detection-For-COSC586/blob/master/A%20light-weight%20model%20for%20target%20detection%20in%20offensive%20language.pdf).

## Desciption

[SemEval-2019 Task 6](https://arxiv.org/pdf/1903.08983.pdf) had defined three sub-tasks for hate speech detection:

- **Sub-task A:** Offensive language identification
- **Sub-task B:** Automatic categorization of offense
- **Sub-task C:** Offense target identification

For this repository, we mainly focus on the sub-task B, namely the automatic categorization of offense. In this task, our goal is to predict the type of offense. Only offensive posts are included. The two categories to be classified in sub-task B are as follow:

- Targeted Insult(TIN): Posts containing an insult/threat to an individual, group, or others
- Untargeted(UNT): Posts containing nontargeted profanity and swearing. Posts with general profanity are not targeted, but they contain non-acceptable language

## Machine learning models

Differences between our models mainly lie on the feature constructing process. As we have tried different classifiers, we found out that logistic regression is the most efficient one among all others. So for the rest of the experiments, we will continuely use logistic regression as our classifier.

### 1. Tfidf baseline

We have referred to this [repository](https://github.com/FTS152/NLP-Project-2-Offensive-Tweet-Classification-SemEval-2019-Task6) while building the basic baseline. This baseline makes use of Tfidf vectorization and uses Logistic Regression to classify offensive langauge. As the f1 score for this baseline is only around **0.499**, it's only slightly better than randomly guessing. Also, several problems arise while processing the data:

- Exclude lots of so called noise in the text data, for example "n't", usually stands for negativity. Besides this, the exclusion also involve certain pattern like "@USER", which often indicates targets in sentences. This kind of data cleaning will definitely remove lots of information from the original data
- Do both stemming and lemmatization to the text data, where different words might have the same stemmed form, and that might cause disambiguity
- The vectors from TfidfVectorization are too sparse: the length of vector is too long, and most of its elements are zero. This will lead to overfitting on the training data, also hurt the performance of the model

Overall the baseline performs very bad. The Tfidf method only captures the shallow features, and neglects most of the semantic and syntactic features. This kind of shortcoming leads to the classfier focusing only on lexical features. Interestingly, using n-gram features in this baseline will even hurt the performance for about 0.03.

### 2. Part-of-speech model

This model is the first model with POS features among all the models that we have tried. It's built based on our first baseline, but with additional part-of-speech tagging features. Most part of this baseline is similar to the first baseline, except that it abandons the use of stopwords as well as stemming, and then makes use of part-of-speech tagging.
Ignoring stopwords can boost the performance of our tfidf baseline for around 0.05, using part-of-speech tagging can boost the performance at around 0.02, and excluding stemming can further boost the performance at around 0.08. The f1 score for our part-of-speech baseline is about **0.651**. Note that this result is already better than the performance of the official SVM baseline given by [SemEval-2019 Task 6](https://arxiv.org/pdf/1903.08983.pdf). Also, using n-gram for part-of-speech tag features in this baseline will even hurt the performance for about 0.03.

### 3. Advanced models with different pre-processing modules 

We have tried different advanced models with different pre-processing modules. For the first step of data cleaning, awaring the problem of previously excluding too much so called noise, we tried another way of data cleaning besides noise_cancelation called noise_replacement. While noise_cancelation directly deletes all the usernames mentioned in text, we reduce duplicate usernames into a single one. The intuition behind is that even though usernames appear in large part of the data, they are still unique in some circumstances and thus shall not be deleted directly. For the tokenizing part, we have tried the normal tokenizer and the tokenizer designed for tweets. Both of the tokenizers are built-in tools within NLTK. For the part-of-speech module, we have tried the spaCy tagger and the NLTK tagger. For the name entity tagger, we choose to use one built by spaCy. A detailed description of pre-processing pipelines for different models as well as their performance is shown below. 

|name|regularization|data cleaning|tokenize|part-of-speech|name entity recognition|lemmatization|Macro F1|
|:--:|:------------:|:-----------:|:------:|:------------:|:---------------------:|:-----------:|:------:|
|Tfidf baseline|l2|cancelation + stop-words|normal| - | - |stemming + lemmatization|0.499|
|POS model|l2|cancelation|normal|NLTK| - |lemmatization|0.651|
|advanced model(1)| - |replacement|tweet|spaCy| - |lemmatization|0.581|
|advanced model(2)|l2|cancelation|normal|NLTK|spaCy|lemmatization|0.657|
|advanced model(3)|l2|replacement|tweet|spaCy|spaCy|lemmatization|**0.679**|
|Official CNN|-|-|-|-|-|-|0.690|
|Official BiLSTM|-|-|-|-|-|-|0.660|
|Official SVM|-|-|-|-|-|-|0.640|