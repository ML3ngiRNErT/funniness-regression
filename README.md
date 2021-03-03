# funniness-regression

This repository contains our coursework submission for the course 70016 - Natural Language Processing taught at Imperial College London in the academic year 2020-2021.

| File | Contents|
|------|---------|
|[NLP_CW_1_Using_Gensim_vectors.ipynb](NLP_CW_1_Using_Gensim_vectors.ipynb)| LSTM model trained using different word embeddings from Gensim. To use a different wird embedding to the one in this notebook, please download the embeddings and substitute the file path in the appropriate line with the correct path to the downloaded embeddings. |
| [NLP_CW_1_BertForSequenceClassification.ipynb](NLP_CW_1_BertForSequenceClassification.ipynb) | This notebook tunes a pre-trained BERT classifier to our regression task. |
| [NLP_CW_1_RoBertaForSequenceClassification.ipynb](NLP_CW_1_RoBertaForSequenceClassification.ipynb) | This notebook fine-tunes a pre-trained RoBERTa Sequence Classification model to our regression task.|
| [Comparison_of_the_three_models.ipynb](Comparison_of_the_three_models.ipynb) | This notebook compares the predictions of the three models on the [dev.csv](data\dev.csv) dataset. __PLEASE NOTE__: This notebook should only be run after running the first three notebooks to get the dev predictions.|
|[Evaluating_BERT_and_RoBERTa_on_the_unseen_test_data.ipynb](Evaluating_BERT_and_RoBERTa_on_the_unseen_test_data.ipynb) | This notebook evaluates the fine-tuned BERT and RoBERTa models on the unseen test data. __PLEASE NOTE__: This requires the saved models that are obtained from training the BERT and RoBERTa models in the [NLP_CW_1_BertForSequenceClassification.ipynb](NLP_CW_1_BertForSequenceClassification.ipynb) and the  [NLP_CW_1_RoBertaForSequenceClassification.ipynb](NLP_CW_1_RoBertaForSequenceClassification.ipynb) notebooks above. |
| [task_1_main.ipynb](task_1_main.ipynb) | This notebook explores approaches to solving this problem that do not use pre-trained embeddings (WACCER, NER, Clustering, Entity Funniness Scores (EFS)). |
| [NLP_CW.pdf](NLP_CW.pdf) | The report for this assignment |