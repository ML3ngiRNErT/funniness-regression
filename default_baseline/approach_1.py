import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset, random_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import codecs
import re
import sys

from training_functions import *
from BiLSTM import BiLSTM
from preprocessing_fn import *

if __name__ == "__main__":

    # Number of epochs
    epochs = 10

    # Proportion of training data for train compared to dev
    train_proportion = 0.8
    # Load data
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/dev.csv')

    # We set our training data and test data
    training_data = train_df['original']
    test_data = test_df['original']

    # Creating word vectors
    training_vocab, training_tokenized_corpus = create_vocab(training_data)
    test_vocab, test_tokenized_corpus = create_vocab(test_data)

    # Creating joint vocab from test and train:
    joint_vocab, joint_tokenized_corpus = create_vocab(pd.concat([training_data, test_data]))

    print("Vocab created.")

    # We create representations for our tokens
    wvecs = [] # word vectors
    word2idx = [] # word2index
    idx2word = []

    # This is a large file, it will take a while to load in the memory!
    with codecs.open('embeddings\glove.6B\glove.6B.100d.txt', 'r','utf-8') as f:
        index = 1
        for line in f.readlines():
            # Ignore the first line - first line typically contains vocab, dimensionality
            if len(line.strip().split()) > 3:
                word = line.strip().split()[0]
                if word in joint_vocab:
                    (word, vec) = (word,
                                list(map(float,line.strip().split()[1:])))
                    wvecs.append(vec)
                    word2idx.append((word, index))
                    idx2word.append((index, word))
                    index += 1

    wvecs = np.array(wvecs)
    word2idx = dict(word2idx)
    idx2word = dict(idx2word)

    vectorized_seqs = [[word2idx[tok] for tok in seq if tok in word2idx] for seq in training_tokenized_corpus]

    # To avoid any sentences being empty (if no words match to our word embeddings)
    vectorized_seqs = [x if len(x) > 0 else [0] for x in vectorized_seqs]

    INPUT_DIM = len(word2idx)
    EMBEDDING_DIM = 100
    BATCH_SIZE = 32


    model = BiLSTM(EMBEDDING_DIM, 50, INPUT_DIM, BATCH_SIZE, device)
    print("Model initialised.")

    model.to(device)
    # We provide the model with our embeddings
    model.embedding.weight.data.copy_(torch.from_numpy(wvecs))

    feature = vectorized_seqs

    # 'feature' is a list of lists, each containing embedding IDs for word tokens
    train_and_dev = Task1Dataset(feature, train_df['meanGrade'])

    train_examples = round(len(train_and_dev)*train_proportion)
    dev_examples = len(train_and_dev) - train_examples

    train_dataset, dev_dataset = random_split(train_and_dev,
                                            (train_examples,
                                                dev_examples))

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn_padd)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_padd)

    print("Dataloaders created.")

    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    train(train_loader, dev_loader, model, epochs, optimizer, loss_fn)