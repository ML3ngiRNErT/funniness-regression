import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, random_split
from transformers import Trainer, TrainingArguments, BertForSequenceClassification, BertTokenizer
from preprocessing_fn import *
from callbacks import *
from models import *

# Model Parameters
BATCH_SIZE = 32
LR = 1e-5
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01
EPOCHS = 5


# Setting random seed and device
SEED = 1

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if __name__ == '__main__':
    # Load data
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/dev.csv')

    # Word replacement
    train_df, test_df = word_replacement(train_df, test_df)

    # We set our training data and test data
    # TODO Change 'original' to 'edited_sentence'
    training_data = train_df['original']
    test_data = test_df['original']

    # Preprocess the data 
    # TODO

    # Create tokenizer, model
    model, tokenizer = get_model("bert", "bert_base_uncased")
    print("Model initialised.")

    # Prepare the dataset
    train_X = tokenizer(training_data.to_list(), add_special_tokens=False, padding=True, return_tensors="pt")
    train_dataset = Task1Dataset(train_X, train_df['meanGrade'])


    model = model.to(device)

    train_proportion = 0.8

    train_examples = round(len(train_dataset)*train_proportion)
    dev_examples = len(train_dataset) - train_examples

    train_dataset, dev_dataset = random_split(train_dataset,
                                            (train_examples,
                                                dev_examples))


    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=EPOCHS,              # total number of training epochs
        per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training
        per_device_eval_batch_size=BATCH_SIZE,   # batch size for evaluation
        warmup_steps=WARMUP_STEPS,                # number of warmup steps for learning rate scheduler
        weight_decay=WEIGHT_DECAY,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=1,
        evaluation_strategy="steps",
        learning_rate=LR,
        eval_steps=50
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=dev_dataset,             # evaluation dataset
        compute_metrics = evaluation_metric
    )

    # trainer.train()