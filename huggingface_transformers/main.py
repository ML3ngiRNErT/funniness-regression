import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, random_split
from preprocessing_fn import Preprocessor, Task1Dataset
from models import TransformerModel


# Setting random seed and device
SEED = 1

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if __name__ == '__main__':
    # Load data
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/dev.csv')

    # Preprocess the data
    preprocessor = Preprocessor.PreprocessorBuilder()\
                    .with_word_replacement()\
                    .with_capitalisation_by_ner()\
                    .with_joining_contraction_tokens()\
                    .with_spell_check()\
                    .with_space_after_hashtags()\
                    .with_ascii_quotes_replacement()\
                    .with_possessive_elimination()\
                    .with_punct_removal()\
                    .with_stopwords_removal()\
                    .with_digit_removal()\
                    .build()

    clean_train_df, edited_col_name_train = preprocessor.preprocess(train_df)
    clean_test_df, edited_col_name_test = preprocessor.preprocess(test_df)

    
    # We set our training data and test data
    training_data = clean_train_df[edited_col_name_train]
    test_data = clean_test_df[edited_col_name_test]

    # Create tokenizer, model
    model = TransformerModel.TransformerModelBuilder().build()
    print(model)
    print("Model initialised.")

    # Prepare the dataset
    train_X = model.tokenize(training_data.to_list())
    train_dataset = Task1Dataset(train_X, train_df['meanGrade'])


    model.to(device)

    train_proportion = 0.8

    train_examples = round(len(train_dataset)*train_proportion)
    dev_examples = len(train_dataset) - train_examples

    train_dataset, dev_dataset = random_split(train_dataset,
                                            (train_examples,
                                                dev_examples))

    model.train(train_dataset, dev_dataset)