import pandas as pd
import re
import nltk
import spacy
from cleantext import clean
from ekphrasis.classes.preprocessor import TextPreProcessor

# Params for 
clean_text_param = {
    "lower":False,                     # lowercase text
    "no_line_breaks":True,           # fully strip line breaks as opposed to only normalizing them
    "no_urls":False,                  # replace all URLs with a special token
    "no_emails":False,                # replace all email addresses with a special token
    "no_phone_numbers":False,         # replace all phone numbers with a special token
    "no_numbers":False,               # replace all numbers with a special token
    "no_digits":False,                # replace all digits with a special token
    "no_currency_symbols":True,      # replace all currency symbols with a special token
    "no_punct":True,                 # remove punctuations
    "replace_with_punct":"",          # instead of removing punctuations you may replace them
    "replace_with_number":"",
    "replace_with_digit":"",
    "replace_with_currency_symbol":"",
    "lang":"en"                       # set to 'de' for German special handling
}
nlp = spacy.load('en_core_web_sm')


def word_replacement(train_df, test_df):
    '''
    Does word replacement, i.e. replaces the word that needs to be edited (in </>)
        with the corresponding edit
        
    Inputs: train_df: pandas dataframe, training dataset
            test_df: pandas dataframe, test dataset
    
    '''
    train_df['edited_sentence'] = train_df[['original', 'edit']].apply(lambda x: re.subn("<(\w| )*/>", x[1], x[0])[0], axis=1)
    test_df['edited_sentence'] = test_df[['original', 'edit']].apply(lambda x: re.subn("<(\w| )*/>", x[1], x[0])[0], axis=1)
    return train_df, test_df

def tokenize(
    data, 
    is_lower=True, 
    remove_stopwords=True, 
    remove_puncts=True, 
    remove_num=True, 
    remove_currency=True
):

    text_processor = TextPreProcessor(
        annotate=['hashtag'],
        fix_html=True,  # fix HTML tokens
        
        # corpus from which the word statistics are going to be used 
        # for word segmentation 
        segmenter="english", 
        
        # corpus from which the word statistics are going to be used 
        # for spell correction
        corrector="english", 
        
        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct=True,
    )

    tokenized_corpus = []

    for sentence in data:

        tokenized_sentence = []
        # processed_sentence = text_processor.pre_process_doc(sentence)
        # clean_sentence = clean(processed_sentence, **clean_text_param)
        spacy_doc = nlp(sentence)

        for token in spacy_doc:
            processed_token = token
            if (remove_stopwords and processed_token.is_stop):
                continue
            elif (remove_puncts and processed_token.is_punct):
              continue
            elif (remove_num and processed_token.is_digit):
              continue
            elif (remove_currency and processed_token.is_currency):
              continue
            elif (is_lower):
              tokenized_sentence.append(token.lower_)
            else:
              tokenized_sentence.append(token.text)

        tokenized_corpus.append(tokenized_sentence)

    return tokenized_corpus

################ Functions provided as part of skeleton ######################

def create_vocab(data):
    """
    Creating a corpus of all the tokens used
    """
    tokenized_corpus = [] # Let us put the tokenized corpus in a list

    for sentence in data:

        tokenized_sentence = []

        for token in sentence.split(' '): # simplest split is

            tokenized_sentence.append(token)

        tokenized_corpus.append(tokenized_sentence)

    # Create single list of all vocabulary
    vocabulary = []  # Let us put all the tokens (mostly words) appearing in the vocabulary in a list

    for sentence in tokenized_corpus:

        for token in sentence:

            if token not in vocabulary:

                if True:
                    vocabulary.append(token)

    return vocabulary, tokenized_corpus