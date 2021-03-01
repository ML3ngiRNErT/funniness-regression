import pandas as pd
import re
import nltk
import spacy
import torch
from nltk.corpus import stopwords
from cleantext import clean
from ekphrasis.classes.preprocessor import TextPreProcessor
from torch.utils.data import Dataset

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

class Task1Dataset(Dataset):

    def __init__(self, train_data, labels):
        self.x_train = train_data
        self.y_train = labels

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.x_train.items()}
        item['labels'] = torch.tensor(self.y_train[idx], dtype=torch.float)
        return item


class Preprocessor:

    @staticmethod
    def PreprocessorBuilder():
        return Preprocessor()

    def __init__(self):
        self.transformations = []
        self.text_processor = TextPreProcessor(
            fix_html=True,  # fix HTML tokens

            # corpus from which the word statistics are going to be used 
            # for word segmentation 
            segmenter="english", 

            # corpus from which the word statistics are going to be used 
            # for spell correction
            corrector="english", 

            unpack_hashtags=False,  # perform word segmentation on hashtags
            unpack_contractions=False,  # Unpack contractions (can't -> can not)
            spell_correct=True,  # spell correction for elongated words
        )

        self.punct = "[\.,:;\(\)\[\]@\-\$£]"

        nltk.download('stopwords')
        self.stops = stopwords.words('english')

        self.nlp = spacy.load('en_core_web_lg')

    def _capitalisation_by_ner(self, sentence, entities=['GPE', 'ORG', 'NORP', 'PERSON']):
        edited_row = []

        trial_doc = self.nlp(sentence)
            
        for tok in trial_doc:
            if tok.ent_type_ in entities:
                edited_row.append(tok.text)
            else:
                edited_row.append(tok.text.lower())
        
        return ' '.join(edited_row)

    def with_word_replacement(self):
        self.transformations.append(("apply", {"func": (lambda x: re.subn("<.*/>", x[1], x[0])[0]), "axis":1}))
        return self

    def with_capitalisation_by_ner(self):
        self.transformations.append(("apply", {"func": (lambda x: self._capitalisation_by_ner(x))}))
        return self

    def with_joining_contraction_tokens(self):
        self.transformations.append(("str.replace", {"pat": " (?P<one>\w*'\w+)", "repl": (lambda x: x.group("one"))}))
        return self

    def with_spell_check(self):
        self.transformations.append(("apply", {"func": (lambda x: self.text_processor.pre_process_doc(x))}))
        return self

    def with_space_after_hashtags(self):
        self.transformations.append(("str.replace", {"pat": "#", "repl": "# "}))
        return self

    def with_ascii_quotes_replacement(self):
        self.transformations.append(("str.replace", {"pat": "[‘’]", "repl": "'"}))
        return self

    def with_possessive_elimination(self):
        self.transformations.append(("str.replace", {"pat": "'s", "repl": ""}))
        return self

    def with_punct_removal(self):
        self.transformations.append(("str.replace", {"pat": self.punct, "repl": "'"}))
        return self

    def with_digit_removal(self):
        self.transformations.append(("str.replace", {"pat": "[0-9]", "repl": ""}))
        return self

    def with_stopwords_removal(self):
        self.transformations.append(("apply", {"func": (lambda x: " ".join([w for w in x.split(" ") if w not in self.stops]))}))
        return self
        
    def build(self):
        return self
    
    def preprocess(self, df, clean_col_name='edited_sentence'):
        _df = pd.DataFrame(index=df.index, columns=[clean_col_name, 'meanGrade'])

        _df['meanGrade'] = df.meanGrade
        
        transformed_cols = df[['original', 'edit']]
        
        for (func, params) in self.transformations:
            func_to_apply = transformed_cols
            for f in func.split("."):
                print(f)
                func_to_apply = getattr(func_to_apply, f)
            transformed_cols = func_to_apply(**params)
        
        _df[clean_col_name] = transformed_cols
        return _df, clean_col_name

