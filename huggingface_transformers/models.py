from transformers import BertForSequenceClassification, BertTokenizer


'''
To use a model from HuggingFace, import it and add the relevant classes to the map
NOTE: This is a sequence classification task, use the models specifically trained for sequence classification only!
'''

MODELS = {
    'bert': BertForSequenceClassification,

}

TOKENIZERS = {
    'bert': BertTokenizer,
}

def get_model(model_name, pretrained_name):
    if model_name in MODELS.() and model_name in TOKENIZERS.keys():
        model = MODELS[model_name].from_pretrained(pretrained_name)
        tokenizer = TOKENIZERS[model_name].from_pretrained(pretrained_name)
        return model, tokenizer