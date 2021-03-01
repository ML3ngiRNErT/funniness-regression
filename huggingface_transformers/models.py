from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from callbacks import *


'''
To use a model from HuggingFace, import it and add the relevant classes to the map
NOTE: This is a sequence classification task, use the models specifically trained for sequence classification only!
'''

# Model Parameters
BATCH_SIZE = 32
LR = 1e-5
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01
EPOCHS = 1

MODELS = {
    'bert': BertForSequenceClassification,

}

TOKENIZERS = {
    'bert': BertTokenizer,
}

class TransformerModel:

    @staticmethod
    def TransformerModelBuilder():
        return TransformerModel()

    def __init__(self, model_name='bert', pretrained_name='bert-base-uncased'):
        self.model = MODELS[model_name].from_pretrained(pretrained_name, num_labels=1)
        self.tokenizer = TOKENIZERS[model_name].from_pretrained(pretrained_name)
        self.lr = LR
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE
    
    def with_lr(self, lr=LR):
        self.lr = lr
        return self
    
    def with_batch_size(self, batch_size=BATCH_SIZE):
        self.batch_size = batch_size
        return this
    
    def with_epochs(self, epochs=EPOCHS):
        self.epochs = epochs
        return self

    def build(self):
        return self

    def tokenize(self, data):
        return self.tokenizer(
            data, 
            add_special_tokens=False, 
            padding=True, 
            return_tensors="pt"
        )

    def to(self, device):
        self.model.to(device)

    def _get_training_args(self):
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=self.epochs,              # total number of training epochs
            per_device_train_batch_size=self.batch_size,  # batch size per device during training
            per_device_eval_batch_size=self.batch_size,   # batch size for evaluation
            warmup_steps=WARMUP_STEPS,                # number of warmup steps for learning rate scheduler
            weight_decay=WEIGHT_DECAY,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=1,
            evaluation_strategy="steps",
            learning_rate=self.lr,
            eval_steps=50
        )

    def train(self, train_dataset, eval_dataset=None):
        trainer = Trainer(
            model=self.model,                         # the instantiated 🤗 Transformers model to be trained
            args=self._get_training_args(),                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=eval_dataset,             # evaluation dataset
            compute_metrics = evaluation_metric
        )

        trainer.train()

        