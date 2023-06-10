from transformers import BertModel, BertTokenizer, pipeline
import torch
import os


class Eval:

    def __init__(self):
        self.model = None
        self.tokenizer = None


    def init_model(self):
        dirname = os.path.dirname(__file__)[:-3]
        filename = os.path.join(dirname, 'data/') 
        self.tokenizer = BertTokenizer.from_pretrained(filename + 'tokenizer-vocab.txt') 
        self.model = BertModel.from_pretrained(filename)
        self.model.eval()

    def eval_model(self,text):
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=512, 
            padding='max_length',
            truncation=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Access the model outputs
        last_hidden_states = outputs.last_hidden_state
        print('hidden states: ', last_hidden_states.shape)

    def mask(self):
        dirname = os.path.dirname(__file__)[:-3]
        filename = os.path.join(dirname, 'data\\')
        print(filename) 
        mask = pipeline('fill-mask', model= filename, tokenizer=filename)
        return mask