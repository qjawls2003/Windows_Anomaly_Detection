from transformers import BertModel, BertTokenizer
import torch



class Eval:

    def __init__(self, model, tokenizer):
        self.model_loc = model
        self.tokenizer_loc = tokenizer
        self.model = None
        self.tokenizer = None


    def init_model(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_loc)
        self.model = BertModel.from_pretrained(self.model_loc)
        self.model.eval()

    def eval_model(self,text):
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Access the model outputs
        last_hidden_states = outputs.last_hidden_state
        print(last_hidden_states)

