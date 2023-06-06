
from transformers import PreTrainedTokenizerFast, BertTokenizer
#mport torch

class Transformer:

    def __init__(self,dataset):
        self.standard_tokenizer = None
        self.dataset = dataset

    def init_tokenizer(self,tokenizer):

        self.standard_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            # tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

        return self.standard_tokenizer
    
    def batch_encoding(self):
        #encode all sentences in the dataset
        batch = self.standard_tokenizer(self.dataset, max_length=512, padding='max_length', truncation=True)

        #prepare tensors
        labels = [x.ids for x in batch]
        mask = [x.attention_mask for x in batch]

        return labels, mask

