
from transformers import PreTrainedTokenizerFast
from transformers import BertTokenizer
import os
class Transformer:

    def __init__(self,dataset):
        self.standard_tokenizer = None
        self.dataset = dataset
        self.filename = ''
        

    def init_tokenizer(self):
        '''
        self.standard_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            # tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        '''
        dirname = os.path.dirname(__file__)
        self.filename = os.path.join(dirname, "..\..\modeling\data/tokenizer-vocab.txt")
        self.standard_tokenizer = BertTokenizer.from_pretrained(self.filename)


        return self.standard_tokenizer, len(self.standard_tokenizer)
    
    def batch_encoding(self):
        #encode all sentences in the dataset
        batch = self.standard_tokenizer(self.dataset, max_length=512, padding='max_length', truncation=True)
        #prepare tensors
        print(len(batch['attention_mask']))
        labels, mask = [], []
        for ids in batch['input_ids']:
            labels.append(ids)
        for m in batch['attention_mask']:
            mask.append(m)
        
        print("Length of labels: {}".format(len(labels)))
        print("Length of mask: {}".format(len(mask)))

        return labels, mask

