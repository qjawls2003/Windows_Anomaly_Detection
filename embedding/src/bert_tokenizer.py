
from tokenizers import BertWordPieceTokenizer


import os

class BertTokenize:

    def __init__(self, dataset,filename):

        #init tokenizer
        self.dataset = dataset
        self.tokenizer = BertWordPieceTokenizer(
                            clean_text=True,
                            handle_chinese_chars=False,
                            strip_accents=False,
                            lowercase=False
                        )        
        self.token_loc = ''



    def tokenize(self):

        #build tokenizer
        self.tokenizer.train_from_iterator(
                self.get_training_corpus(), vocab_size=30_000, min_frequency=2,
                limit_alphabet=1000, wordpieces_prefix='##',
                special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
                )
        
        #save model
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../../modeling/data/') #this will be saved in bazel's local
        self.token_loc = filename
        self.tokenizer.save_model(filename, 'tokenizer')
        print("Tokenizer file line count: ", len(self.tokenizer))
        return self.tokenizer

    def get_training_corpus(self):
        for i in range(0, len(self.dataset), 1000):
            yield self.dataset[i : i + 1000]
