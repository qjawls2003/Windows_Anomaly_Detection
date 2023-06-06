
from transformers import BertTokenizer
import os


class Transformer:

    def __init__(self):
        pass

    def init_tokenizer(self):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../data/tokenizer.json') #this will be saved in bazel's local

        tokenizer = BertTokenizer.from_pretrained(filename)

        return tokenizer

