#https://huggingface.co/learn/nlp-course/chapter6/8?fw=pt
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import os

class Tokenize:

    def __init__(self, dataset):

        #init tokenizer
        self.dataset = dataset
        self.tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]")) #We have to specify the unk_token so the model knows what to return when it encounters characters it hasn’t seen before.
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace() #we only need to split the words by white spaces for normalization
        self.special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        self.trainer = trainers.WordPieceTrainer(vocab_size=8694, special_tokens=self.special_tokens,min_frequency=5) #change min_frequency and vocab_size for fine-tuning
        self.token_loc = ''



    def tokenize(self):

        #build tokenizer
        self.tokenizer.train_from_iterator(self.get_training_corpus(), trainer=self.trainer)
        cls_token_id = self.tokenizer.token_to_id("[CLS]")
        sep_token_id = self.tokenizer.token_to_id("[SEP]")
        self.tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
        )        

        self.tokenizer.decoder = decoders.WordPiece(prefix="##")

        #save model
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../../modeling/data/tokenizer.json') #this will be saved in bazel's local
        self.token_loc = filename
        self.tokenizer.save(filename)
        
        return self.tokenizer

    def get_training_corpus(self):
        for i in range(0, len(self.dataset), 1000):
            yield self.dataset[i : i + 1000]

    def get_tokenizer(self):
        self.tokenize()
        return Tokenizer.from_file(self.token_loc)
    

