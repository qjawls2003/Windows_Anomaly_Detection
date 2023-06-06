from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

class Tokenize:

    def __init__(self, dataset):
        self.dataset = dataset
        self.tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]")) #We have to specify the unk_token so the model knows what to return when it encounters characters it hasn’t seen before.
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace() #we only need to split the words by white spaces for normalization
        self.special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        self.trainer = trainers.WordPieceTrainer(vocab_size=50000, special_tokens=self.special_tokens)



    def tokenize(self):
        self.tokenizer.train_from_iterator(self.dataset, trainer=self.trainer)
        cls_token_id = self.tokenizer.token_to_id("[CLS]")
        sep_token_id = self.tokenizer.token_to_id("[SEP]")
        print(cls_token_id, sep_token_id)
        

        return self.tokenizer
