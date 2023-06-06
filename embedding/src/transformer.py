
from transformers import PreTrainedTokenizerFast


class Transformer:

    def __init__(self):
        pass

    def init_tokenizer(self,tokenizer):

        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            # tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

        return wrapped_tokenizer

