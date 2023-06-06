import sys
from pre_processing.src.preprocess import *
from embedding.src.tokenizer import *
from embedding.src.transformer import *
from modeling.src.model import *


def main():

    prep = Preprocess()
    #csv to df to separate columns
    df_list_columns = prep.csv_to_df()

    #clean the command line column
    sentences_to_train = prep.data_prep(df_list_columns)

    #build a tokenizer from scratch!
    tokenizer_init = Tokenize(sentences_to_train) #tokenize all words in the sentences
    tokenizer_model = tokenizer_init.get_tokenizer() #make pretrained and saves it

    #make custom masked language model
    transformer = Transformer(sentences_to_train)
    tokenizer_fast = transformer.init_tokenizer(tokenizer_model)
    batch = transformer.batch_encoding()
    

    #test
    #'''
    print("testing:")
    encoding = tokenizer_fast.encode('Harambe DESKTOP-7UHDSLL 18hour 45minute night powershell.exe explorer.exe C: Program Files powershell.exe x -iext -ow -ver -- C: Users Harambe Document project windows Sysmon.zip C: Users Harambe Downloads project windows')
    token = tokenizer_fast('Harambe DESKTOP-7UHDSLL 18hour 45minute night powershell.exe explorer.exe C: Program Files powershell.exe x -iext -ow -ver -- C: Users Harambe Document project windows Sysmon.zip C: Users Harambe Downloads project windows')
    print(token)
    print(tokenizer_fast.convert_ids_to_tokens(encoding))
    #'''
    
    

if __name__ == '__main__':
    main()