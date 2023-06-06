import sys
from pre_processing.src.preprocess import *
from embedding.src.tokenizer import *
from embedding.src.transformer import *


def main():

    prep = Preprocess()
    #csv to df to separate columns
    df_list_columns = prep.csv_to_df()

    #clean the command line column
    sentences_to_train = prep.data_prep(df_list_columns)

    tokenizer_init = Tokenize(sentences_to_train) #train the sentences
    tokenizer_init.tokenize() #make model and saves it

    tokenizer_model = Transformer().init_tokenizer()

    #test
    #'''
    encoding = tokenizer_model.encode('Harambe DESKTOP-7UHDSLL 4hour 45minute day WinRAR.exe explorer.exe C: Program Files WinRAR WinRAR.exe x -iext -ow -ver -- C: Users Harambe Document project windows Sysmon (1).zip C: Users Harambe Downloads project windows')
    print(encoding)
    print(tokenizer_model.convert_ids_to_tokens(encoding))

    #'''
    
    

if __name__ == '__main__':
    main()