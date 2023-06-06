import sys
from pre_processing.src.preprocess import *
from embedding.src.tokenizer import *

def main():

    prep = Preprocess()
    #csv to df to separate columns
    df_list_columns = prep.csv_to_df()

    #clean the command line column
    sentences_to_train = prep.data_prep(df_list_columns)

    tokenizer = Tokenize(sentences_to_train) #train the sentences
    model = tokenizer.tokenize() #make model

    #test
    '''
    encoding = model.encode('Harambe DESKTOP-7UHDSLL 4hour 45minute day WinRAR.exe explorer.exe C: Program Files WinRAR WinRAR.exe x -iext -ow -ver -- C: Users Harambe Document project windows Sysmon (1).zip C: Users Harambe Downloads project windows')
    print(encoding.tokens)
    print(encoding.type_ids)
    print(model.decode(encoding.ids))
    '''

    

if __name__ == '__main__':
    main()