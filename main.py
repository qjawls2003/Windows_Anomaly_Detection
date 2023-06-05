import sys
from pre_processing.src.csv_to_df import *
from pre_processing.src.vectorize import *

def main():
    #csv to df to separate columns
    df_list_columns = csv_to_df()

    #clean the command line column
    sentences_to_train = preprocess(df_list_columns)

    Word2Vec_model = vectorize(sentences_to_train)

    print(Word2Vec_model.wv['Harambe'])


if __name__ == '__main__':
    main()