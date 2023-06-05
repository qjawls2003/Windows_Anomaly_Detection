import sys
from pre_processing.src.csv_to_df import *

def main():
    #csv to df to separate columns
    df_list_columns = csv_to_df()

    #clean the command line column
    sentences_to_train = preprocess(df_list_columns)



if __name__ == '__main__':
    main()