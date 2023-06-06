import sys
from pre_processing.src.preprocess import *
from pre_processing.src.tokenizer import *

def main():

    prep = Preprocess()
    #csv to df to separate columns
    df_list_columns = prep.csv_to_df()

    #clean the command line column
    sentences_to_train = prep.data_prep(df_list_columns)

    tokenizer = Tokenize(sentences_to_train)
    model = tokenizer.tokenize()

    encoding = model.encode('DESKTOP-7UHDSLL$ WORKGROUP 5hour 47minute night filebeat.exe elastic-agent.exe C: Program Files Elastic')
    print(encoding.tokens)


if __name__ == '__main__':
    main()