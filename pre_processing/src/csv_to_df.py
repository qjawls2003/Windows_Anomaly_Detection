import pandas as pd
import os

def csv_to_df():

    #get the file with raw data
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../data/WinEvent4688.csv')
    #print(filename)

    df = pd.read_csv(filename)

    #make variable for each column for tokenization
    commands = df['process.command_line'].values.tolist()
    hosts = df['host.name'].values.tolist()
    processes = df['process.name'].values.tolist()
    parent_processes = df['process.parent.name'].values.tolist()
    usernames = df['user.name'].values.tolist()
    user_ids = df['user.id'].values.tolist()
    user_domains = df['user.domain'].values.tolist()

    return [commands,hosts,processes,parent_processes,usernames,user_ids,user_domains]

def preprocess(columns):

    commands,hosts,processes,parent_processes,usernames,user_ids,user_domains =columns
    #make a list of sentences to feed the Word2Vec
    train_commands = []
    for cmd in commands:
        words = cmd.split("\\")
        temp = []
        for word in words:
            if word and word not in ('??','-'):     
                if word[0] == '"' and len(word) < 4:
                    temp.append(word[1:])

                else:
                    sub_words = word.split()
                    for w in sub_words:
                        temp.append(w)
            if temp:
                train_commands.append(temp)

    return train_commands

   


