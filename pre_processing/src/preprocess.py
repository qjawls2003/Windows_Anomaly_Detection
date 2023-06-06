import pandas as pd
import os

class Preprocess:

    def __init__(self):
        pass
    
    def csv_to_df(self):

        #get the file with raw data
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../data/WinEvent4688.csv')
        #print(filename)

        df = pd.read_csv(filename)

        #make variable for each column for tokenization
        timestamps = df['@timestamp'].values.tolist()
        commands = df['process.command_line'].values.tolist()
        hosts = df['host.name'].values.tolist()
        processes = df['process.name'].values.tolist()
        parent_processes = df['process.parent.name'].values.tolist()
        usernames = df['user.name'].values.tolist()
        user_ids = df['user.id'].values.tolist()
        user_domains = df['user.domain'].values.tolist()

        return (timestamps,commands,hosts,processes,parent_processes,usernames,user_ids,user_domains)

    def data_prep(self,columns):

        #get all the columns from csv_to_df
        timestamps,commands,hosts,processes,parent_processes,usernames,user_ids,user_domains = columns
        #make a list of sentences to feed the Word2Vec
        train_commands = []
        for timestamp,cmd,hosts,process,parent_process,username,user_id,user_domain in zip(timestamps,commands,hosts,processes,parent_processes,usernames,user_ids,user_domains):
            words = cmd.split("\\")
            temp = []
            times = self.generalize_timestamp(timestamp) #list
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

    def generalize_timestamp(self,timestamp):
        print(timestamp)
        return timestamp



   


