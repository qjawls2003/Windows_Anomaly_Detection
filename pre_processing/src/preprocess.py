import pandas as pd
import os
import datetime


class Preprocess:

    def __init__(self):
        self.months = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
    
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
        #make a list of sentences to feed the word embedding model

        sentences = []

        #create a list of tokens per log = sentence
        for timestamp,cmd,hosts,process,parent_process,username,user_id,user_domain in zip(timestamps,commands,hosts,processes,parent_processes,usernames,user_ids,user_domains):
            words = cmd.split("\\") #get rid of // for directories
            temp = []
            month, day, year, weekday, hour, minute, second, daynight = self.generalize_timestamp(timestamp) #list of times
            process_time = [month, day, year, hour, minute, second, daynight, process, parent_process]

            #clean up command lines
            for word in words:
                if word and word not in ('??','-'):     
                    if word[0] == '"' and len(word) < 4:
                        temp.append(word[1:])

                    else:
                        sub_words = word.split()
                        for w in sub_words:
                            temp.append(w)
                if temp:
                    #put together time tokens and command line tokens
                    sentences.append(process_time + temp)

        return sentences

    def generalize_timestamp(self,timestamp):
        sep = timestamp.split()
        month = self.months[sep[0]]
        day = int(sep[1][:-1])
        year = int(sep[2])
        timeoftheday = sep[4].split(":")
        hour, minute, second = int(timeoftheday[0]), int(timeoftheday[1]), int(timeoftheday[2].split(".")[0])
        x_date = datetime.date(year, month, day)
        weekday = int(x_date.strftime('%w'))
        daynight = self.generalize_day_night(hour)
        
        return month, day, year, weekday, hour, minute, second, daynight

    def generalize_day_night(self,hour):
        if hour >= 7 and hour < 17:
            return 'day'
        else:
            return 'night'

   


