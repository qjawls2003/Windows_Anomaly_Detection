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
            process_time = [str(hour) + 'hour', str(minute) + 'minute', daynight, process, parent_process]
            user_info = [username, user_domain]
            #clean up command lines
            for word in words:
                if word and word not in ('??',""):
                        
                    if word[0] == '"': #normalize "
                        word = word[1:]
                    if len(word) > 0 and word[-1] == '"': #normalize ""
                        word = word[:-1] 
                    
                    sub_words = word.split()
                    if sub_words:
                        for w in sub_words:
                            if word and word not in ('??',""):
                                if w[0] == '"': #normalize "
                                    w = w[1:]
                                if len(w) > 0 and w[-1] == '"': #normalize ""
                                    w = w[:-1]
                            temp.append(w)
            if temp:
                #put together time tokens and command line tokens
                sentence = ' '.join(user_info + process_time + temp)
            else:
                sentence = ' '.join(user_info + process_time)
            sentences.append(sentence)
                    #print(sentence)
        return sentences

    def generalize_timestamp(self,timestamp):
        sep = timestamp.split()
        month = self.months[sep[0]]
        month_name = sep[0]
        day = int(sep[1][:-1])
        year = int(sep[2])
        timeoftheday = sep[4].split(":")
        hour, minute, second = int(timeoftheday[0]), int(timeoftheday[1]), int(timeoftheday[2].split(".")[0])
        x_date = datetime.date(year, month, day)
        weekday = x_date.strftime('%A')
        daynight = self.generalize_day_night(hour)
        #print(month_name, day, year, weekday, hour, minute, second, daynight)
        return month_name, day, year, weekday, hour, minute, second, daynight

    def generalize_day_night(self,hour):
        if hour >= 7 and hour < 12:
            return 'morning'
        elif hour >= 12 and hour < 18:
            return 'afternoon'
        else:
            return 'night'

   


