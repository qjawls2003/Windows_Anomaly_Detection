import sys
from pre_processing.src.preprocess import *
from embedding.src.tokenizer import *
from embedding.src.bert_tokenizer import *
from embedding.src.transformer import *
from modeling.src.model import *
from modeling.src.model_test import *


class Init:

    def __init__(self,argv):
        self.sentences_to_train = None
        self.tokenizer_model = None
        
        for i in range(1,len(argv)):
            f = 'self.{}()'.format(argv[i]) 
            eval(f)

    def preprocess(self):

        prep = Preprocess()
        #csv to df to separate columns
        df_list_columns = prep.csv_to_df()

        #clean the command line column
        self.sentences_to_train = prep.data_prep(df_list_columns)

        print(self.sentences_to_train[:150])

    
    def tokenize(self):
        #build a standard tokenizer from scratch!
        #tokenizer_init = Tokenize(self.sentences_to_train) #tokenize all words in the sentences
        #self.tokenizer_model = tokenizer_init.get_tokenizer() #make pretrained and saves it

        #build BertTokenizer from scratch that is compatible with BERT
        tokenizer_init = BertTokenize(self.sentences_to_train) #tokenize all words in the sentences
        tokenizer_init.tokenize

    def train(self):
        #make custom masked language model
        transformer = Transformer(self.sentences_to_train) #Step 1: Init class
        tokenizer_fast, tokenizer_length = transformer.init_tokenizer() #Step 2: Load the custom tokenizer
        print("Tokenizer Length is: ", tokenizer_length)   

         #test
        '''
        print("testing:")
        print(tokenizer_fast('Harambe DESKTOP-7UHDSLL 18hour 45minute night powershell.exe explorer.exe C: Program Files powershell.exe x -iext -ow -ver -- C: Users Harambe Document project windows Sysmon.zip C: Users Harambe Downloads project wind'))
        
        encoding = tokenizer_fast.encode('Harambe DESKTOP-7UHDSLL 18hour 45minute night powershell.exe explorer.exe C: Program Files powershell.exe x -iext -ow -ver -- C: Users Harambe Document project windows Sysmon.zip C: Users Harambe Downloads project windows')
        token = tokenizer_fast('Harambe DESKTOP-7UHDSLL 18hour 45minute night powershell.exe explorer.exe C: Program Files powershell.exe x -iext -ow -ver -- C: Users Harambe Document project windows Sysmon.zip C: Users Harambe Downloads project windows')
        print(token)
        print(tokenizer_fast.convert_ids_to_tokens(encoding))
        '''
        print("Training may take a while...")
        labels, mask = transformer.batch_encoding() #Step 3: encode the dataset using the tokenizer
        mlm = MLM(labels,mask,tokenizer_length)
        dataloader = mlm.load_dataset() #Step 4: prepare a dataset for training the model
        model, device = mlm.prep_model()
        mlm.train_model(model, dataloader, device)
        
    def test(self):
        #test model
    
        evaluate = Eval()
        mask = evaluate.mask()
        string = 'Harambe DESKTOP-7UHDSLL <mask> conhost.exe git.exe C: WINDOWS system32 conhost.exe'

        print(mask(string))
        '''
        evaluate.init_model()
        text = ['Harambe DESKTOP-7UHDSLL 18hour 45minute night powershell.exe explorer.exe C: Program Files powershell.exe x -iext -ow -ver -- C: Users Harambe Document project windows Sysmon.zip C: Users Harambe Downloads project windows', 'Harambe DESKTOP-7UHDSLL 18hour 45minute night powershell.exe explorer.exe C: Program Files powershell.exe x -iext -ow -ver -- C: Users Harambe Document project windows Sysmon.zip C: Users Harambe Downloads project windows']
        print("Testing: ", text)
        evaluate.eval_model(text)
        '''
        'http://localhost:5601/api/reporting/generate/csv_searchsource?jobParams=%28browserTimezone%3AAmerica%2FNew_York%2Ccolumns%3A%21%28%27%40timestamp%27%2Chost.name%2Cprocess.name%2Cprocess.parent.name%2Cprocess.pid%2Cprocess.command_line%2Cuser.name%2Cuser.id%2Cuser.domain%29%2CobjectType%3Asearch%2CsearchSource%3A%28fields%3A%21%28%28field%3A%27%40timestamp%27%2Cinclude_unmapped%3Atrue%29%2C%28field%3Ahost.name%2Cinclude_unmapped%3Atrue%29%2C%28field%3Aprocess.name%2Cinclude_unmapped%3Atrue%29%2C%28field%3Aprocess.parent.name%2Cinclude_unmapped%3Atrue%29%2C%28field%3Aprocess.pid%2Cinclude_unmapped%3Atrue%29%2C%28field%3Aprocess.command_line%2Cinclude_unmapped%3Atrue%29%2C%28field%3Auser.name%2Cinclude_unmapped%3Atrue%29%2C%28field%3Auser.id%2Cinclude_unmapped%3Atrue%29%2C%28field%3Auser.domain%2Cinclude_unmapped%3Atrue%29%29%2Cfilter%3A%21%28%28meta%3A%28field%3A%27%40timestamp%27%2Cindex%3A%27logs-%2A%27%2Cparams%3A%28%29%29%2Cquery%3A%28range%3A%28%27%40timestamp%27%3A%28format%3Astrict_date_optional_time%2Cgte%3Anow-30d%2Fd%2Clte%3Anow%29%29%29%29%29%2Cindex%3A%27logs-%2A%27%2Cparent%3A%28filter%3A%21%28%29%2Cindex%3A%27logs-%2A%27%2Cquery%3A%28language%3Akuery%2Cquery%3A%27winlog.event_id%3A%204688%27%29%29%2Csort%3A%21%28%28%27%40timestamp%27%3Adesc%29%29%29%2Ctitle%3A%27Win%20Event%204688%27%2Cversion%3A%278.7.0%27%29'
       
    def all(self):
        self.preprocess()
        self.tokenize()
        self.train()
        self.test()
        

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please input command: preprocess, tokenize, train, test, or all')
        exit(0)
    init = Init(sys.argv)