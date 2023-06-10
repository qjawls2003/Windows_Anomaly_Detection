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
        labels, mask = transformer.batch_encoding() #Step 3: encode the dataset using the tokenizer
        mlm = MLM(labels,mask,tokenizer_length)
        dataloader = mlm.load_dataset() #Step 4: prepare a dataset for training the model
        model, device = mlm.prep_model()
        mlm.train_model(model, dataloader, device)
        
    def test(self):
        #test model
    
        evaluate = Eval()
        mask = evaluate.mask()
        print(mask(f'{mask.tokenizer.mask_token} DESKTOP-7UHDSLL 18hour 45minute night powershell.exe explorer.exe C: Program Files powershell.exe x -iext -ow -ver -- C: Users Harambe Document project windows Sysmon.zip C: Users Harambe Downloads project windows'))
        '''
        evaluate.init_model()
        text = ['Harambe DESKTOP-7UHDSLL 18hour 45minute night powershell.exe explorer.exe C: Program Files powershell.exe x -iext -ow -ver -- C: Users Harambe Document project windows Sysmon.zip C: Users Harambe Downloads project windows', 'Harambe DESKTOP-7UHDSLL 18hour 45minute night powershell.exe explorer.exe C: Program Files powershell.exe x -iext -ow -ver -- C: Users Harambe Document project windows Sysmon.zip C: Users Harambe Downloads project windows']
        print("Testing: ", text)
        evaluate.eval_model(text)
        '''
        
       
    def all(self):
        self.preprocess()
        self.tokenize()
        self.train()
        self.test()
        

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please input command: preprocess, tokenize, train, test')
        exit(0)
    init = Init(sys.argv)