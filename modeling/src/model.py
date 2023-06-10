import torch
from transformers import RobertaConfig, RobertaForMaskedLM, AdamW
import os
from tqdm import *

class MLM:

    def __init__(self, labels, mask, tokenizer_length):
        self.labels = torch.tensor(labels)
        self.mask = torch.tensor(mask)
        self.vocab_length = tokenizer_length

    def prep_tensor(self):
        
        # make copy of labels tensor, this will be input_ids
        input_ids = self.labels.detach().clone()

        # create random array of floats with equal dims to input_ids
        rand = torch.rand(input_ids.shape)

        # mask random 15% where token is not 0 [UNK] or 1 [PAD], 2 [CLS], or 3 [SEP]
        mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2) * (input_ids != 3)

        # loop through each row in input_ids tensor (cannot do in parallel)
        for i in range(input_ids.shape[0]):

            # get indices of mask positions from mask array
            selection = torch.flatten(mask_arr[i].nonzero()).tolist()

            # mask input_ids
            input_ids[i, selection] = 4  # our custom [MASK] token == 4
        
        encodings = {'input_ids': input_ids, 'attention_mask': self.mask, 'labels': self.labels}

        print("{} tokenized sequences and each containing {} tokens".format(input_ids.shape[0],input_ids.shape[1]))
        print("Input shape: ", input_ids.shape)
        return encodings

    def load_dataset(self):
        tensor_dataset = Dataset(self.prep_tensor())
        loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=16, shuffle=True)
        return loader
    

    def prep_model(self):
        config = RobertaConfig(
            vocab_size=self.vocab_length,  # we align this to the tokenizer vocab_size
            max_position_embeddings=514,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1
        )

        model = RobertaForMaskedLM(config)
        device = torch.device('cuda') #if torch.cuda.is_available() else torch.device('cpu') 
        model.to(device)
        return model, device
    
    def train_model(self, model, loader,device):
        # activate training mode
        model.train()
        # initialize optimizer
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

        epochs = 2

        for epoch in range(epochs):
            # setup loop with TQDM and dataloader
            loop = tqdm(loader, leave=True)
            for batch in loop:
                # initialize calculated gradients (from prev step)
                optim.zero_grad()
                # pull all tensor batches required for training
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                # process
                outputs = model(input_ids, attention_mask=attention_mask,
                                labels=labels)
                # extract loss
                loss = outputs.loss
                # calculate loss for every parameter that needs grad update
                loss.backward()
                # update parameters
                optim.step()
                # print relevant info to progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())
        
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../data/') 
        model.save_pretrained(filename)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}
