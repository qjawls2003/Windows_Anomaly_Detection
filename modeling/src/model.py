import torch


class MLM:

    def __init__(self, labels, mask):
        self.labels = torch.tensor(labels)
        self.mask = torch.tensor(mask)

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
            input_ids[i, selection] = 3  # our custom [MASK] token == 4