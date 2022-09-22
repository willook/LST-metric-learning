import torch
import numpy as np

def collate_fn_padd(device, batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    input_list, label_list = [], []
    for keypoints, label, index in batch:
        
        tensor = torch.tensor(keypoints).to(device)
        label = torch.tensor(label).to(device)
        input_list.append(tensor)
        label_list.append(label)
    input_tensor = torch.nn.utils.rnn.pad_sequence(input_list, batch_first=False)
    label_tensor = torch.tensor(label_list)
    mask_list = []
    seq_len = input_tensor.size()[0]
    for x in input_list:
        sample_len = len(x)
        indice = np.array(range(seq_len))
        mask = np.full(seq_len, False)
        mask[indice >= sample_len] = True
        mask_list.append(mask)

    mask_tensor = torch.tensor(np.array(mask_list)).to(device)
    return input_tensor.float(), label_tensor, mask_tensor

if __name__ == '__main__':
    collate_fn_padd()