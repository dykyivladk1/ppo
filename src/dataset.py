import torch
from torch.utils.data import Dataset

SRC_VOCAB_SIZE = 1000  
TGT_VOCAB_SIZE = 1000 


class ExampleDataset(Dataset):
    def __init__(self, num_samples=1000, src_seq_len=20, tgt_seq_len=20, src_vocab_size=SRC_VOCAB_SIZE, tgt_vocab_size=TGT_VOCAB_SIZE, pad_idx=0):
        super(ExampleDataset, self).__init__()
        self.num_samples = num_samples
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.pad_idx = pad_idx

        self.src_data = torch.randint(1, src_vocab_size, (num_samples, src_seq_len))
        self.tgt_data = torch.randint(1, tgt_vocab_size, (num_samples, tgt_seq_len))
        for i in range(num_samples):
            pad_length = torch.randint(5, src_seq_len, (1,)).item()
            self.src_data[i, pad_length:] = pad_idx
            pad_length = torch.randint(5, tgt_seq_len, (1,)).item()
            self.tgt_data[i, pad_length:] = pad_idx

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]
