import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class CopyTaskDataset(Dataset):
    def __init__(self, dataset, seq_length, T=10, samples=10000, rnd_seed=None, device='cuda'):
        super(CopyTaskDataset, self).__init__()
        self.device = device
        self.dataset = dataset
        if rnd_seed:
            torch.random.manual_seed(rnd_seed)
        self.samples = samples
        self.seq_length = seq_length
        self.T = T
        self.data = torch.zeros((samples, seq_length))
        self.target = torch.zeros((samples, seq_length))
        DUMMY = [8]
        SIGNAL = [9]
        for i in range(samples):
            after_letter = torch.tensor(DUMMY*(self.T-1)+SIGNAL+DUMMY*9)
            target_letters = torch.randint(0, 8, (10,))
            self.data[i,:] = torch.cat((target_letters,after_letter))
            self.target[i,:] = torch.cat((torch.tensor(DUMMY*(T+9)),target_letters))
        self.data = self.data.long().to(device)
        self.target = self.target.to(device)
        print(self.target.device)

    def __len__(self):
        return self.samples

    def __getitem__(self, item):
        input_data = self.data[item, :].unsqueeze(-1)
        output_data = self.target[item, :].unsqueeze(-1)

        sample = {
            'input': input_data,
            'output': output_data
        }

        return sample
