import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class GRU4RecDataset(Dataset):
    def __init__(self, sequences):
        self.inputs = []
        self.targets = []

        for seq in sequences:
            for t in range(1, len(seq)):
                self.inputs.append(seq[:t])
                self.targets.append(seq[t])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long),
        )

def collate_fn(batch):
    x, y = zip(*batch)
    x = pad_sequence(x, batch_first=True)
    y = torch.stack(y)
    return x, y
