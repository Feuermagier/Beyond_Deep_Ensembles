import torch

from uci_datasets import Dataset
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

class UCIDataset:
    def __init__(self, name, split=0, normalize=True, val_percentage=1.0):
        super().__init__()
        self.val_percentage = val_percentage
        data = Dataset(name, print_stats=False)
        x_train, y_train, x_test, y_test = data.get_split(split)
        self.x_train, self.y_train, self.x_test, self.y_test = _convert_data(x_train), _convert_data(y_train), _convert_data(x_test), _convert_data(y_test)

        if normalize:
            self.x_mean = torch.cat((self.x_train, self.x_test)).mean(dim=0)
            self.x_std = torch.cat((self.x_train, self.x_test)).std(dim=0)
            self.y_mean = torch.cat((self.y_train, self.y_test)).mean(dim=0)
            self.y_std = torch.cat((self.y_train, self.y_test)).std(dim=0)
        else:
            self.x_mean = 0
            self.x_std = 1
            self.y_mean = 0
            self.y_std = 1

    def denormalize(self, output, target):
        return output * self.y_std + self.y_mean, target * self.y_std + self.y_mean

    def get_dataset(self, split, gap, device=None):
        if gap is None:
            x_train, y_train, x_test, y_test = self.x_train, self.y_train, self.x_test, self.y_test
        else:
            x_train, y_train, x_test, y_test = _reshuffle_for_gap(self.x_train, self.y_train, self.x_test, self.y_test, gap)
            
        if split == "train":
            x = x_train
            y = y_train
        elif split == "test":
            x = x_test
            y = y_test
        elif split == "val_train":
            x = x_train[ : int(0.9 * x_train.shape[0] * self.val_percentage)]
            y = y_train[ : int(0.9 * y_train.shape[0] * self.val_percentage)]
        elif split == "val_test":
            x = x_train[int(0.9 * x_train.shape[0]) : ]
            y = y_train[int(0.9 * x_train.shape[0]) : ]
        
        return _prepare_dataset(x, y, self.x_mean, self.x_std, self.y_mean, self.y_std, device=device)


def _prepare_dataset(x, y, x_mean, x_std, y_mean, y_std, device):
    x = (x - x_mean) / x_std
    y = (y - y_mean) / y_std
    
    if device is not None:
        x, y = x.to(device), y.to(device)

    return TensorDataset(x, y)

def _convert_data(data):
    return torch.from_numpy(data).float()

def _reshuffle_for_gap(x_train, y_train, x_test, y_test, dim):
        x = torch.cat((x_train, x_test))
        y = torch.cat((y_train, y_test))

        indices = torch.argsort(x, dim=0)[:,dim]
        train_indices = torch.cat((indices[: len(indices) // 3], indices[2 * len(indices) // 3 :]))
        train_indices = train_indices[torch.randperm(len(train_indices))]
        test_indices = indices[len(indices) // 3 : 2 * len(indices) // 3]
        test_indices = test_indices[torch.randperm(len(test_indices))]

        x_train = torch.cat((x[indices[: len(indices) // 3]], x[indices[2 * len(indices) // 3 :]]))
        y_train = torch.cat((y[indices[: len(indices) // 3]], y[indices[2 * len(indices) // 3 :]]))
        x_test = x[indices[len(indices) // 3 : 2 * len(indices) // 3]]
        y_test = y[indices[len(indices) // 3 : 2 * len(indices) // 3]]
        
        return x[train_indices], y[train_indices], x[test_indices], y[test_indices]

def get_loader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)