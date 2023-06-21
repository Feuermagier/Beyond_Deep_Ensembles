import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import TensorDataset, ConcatDataset
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

# Values taken from Wilson et al.
MEAN = torch.tensor([0.49, 0.48, 0.44])
STD = torch.tensor([0.2, 0.2, 0.2])
normalize = transforms.Normalize(MEAN, STD)

# From https://github.com/izmailovpavel/understandingbdl/blob/5d1004896ea4eb674cff1c2088dc49017a667e9e/swag/models/preresnet.py
transform_train = transforms.Compose([
    transforms.Resize(32), # For STL10
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.Resize(32), # For STL10
    transforms.ToTensor(),
    normalize,
])


def cifar10_trainloader(path, batch_size: int = 4, shuffle: bool = True, exclude_classes = [], subsample=None):
    dataset = torchvision.datasets.CIFAR10(root=path, train=True, download=False, transform=transform_train)
    dataset.targets = torch.tensor(dataset.targets)
    _select_classes(dataset, exclude_classes)
    if subsample is not None:
        dataset.targets = dataset.targets[0:subsample]
        dataset.data = dataset.data[0:subsample]
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

def cifar10_testloader(path, batch_size: int = 4, shuffle: bool = True, exclude_classes = []):
    dataset = torchvision.datasets.CIFAR10(root=path, train=False, download=False, transform=transform_test)
    dataset.targets = torch.tensor(dataset.targets)
    _select_classes(dataset, exclude_classes)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

def stl10_testloader(path, batch_size, shuffle: bool = True):
    dataset = torchvision.datasets.STL10(root=path, split="test", download=True, transform=transform_test)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

def _select_classes(dataset, exclude_classes):
    indices = torch.full_like(dataset.targets, False, dtype=torch.bool)
    for i in range(10):
        if not i in exclude_classes:
            indices |= dataset.targets == i
    dataset.targets = dataset.targets[indices]
    dataset.data = dataset.data[indices]

def cifar10_corrupted_testloader(path, intensity, batch_size, shuffle=True):
    single_datasets = []
    labels = torch.from_numpy(np.load(path + "CIFAR-10-C/labels.npy")).long()

    for file in os.listdir(path + "CIFAR-10-C/"):
        if file == "labels.npy":
            continue

        tensors = torch.from_numpy(np.load(path + "CIFAR-10-C/" + file)).float() / 256
        data = tensors[intensity * 10000:(intensity+1) * 10000].permute((0, 3, 1, 2))
        data = normalize(data)
        single_datasets.append(TensorDataset(data, labels[intensity * 10000:(intensity+1) * 10000]))
    dataset = ConcatDataset(single_datasets)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

def imshow(img):
    #img = img.reshape((IMAGE_SIZE, IMAGE_SIZE, 1)) # unflatten
    img = img.permute((1, 2, 0))
    img = img * STD + MEAN     # denormalize
    npimg = img.numpy()
    plt.axis("off")
    plt.tight_layout(pad=0)
    return plt.imshow(npimg)