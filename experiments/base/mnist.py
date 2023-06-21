import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import random
import medmnist

# Python-Fu to import from mnist-c which has a dash in it
# import importlib
# import os
# os.chdir("./mnist-c/")
# mnist_c = importlib.import_module("corruptions", "mnist-c")
# os.chdir("..")

IMAGE_SIZE = 28

def _flatten(x):
    return torch.flatten(x)

# Currently not used
# class CorruptTransform:
#     def __init__(self, seed):
#         self.rng = random.Random(seed)

#     def __call__(self, x):
#         corruption = self.rng.choice(mnist_c.CORRUPTIONS)
#         return getattr(mnist_c, corruption)(x)

def gen_transform(flatten):
    transform = [transforms.Grayscale()]

    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(0.1307, 0.3081)) # Values taken from Wilson et al.

    if flatten:
        transform.append(transforms.Lambda(_flatten))

    return transforms.Compose(transform)

def trainloader(path, batch_size: int = 5, shuffle: bool = True, flatten: bool = False) -> torch.utils.data.DataLoader:
    dataset = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=gen_transform(flatten))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def testloader(path, batch_size: int = 5, shuffle: bool = True, flatten: bool = False) -> torch.utils.data.DataLoader:
    dataset = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=gen_transform(flatten))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

def fashion_trainloader(path, batch_size: int = 5, shuffle: bool = True, exclude_classes = [], flatten: bool = False) -> torch.utils.data.DataLoader:
    dataset = torchvision.datasets.FashionMNIST(root=path, train=True, download=True, transform=gen_transform(flatten))
    _select_classes(dataset, exclude_classes)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=5)

def fashion_testloader(path, batch_size: int = 5, shuffle: bool = True, exclude_classes = [], flatten: bool = False) -> torch.utils.data.DataLoader:
    dataset = torchvision.datasets.FashionMNIST(root=path, train=False, download=True, transform=gen_transform(flatten))
    _select_classes(dataset, exclude_classes)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=5)

class CorruptedFashionMNIST(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, exclude_classes):
        self.exclude_classes = exclude_classes
        super().__init__(root, transform)

    def find_classes(self, directory):
        orig_classes, orig_mappings = super().find_classes(directory)
        for c in self.exclude_classes:
            orig_classes.remove(str(c))
            orig_mappings.pop(str(c))
        return orig_classes, orig_mappings


def corrupted_fashion_testloader(path, batch_size: int = 5, shuffle: bool = True, exclude_classes = [], flatten: bool = False) -> torch.utils.data.DataLoader:
    if exclude_classes != []:
        raise ValueError("Cannot select classes from the corrupted dataset")
    dataset = torchvision.datasets.ImageFolder(path + "FashionMNIST-Test(C)", gen_transform(flatten))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

medmnist_classes = {
    "path": 9,
    "derma": 7,
    "oct": 4,
    "pneumonia": 2,
    "breast": 2,
    "blood": 8,
    "tissue": 8,
    "organA": 11,
    "organC": 11,
    "organS": 11
}

medmnist_datasets = {
    "path": medmnist.PathMNIST,
    "derma": medmnist.DermaMNIST,
    "oct": medmnist.OCTMNIST,
    "pneumonia": medmnist.PneumoniaMNIST,
    "breast": medmnist.BreastMNIST,
    "blood": medmnist.BloodMNIST,
    "tissue": medmnist.TissueMNIST,
    "organA": medmnist.OrganAMNIST,
    "organC": medmnist.OrganCMNIST,
    "organS": medmnist.OrganSMNIST
}

def medmnist_trainloader(path, dataset, batch_size: int, shuffle: bool =True, exclude_classes=[]) -> torch.utils.data.DataLoader:
    dataset = medmnist_datasets[dataset](split="train", download=True, transform=gen_transform(False), target_transform=transforms.Lambda(lambda target: target.squeeze(-1)))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

def medmnist_testloader(path, dataset, batch_size: int, shuffle: bool =True, exclude_classes=[]) -> torch.utils.data.DataLoader:
    dataset = medmnist_datasets[dataset](split="test", download=True, transform=gen_transform(False), target_transform=transforms.Lambda(lambda target: target.squeeze(-1)))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)


def _select_classes(dataset, exclude_classes):
    indices = torch.full_like(dataset.targets, False, dtype=torch.bool)
    for i in range(10):
        if not i in exclude_classes:
            indices |= dataset.targets == i
    dataset.targets = dataset.targets[indices]
    dataset.data = dataset.data[indices]

def imshow(img):
    #img = img.reshape((IMAGE_SIZE, IMAGE_SIZE, 1)) # unflatten
    img = img / 2 + 0.5     # denormalize
    npimg = img.numpy()
    plt.imshow(npimg, cmap=plt.get_cmap('gray'))