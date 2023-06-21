import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import random
import itertools

from wilds import get_dataset
from wilds.datasets import poverty_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader

from transformers import DistilBertTokenizerFast

import os.path

POVERTY_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        # Images are already normalized tensors
    ]
)

CAMELYON_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor()
        # Images are already normalized tensors
    ]
)

IWILDCAM_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((448, 448))
        # Images are already normalized tensors
    ]
)

RXRX1_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((256, 256))
        # Images are already normalized tensors
    ]
)

FMOW_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224))
        # Images are already normalized tensors
    ]
)

def poverty_split(path, split, fold):
    '''
        split = "train" | "test" | "val" | "id_val" | "id_test"
    '''
    return get_dataset(dataset="poverty", download=False, root_dir=_expand_path(path), fold=fold).get_subset(split, transform=POVERTY_TRANSFORM)

def poverty_loader(dataset, batch_size, subsample=None):
    return get_train_loader("standard", _wilds_subsample(dataset, subsample, batch_size), batch_size=batch_size, num_workers=6)


def poverty_trainloader(path, batch_size, val=False, subsample=None):
    split = "id_val" if val else "train"
    dataset = get_dataset(
        dataset="poverty", download=False, root_dir=_expand_path(path)
    ).get_subset(split, transform=POVERTY_TRANSFORM)
    return get_train_loader("standard", _wilds_subsample(dataset, subsample, batch_size), batch_size=batch_size, num_workers=6)

def poverty_valloader(path, batch_size, subsample=None):
    dataset = get_dataset(
        dataset="poverty", download=False, root_dir=_expand_path(path)
    ).get_subset("val", transform=POVERTY_TRANSFORM)
    return get_train_loader("standard", _wilds_subsample(dataset, subsample, batch_size), batch_size=batch_size, num_workers=6)

def poverty_testloader(path, batch_size, ood=False, subsample=None):
    split = "test" if ood else "id_test"
    dataset = get_dataset(
        dataset="poverty", download=False, root_dir=_expand_path(path)
    ).get_subset(split, transform=POVERTY_TRANSFORM)
    return get_train_loader("standard", _wilds_subsample(dataset, subsample, batch_size), batch_size=batch_size, num_workers=6)

def eval_poverty(path, output, target, metadata, ood=False):
    split = "test" if ood else "id_test"
    return get_dataset(
        dataset="poverty", download=False, root_dir=_expand_path(path)
    ).get_subset(split, transform=POVERTY_TRANSFORM).eval(output, target, metadata)


def civil_comments_trainloader(path, batch_size, val=False, subsample=None):
    split = "val" if val else "train"
    dataset = get_dataset(dataset="civilcomments", download=False, root_dir=_expand_path(path)).get_subset(split, transform=get_bert_transform(300))
    return get_train_loader("standard", _wilds_subsample(dataset, subsample, batch_size), batch_size=batch_size, num_workers=1)

def civil_comments_testloader(path, batch_size, subsample=None):
    dataset = get_dataset(dataset="civilcomments", download=False, root_dir=_expand_path(path)).get_subset("test", transform=get_bert_transform(300))
    return get_eval_loader("standard", _wilds_subsample(dataset, subsample, batch_size), batch_size=batch_size, num_workers=1)

def eval_civil_comments(path, output, target, metadata):
    return get_dataset(dataset="civilcomments", download=False, root_dir=_expand_path(path)).get_subset("test", transform=get_bert_transform(300)).eval(output, target, metadata)


def camelyon_split(path, split="train"):
    '''
        split = "train" | "test" | "val" | "id_val"
    '''
    return get_dataset(dataset="camelyon17", download=False, root_dir=_expand_path(path)).get_subset(split, transform=CAMELYON_TRANSFORM)

def camelyon_loader(dataset, batch_size, subsample=None):
    return get_train_loader("standard", _wilds_subsample(dataset, subsample, batch_size), batch_size=batch_size, num_workers=8)


def iwildcam_split(path, split="train"):
    '''
        split = "train" | "test" | "val" | "id_val" | "id_test"
    '''
    return get_dataset(dataset="iwildcam", download=False, root_dir=_expand_path(path)).get_subset(split, transform=IWILDCAM_TRANSFORM)

def iwildcam_loader(dataset, batch_size, subsample=None):
    return get_train_loader("standard", _wilds_subsample(dataset, subsample, batch_size), batch_size=batch_size, num_workers=8)


def rxrx1_split(path, split):
    '''
        split = "train" | "test" | "val" | "id_val" | "id_test"
    '''
    return get_dataset(dataset="rxrx1", download=False, root_dir=_expand_path(path)).get_subset(split, transform=RXRX1_TRANSFORM)

def rxrx1_loader(dataset, batch_size, subsample=None):
    return get_train_loader("standard", _wilds_subsample(dataset, subsample, batch_size), batch_size=batch_size, num_workers=6)


def amazon_split(path, split):
    '''
        split = "train" | "test" | "val" | "id_val" | "id_test"
    '''
    return get_dataset(dataset="amazon", download=False, root_dir=_expand_path(path)).get_subset(split, transform=get_bert_transform(512))

def amazon_loader(dataset, batch_size, subsample=None):
    return get_train_loader("standard", _wilds_subsample(dataset, subsample, batch_size), batch_size=batch_size, num_workers=1)


def fmow_split(path, split):
    '''
        split = "train" | "test" | "val" | "id_val" | "id_test"
    '''
    return get_dataset(dataset="fmow", download=False, root_dir=_expand_path(path)).get_subset(split, transform=FMOW_TRANSFORM)

def fmow_loader(dataset, batch_size, subsample=None):
    return get_train_loader("standard", _wilds_subsample(dataset, subsample, batch_size), batch_size=batch_size, num_workers=6)


def download_dataset(path, dataset):
    get_dataset(dataset=dataset, download=True, root_dir=_expand_path(path))


def get_bert_transform(max_tokens):
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def transform(text):
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        )
        x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    return transform

def imshow(img, dataset="unknown"):
    img = img[0:3].permute((2, 1, 0))
    if dataset == "poverty":
        img = img * torch.tensor([
            poverty_dataset._STD_DEVS_2009_17["RED"],
            poverty_dataset._STD_DEVS_2009_17["GREEN"],
            poverty_dataset._STD_DEVS_2009_17["BLUE"],
        ]) + torch.tensor([
            poverty_dataset._MEANS_2009_17["RED"],
            poverty_dataset._MEANS_2009_17["GREEN"],
            poverty_dataset._MEANS_2009_17["BLUE"],
        ])
    npimg = img.numpy()
    plt.axis("off")
    plt.tight_layout(pad=0)
    return plt.imshow(npimg)

def _wilds_subsample(dataset, subsample, batch_size):
    if subsample is None or subsample <= 0:
        return dataset
    
    new_dataset = torch.utils.data.Subset(dataset, range(subsample * batch_size))
    new_dataset.metadata_array = dataset.metadata_array
    new_dataset.collate = dataset.collate
    return new_dataset

def _expand_path(path):
    return os.path.expandvars(path + "/wilds")
