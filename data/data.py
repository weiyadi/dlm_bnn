"""
modified from https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/data/data.py
"""
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample, label


def getDataset(dataset, transform_train, transform_test=None):
    if transform_test is None:
        transform_test = transform_train

    if (dataset == 'CIFAR10'):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
        inputs = 3

    elif (dataset == 'CIFAR100'):
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100
        inputs = 3

    elif (dataset == 'SVHN'):
        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
        num_classes = 10
        inputs = 3

    elif (dataset == 'STL10'):
        trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform_test)
        num_classes = 10
        inputs = 3

    else:
        raise NotImplementedError

    return trainset, testset, inputs, num_classes


def getDataloader(trainset, testset, valid_size, batch_size, num_workers, split_train=True):
    if split_train:
        num_train = len(trainset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   sampler=train_sampler, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   sampler=valid_sampler, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  num_workers=num_workers)

        return train_loader, valid_loader, test_loader
    else:
        num_test = len(testset)
        indices = list(range(num_test))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_test))
        test_idx, valid_idx = indices[split:], indices[:split]

        test_sampler = SubsetRandomSampler(test_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                   sampler=valid_sampler, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, sampler=test_sampler,
                                                  num_workers=num_workers)

        return train_loader, valid_loader, test_loader


def getTransformedDataset(dataset, model_cfg):
    transform_train, transform_test = model_cfg.transform_train, model_cfg.transform_test
    return getDataset(dataset, transform_train, transform_test)
