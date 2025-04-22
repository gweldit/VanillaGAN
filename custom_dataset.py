import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class CustomSequenceDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


class BalancedSampler(Sampler):
    def __init__(self, dataset, class_vector):
        self.dataset = dataset
        self.class_vector = class_vector
        self.class_count = np.unique(class_vector, return_counts=True)[1]
        self.weight = 1.0 / self.class_count
        self.weight_per_class = self.weight / self.weight.sum()

    def __iter__(self):
        # Create a list to hold the indices for each class
        indices_per_class = [
            np.where(self.class_vector == i)[0] for i in range(len(self.class_count))
        ]
        # Shuffle the indices for each class
        indices_per_class = [
            torch.randperm(len(indices)).tolist() for indices in indices_per_class
        ]
        # Calculate the number of samples per class in each batch
        samples_per_class = int(self.dataset.__len__() / len(self.class_count))
        # Create the final list of indices for sampling
        indices = []
        for _ in range(samples_per_class):
            for class_indices in indices_per_class:
                indices.extend(
                    class_indices.pop()
                    for _ in range(self.weight_per_class[class_indices[0]])
                )
        return iter(indices)

    def __len__(self):
        return self.dataset.__len__()
