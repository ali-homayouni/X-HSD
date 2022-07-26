import torch
from torch.utils.data import Dataset
from config import *

LABEL_DICT = {
    'en': LABEL_DICT_OLID,
    'de': LABEL_DICT_GERMEVAL,
    'fa': LABEL_DICT_FA,

    'train_en_test_en':LABEL_DICT_EN_DE,
    'train_en_test_de':LABEL_DICT_EN_DE,
    'train_fa_test_fa':LABEL_DICT_FA,

    'train_de_test_de':LABEL_DICT_EN_DE,
    'train_de_test_en':LABEL_DICT_EN_DE,

    'train_ende_test_ende':LABEL_DICT_EN_DE,
    'train_ende_test_en':LABEL_DICT_EN_DE,
    'train_ende_test_de':LABEL_DICT_EN_DE,

    'en_de': LABEL_DICT_EN_DE,
}

SEPERATOR = ','

def get_labels(labels):
    return labels.strip().split(SEPERATOR)
class HuggingfaceDataset(Dataset):
    def __init__(self, input_ids, lens, mask, labels, task, data='en'):
        self.input_ids = torch.tensor(input_ids)
        self.lens = lens
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.labels = labels
        self.task = task
        self.data = data
        self.label_dict = LABEL_DICT[self.data][self.task]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        input = self.input_ids[idx]
        length = self.lens[idx]
        mask = self.mask[idx]
        total = get_labels(self.labels[idx])
        label = {'label': torch.tensor([self.label_dict[i] for i in total])}
        return input, length, mask, label

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """
    Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset.labels))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, id_):
        return dataset.labels[id_]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
