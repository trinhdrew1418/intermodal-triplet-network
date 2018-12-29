import numpy as np
from PIL import Image
import torchvision as tv
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import torch


class NUS_WIDE(Dataset):
    def __init__(self, root, transform):
        self.imgs = tv.datasets.ImageFolder(root=root)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (index, sample, target) where target is class_index of the target class.
        """
        if self.transform is not None:
            return index, self.transform(self.imgs[index][0]), self.imgs[index][1]
        return index, self.data[index], self.labels[index]

    def __len__(self):
        return len(self.imgs)

    def init_concept_matrix(self):
        fname = "AllTags81.txt"

        with open(fname) as f:
            content = f.readlines()

        fname = "Concepts81.txt"

        with open(fname) as f:
            idx_to_concept = f.readlines()

        for idx, line in enumerate(idx_to_concept):
            idx_to_concept[idx] = line.split('\n')[0]

        n = len(content)
        self.concept_list = [None] * n

        for idx, line in enumerate(content):
            concepts = []
            for count, indicator in enumerate(line.split(' ')):
                if indicator != '\n' and int(indicator) == 1:
                    concepts.append(idx_to_concept[count])
            concept_list[idx] = concepts


    def init_tag_matrix(self):
        fname = "All_Tags.txt"

        with open(fname) as f:
            content = f.readlines()

        n = len(content)
        self.tag_list = [None] * n

        for line, idx in zip(content, range(n)):
            self.tag_list[idx] = line.split(' ')[1:]

    def init_relevancy_matrix(self):
        fname = "nuswide_metadata/AllTags81.txt"

        with open(fname) as f:
            content = f.readlines()

        n = len(content)
        self.relevancy_matrix = np.zeros((n,81), dtype=int)

        for line, idx in zip(content, range(n)):
            self.relevancy_matrix[idx,:] = np.array([int(c) for c in line.split(' ') if c is not '\n'])



# Dataset used for nearest neighbors loading
class NUS_WIDE_KNN(Dataset):
    def __init__(self, root, transform, text_labels):
        self.imgs = tv.datasets.ImageFolder(root=root)
        self.transform = transform
        self.text_labels = text_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        return self.transform(self.imgs[index][0]), index

    def get_text_label(self, index):
        return self.text_labels[self.imgs[index][1]]

    def get_raw_image(self, index):
        return self.imgs[index][0]

    def __len__(self):
        return len(self.imgs)
