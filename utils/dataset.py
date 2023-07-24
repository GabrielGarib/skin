import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MetaDataset(Dataset):

    def __init__(self, img_paths, labels, metadata=None, transform=None):

        super().__init__()
        self.img_paths = img_paths
        self.labels    = labels
        self.metadata  = metadata

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        image = Image.open(self.img_paths[item]).convert('RGB')

        # Applying the transformations
        image = self.transform(image)

        if self.metadata is None:
            metadata = []
        else:
            metadata = self.metadata[item]

        if self.labels is None:
            labels = []
        else:
            labels = self.labels[item]

        return image, labels, metadata


def get_data_loader(img_paths, labels, metadata=None, transform=None, batch_size=30, shuffle=True, num_workers=4,
                     pin_memory=True):

    dataset    = MetaDataset(img_paths, labels, metadata, transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          pin_memory=pin_memory)
    return dataloader
