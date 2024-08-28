import torch
import math
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from typing import Callable
import matplotlib.pyplot as plt

class CelebADataset(Dataset):

    def __init__(self, dataset_path: str = "dataset", size: int = 256):
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        self.image_dataset = ImageFolder(root="dataset", transform=self.transform)

    def __getitem__(self, index: int):
        return self.image_dataset[index][0]

    def __len__(self):
        return len(self.image_dataset)

class NoisyBatchGenerator:

    def __init__(
        self,
        image_batch: torch.Tensor, 
        alpha_bar_t_values: list[float],
    ):
        self.image_batch = image_batch
        self.alpha_bar_t_values = alpha_bar_t_values
        self.t = 0
        self.max_t = len(alpha_bar_t_values)

    def __iter__(self):
        return self

    def __len__(self):
        return self.max_t

    def __next__(self):
        if self.t >= self.max_t:
            raise StopIteration
        else:
            noise = torch.randn_like(self.image_batch)
            abt = self.alpha_bar_t_values[self.t]
            self.t += 1
            return noise, math.sqrt(abt) * self.image_batch + torch.sqrt(1 - abt) * noise

class CelebABatched:

    def __init__(
        self,
        var_sch: Callable[int, float],
        max_t: int,
        batch_size: int = 64,
        dataset_path: str = "dataset",
        size: int = 256,
    ):
        self.celeba_loader = iter(DataLoader(CelebADataset(dataset_path, size), batch_size=batch_size))
        var_sch_values = torch.tensor([1 - var_sch(i) for i in range(0, max_t + 1)])
        self.alpha_bar_t = [torch.prod(var_sch_values[:i]) for i in range(1, max_t + 1)]

    def __len__(self):
        return len(self.celeba_loader)

    def __iter__(self):
        return self

    def __next__(self):
        img_batch = next(self.celeba_loader)
        noisy_batch_iterator = NoisyBatchGenerator(img_batch, self.alpha_bar_t)
        return noisy_batch_iterator


def plot_random_img(dataset):
    idx = random.randint(0, 10)
    i = 0
    while i < idx:
        gt_b, b = next(dataset)
        i += 1

    plt.imshow(gt_b[0].permute(1, 2, 0))

    fig, ax = plt.subplots(nrows=4, ncols=4)
    ax = ax.flatten()
    print(len(b))
    for noisy_batch_t, axis in zip(b, ax):
        axis.imshow(noisy_batch_t[0].permute(1, 2, 0).clip(0, 1))
    
dataset = CelebABatched(
    lambda x: x / 16,
    16
)
plot_random_img(dataset)