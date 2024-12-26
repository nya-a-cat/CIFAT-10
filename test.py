import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor



training_data = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=ToTensor()
)

data_loader = torch.utils.data.DataLoader(training_data,
                                          batch_size=20,
                                          shuffle=True,)
