import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.models import resnet50


class CIFARModel(pl.LightningModule):

    def __init__(self, out_channels=10, data='CIFAR10', batch_size=32):
        super(CIFARModel, self).__init__()
        self.model = resnet50()
        self.model.fc = nn.Linear(2048, out_channels)
        self.data = data
        self.batch_size = batch_size

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)
        pbar =   {'Train_accuracy': acc}
        return {'loss': loss, 'progress_bar': pbar}

    def training_step_end(self, batch_part_outputs):
        print("Training step done! Entering the end function")

    def validation_step_end(self, batch_part_outputs):
        print("validation step done! Entering the val end function")

    def validation_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)
        pbar = {'val_accuracy': acc}
        return {'val_loss': loss, 'progress_bar': pbar}
            
    def val_dataloader(self):
        dataset = CIFAR10("./", train=False, download=True)
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return Dataloader(dataset, batch_size=self.batch_size, sampler=dist_sampler)
