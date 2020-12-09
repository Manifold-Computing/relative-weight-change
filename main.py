import os

import pytorch_lightning as pl
import torch
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

    def prepare_data(self):
        if self.data == 'CIFAR10':
            self.train_data = CIFAR10("./", train=True, download=True)
            self.val_data = CIFAR10("./", train=False, download=True)
        elif self.data == 'CIFAR100':
            self.train_data = CIFAR100("./", train=True, download=True)
            self.val_data = CIFAR100("./", train=False, download=True)
        elif self.data == 'MNIST':
            self.train_data = MNIST("./", train=True, download=True)
            self.val_data = MNIST("./", train=False, download=True)
         

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)
        pbar =   {'Train_accuracy': acc}
        return {'loss': loss, 'progress_bar': pbar}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        val_loss = F.cross_entropy(logits, y)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        return Dataloader(self.train_data, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return Dataloader(self.val_data, batch_size=self.batch_size)

# train_loader = DataLoader(CIFAR10(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

model = CIFARModel()
trainer = pl.Trainer(progress_bar_refresh_rate=20)    
trainer.fit(model)  

