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

from rwc import RWC


class CIFARModel(pl.LightningModule):

    def __init__(self, out_channels=10, data='CIFAR10', batch_size=32):
        super(CIFARModel, self).__init__()
        self.model = resnet50()
        self.model.fc = nn.Linear(2048, out_channels)
        self.data = data
        self.batch_size = batch_size
        self.rwc = RWC()
        self.prev_weights, self.rwc_delta_dict = self.rwc.setup_delta_tracking(self)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)
        self.log('train_loss', loss,  on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', acc,  on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss}
        # pbar =   {'Train_accuracy': acc}

    def training_step_end(self, batch_part_outputs):
        # print("Training step done! Entering the end function")
        self.rwc_delta_dict, self.prev_weights = self.rwc.compute_delta(self.prev_weights, self.rwc_delta_dict)

        return {'loss': loss}

    def on_train_end(self, trainer, pl_module):
        for layer, values in self.rwc_delta_dict.items():
            count = 0
            for value in values:
                count += 1
                self.logger.experiment.log_metrics(layer, value, epoch=count)

    def validation_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ])
        dataset = CIFAR10("./", train=True, download=True, transform=train_transforms)
        return DataLoader(dataset, batch_size=self.batch_size)
            
    def val_dataloader(self):
        test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ]) 
        dataset = CIFAR10("./", train=False, download=True, transform=test_transforms)
        return DataLoader(dataset, batch_size=self.batch_size)
