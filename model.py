import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

from utils.load_data import train_dataset, val_dataset
from utils.load_model import model
from utils.rwc import RWC


class RWCModel(pl.LightningModule):

    def __init__(self, configs):
        super(RWCModel, self).__init__()
        self.model = model(name=configs.model_name, data = configs.data, out_channels=configs.out_channels, is_pretrained=False)
        self.data = configs.data
        self.lr = configs.lr
        self.n_workers= configs.num_workers
        self.batch_size = configs.batch_size
        self.rwc = RWC()
        self.prev_weights, self.rwc_delta_dict = self.rwc.setup_delta_tracking(self.model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)

        self.log('train_loss', loss,  on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', acc,  on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def training_epoch_end(self, training_step_outputs):
        self.rwc_delta_dict, self.prev_weights, rwc_curr_dict = self.rwc.compute_delta(self.model, self.prev_weights, self.rwc_delta_dict)
        
        for layer, value in rwc_curr_dict.items():
            self.log(layer, value,  on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def validation_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return train_dataset(name=self.data, b_size=self.batch_size, n_workers=self.n_workers)
            
    def val_dataloader(self):
        return val_dataset(name=self.data, b_size=self.batch_size, n_workers=self.n_workers)
