import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.models import resnet50]
import argparse

def main(args):
    model = CIFARModel()
    trainer = pl.Trainer(
            gpus=args.gpus,
            num_nodes=args.nodes,
            accelerator='ddp'
        )  
    trainer.fit(model)  

class CIFARModel(pl.LightningModule):

    def __init__(self, out_channels=10, data='CIFAR10', batch_size=32):
        super(CIFARModel, self).__init__()
        self.model = resnet50()
        self.model.fc = nn.Linear(2048, out_channels)
        self.data = data
        self.batch_size = batch_size

    def forward(self, x):
        return self.model(x)

    # def prepare_data(self):
    #     if self.data == 'CIFAR10':
    #         self.train_data = CIFAR10("./", train=True, download=True)
    #         self.val_data = CIFAR10("./", train=False, download=True)
    #     elif self.data == 'CIFAR100':
    #         self.train_data = CIFAR100("./", train=True, download=True)
    #         self.val_data = CIFAR100("./", train=False, download=True)
    #     elif self.data == 'MNIST':
    #         self.train_data = MNIST("./", train=True, download=True)
    #         self.val_data = MNIST("./", train=False, download=True)
         

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
        dataset = CIFAR10("./", train=True, download=True)
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return Dataloader(dataset, batch_size=self.batch_size, sampler=dist_sampler)
    
    def val_dataloader(self):
        dataset = CIFAR10("./", train=False, download=True)
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return Dataloader(dataset, batch_size=self.batch_size, sampler=dist_sampler)

# train_loader = DataLoader(CIFAR10(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser for RWC")

    gpus = parser.add_argument('--gpus', type=int, default=1, help="Number of GPUs per node")
    nodes = parser.add_argument('--nodes', type=int, default=1, help="Number of nodes to use for the job")

    args = parser.parse_args()
    main(args)

