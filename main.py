import argparse
import os

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


def main(args):
    print(f"Nodes: {args.nodes} \tGPUs: {args.gpus}")
    model = CIFARModel()

    early_stop_callback = EarlyStopping(
                            monitor='val_accuracy',
                            min_delta=0.00,
                            patience=5,
                            verbose=False,
                            mode='max')
    trainer = pl.Trainer(
            gpus=args.gpus,
            num_nodes=args.nodes,
            distributed_backend='ddp',
            fast_dev_run=args.test_run
            callbacks=[early_stop_callback])
    
    trainer.fit(model)

    print("Done!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser for RWC")

    test_run = parser.add_argument('--test_run', action='store_true')
    gpus = parser.add_argument('--gpus', type=int, default=1, help="Number of GPUs per node")
    nodes = parser.add_argument('--nodes', type=int, default=1, help="Number of nodes to use for the job")

    args = parser.parse_args()
    main(args)

