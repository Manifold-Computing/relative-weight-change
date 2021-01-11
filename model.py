""" Module contains the Lightning Class. This class loads data, trains and validates the model 
"""

import numpy as np
import pytorch_lightning as pl
import torch.optim as optim
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F

from utils.load_data import train_dataset, val_dataset
from utils.load_model import model
from utils.rwc import RWC


class RWCModel(pl.LightningModule):
    """ Class contains different methods and callback functions to train a model
    """

    def __init__(self, configs):
        super(RWCModel, self).__init__()
        self.model = model(name=configs.model_name, data = configs.data, 
                            out_features=configs.out_channels, is_pretrained=False)
        self.data = configs.data
        self.learning_rate = configs.lr
        self.momentum = configs.momentum
        self.weight_decay = configs.weight_decay
        self.n_workers= configs.num_workers
        self.batch_size = configs.batch_size
        self.rwc = RWC()
        self.configs = configs
        self.prev_weights, self.rwc_delta_dict = self.rwc.setup_delta_tracking(self.model)

    def forward(self, x):
        """Method runs a forward pass through the model

        Args:
            x (image matrix): The input image as a matrix 

        Returns:
            output: Classification of images 
        """
        return self.model(x)

    def training_step(self, batch, batch_nb):
        """ Method reponsible to run a single training pass through the model

        Args:
            batch (list): List of different images as a matrix used in a single training step
            batch_nb (int): Current batch number

        Returns:
            loss: Cross Entropy Loss after the epoch
        """

        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)

        self.log('train_loss', loss,  on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', acc,  on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def training_epoch_end(self, training_step_outputs):
        """Method runs after a each training epoch is complete

        Args:
            training_step_outputs (list): List of metrics returned from the training step
        """

        self.rwc_delta_dict, self.prev_weights, rwc_curr_dict = \
            self.rwc.compute_delta(self.model, self.prev_weights, self.rwc_delta_dict)
        for layer, value in rwc_curr_dict.items():
            self.log(layer, value,  on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def on_train_end(self, trainer, pl_module):
        """Method runs after the training is complete
        """

        save_dir = f"{self.configs.root_dir}/{self.configs.experiment_name}/{self.configs.curr_seed}.npy"
        print(f"Trainer: {trainer}")
        print(f"PL_Module: {pl_module}")
        print(f"\n\n Saving RWC Delta Dict at {save_dir}...")
        np.save(save_dir, self.rwc_delta_dict)
        print("Done Saving!")

    def validation_step(self, batch, batch_nb):
        """ Method runs a single validation step on the test data

        Returns:
            loss: Validation loss on the current model
        """
        images, outputs = batch
        logits = self(images)
        loss = F.cross_entropy(logits, outputs)
        acc = accuracy(logits, outputs)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    def configure_optimizers(self):
        optimizer =  optim.SGD(self.parameters(), lr=self.learning_rate,
                      momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])

        return optimizer, scheduler

    def train_dataloader(self):
        return train_dataset(name=self.data, b_size=self.batch_size, n_workers=self.n_workers)
    def val_dataloader(self):
        return val_dataset(name=self.data, b_size=self.batch_size, n_workers=self.n_workers)
