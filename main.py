""" Module is the entry point for the repository. Used to start training
"""
import argparse
import json
from types import SimpleNamespace

from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.trainer import seed_everything

from model import RWCModel
from utils.logger import lightning_logger


def main(args, configs):
    """Function runs the training code for RWC

    Args:
        args (dict): Contains the arguments for training
        configs (dict): Containts the hyperparameters required for taining
    """

    seed_everything(configs.curr_seed)

    model = RWCModel(configs)

    # early stop when validation accuracy stops increasing
    # early_stop_callback = EarlyStopping(
    #                         monitor='val_accuracy',
    #                         min_delta=0.00,
    #                         patience=5,
    #                         verbose=False,
    #                         mode='max')
    # lightning trainer
    trainer = Trainer(
            deterministic=True,
            gpus=configs.gpus,
            accelerator='dp',
            fast_dev_run=args.test_run,
            logger=lightning_logger(configs),
            default_root_dir=configs.root_dir
            # callbacks=[early_stop_callback]
            )   
    trainer.fit(model)

    print("Done!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser for RWC")

    test_run = parser.add_argument('--test_run', default=0, type=int)

    arguments = parser.parse_args()

    # load configs
    configurations = SimpleNamespace(**json.load(open('./configs.json')))

    # for multiple runs with different random weights
    for seed in configurations.seed:
        configurations.curr_seed = seed
        main(arguments, configurations)
