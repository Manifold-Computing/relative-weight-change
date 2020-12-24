import argparse
import json
import os
from types import SimpleNamespace

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.trainer import seed_everything

from model import RWCModel
from utils.logger import lightningLogger


def main(args, configs):

    seed_everything(42)

    model = RWCModel(configs)

    # early stop when validation accuracy stops increasing
    early_stop_callback = EarlyStopping(
                            monitor='val_accuracy',
                            min_delta=0.00,
                            patience=5,
                            verbose=False,
                            mode='max')
    # lightning trainer 
    trainer = Trainer(
            deterministic=True,
            gpus=configs.gpus,
            accelerator='dp',
            fast_dev_run=args.test_run,
            logger=lightningLogger(configs.experimentName),
            callbacks=[early_stop_callback])
    
    trainer.fit(model)

    print("Done!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser for RWC")

    test_run = parser.add_argument('--test_run', default=0, type=int)

    args = parser.parse_args()

    # load configs
    configs = SimpleNamespace(**json.load(open('./configs.json')))

    main(args, configs)
