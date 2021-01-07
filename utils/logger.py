from pytorch_lightning.loggers.comet import CometLogger


def lightningLogger(configs):
    logger = CometLogger(api_key="ZgD8zJEiZErhwIzPMfZpitMjq",
                    project_name=configs.project_name,
                    experiment_name=f'{configs.curr_seed}_{configs.experiment_name}')

    return logger
