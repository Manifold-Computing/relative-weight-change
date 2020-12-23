from pytorch_lightning.loggers.comet import CometLogger


def lightningLogger(experiment_name):
    logger = CometLogger(api_key="ZgD8zJEiZErhwIzPMfZpitMjq",
                    project_name=experiment_name)

    return logger
