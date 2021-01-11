"""Module loads the Logger to log the graphs for RWC

Returns:
    logger: The Logger object used in logging different metrics
"""
from pytorch_lightning.loggers.comet import CometLogger


def lightning_logger(configs):
    """Method loads the comet logger for training

    Args:
        configs (dict): Dictionary contains logger experiment and project name

    Returns:
        logger: Logger object for the project
    """
    logger = CometLogger(api_key="ZgD8zJEiZErhwIzPMfZpitMjq",
                    project_name=configs.project_name,
                    experiment_name=f'{configs.curr_seed}_{configs.experiment_name}')

    return logger
