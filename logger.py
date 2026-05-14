import logging
import colorlog
import sys
import warnings


warnings.filterwarnings("ignore")


def filter(record) -> bool:
    if "cmdstanpy" in record.name:
        return False
    return True


def get_logger(name: str):
    logger = logging.getLogger(name)
    handler = colorlog.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "CRITICAL": "red",
            },
        )
    )
    handler.addFilter(filter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger
