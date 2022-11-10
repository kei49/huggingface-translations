import logging
from dataclasses import dataclass
from datetime import datetime


STD_FORMATTER = logging.Formatter(
    '%(levelname)s - "%(pathname)s", line %(lineno)d, in %(module)s - %(funcName)s \n  %(message)s')
FILE_FORMATTER = logging.Formatter(
    '%(asctime)s - %(levelname)s - "%(pathname)s", line %(lineno)d, in %(module)s - %(funcName)s \n  %(message)s')


@dataclass
class StreamHandlerConfig:
    stream_handler: logging.StreamHandler
    logging_level: logging.INFO | logging.DEBUG | logging.ERROR
    formatter: logging.Formatter


def use_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    [info_log_file_path, error_log_file_path] = [
        setup_log_file(level) for level in ['info', 'error']]

    std_sh_config = StreamHandlerConfig(
        logging.StreamHandler(), logging.DEBUG, STD_FORMATTER)
    info_file_sh_config = StreamHandlerConfig(logging.FileHandler(
        info_log_file_path), logging.INFO, FILE_FORMATTER)
    error_file_sh_config = StreamHandlerConfig(logging.FileHandler(
        error_log_file_path), logging.ERROR, FILE_FORMATTER)

    for sh_config in [std_sh_config, info_file_sh_config, error_file_sh_config]:
        sh = sh_config.stream_handler
        sh.setLevel(sh_config.logging_level)
        sh.setFormatter(sh_config.formatter)

        logger.addHandler(sh)

    return logger


def setup_log_file(level: str):
    log_base_dir = "logs"
    today_str = get_today()

    log_file_path = f"{log_base_dir}/{today_str}-{level}.log"
    return log_file_path


def get_today():
    return datetime.now().strftime('%Y%m%d')
