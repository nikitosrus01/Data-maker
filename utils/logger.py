"""
logger.py
---------
Единая настройка логирования для всего приложения.
Логи пишутся одновременно в консоль и в файл app.log.
"""

import logging
import sys
from config import LOG_LEVEL, LOG_FILE


def get_logger(name: str) -> logging.Logger:
    """
    Возвращает настроенный логгер с заданным именем (обычно __name__ модуля).
    Повторные вызовы с одним и тем же именем не создают дублирующихся хендлеров.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Логгер уже настроен ранее — не добавляем хендлеры повторно
        return logger

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # Если файл лога недоступен для записи (например, read-only ФС),
        # продолжаем работу только с логированием в консоль.
        logger.warning("Не удалось открыть файл лога %s, пишем только в консоль", LOG_FILE)

    logger.propagate = False
    return logger
