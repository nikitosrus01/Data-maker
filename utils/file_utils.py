"""
file_utils.py
-------------
Вспомогательные функции для работы с файловой системой:
безопасные имена файлов, создание уникальных рабочих директорий,
упаковка результатов в zip-архив для скачивания пользователем.
"""

import os
import re
import shutil
import uuid
import zipfile
from typing import Iterable

from config import TMP_DIR


def safe_filename(name: str) -> str:
    """
    Убирает из имени файла символы, недопустимые в путях (актуально для
    названий, которые могут содержать текстовый запрос пользователя, например
    "самолёт" -> используется как часть имени файла аннотации).
    """
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^\w\-.]", "", name, flags=re.UNICODE)
    return name or "unnamed"


def create_run_dir(prefix: str = "run") -> str:
    """
    Создаёт уникальную временную директорию для одного запуска обработки
    (нужна, чтобы результаты разных пользователей/запусков не перезаписывали
    друг друга при параллельном использовании веб-интерфейса).
    """
    run_id = f"{prefix}_{uuid.uuid4().hex[:10]}"
    run_dir = os.path.join(TMP_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def zip_directory(source_dir: str, zip_path: str) -> str:
    """
    Упаковывает содержимое директории source_dir в архив zip_path.
    Возвращает путь к созданному архиву.
    """
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, source_dir)
                zf.write(full_path, arcname)

    return zip_path


def cleanup_dir(path: str) -> None:
    """Удаляет временную директорию, игнорируя ошибки (best-effort очистка)."""
    shutil.rmtree(path, ignore_errors=True)


def list_supported_files(paths: Iterable[str], extensions: Iterable[str]) -> list:
    """
    Фильтрует список путей, оставляя только файлы с поддерживаемыми
    расширениями. Используется при пакетной загрузке, когда пользователь
    может случайно приложить файлы неподдерживаемых форматов.
    """
    extensions = tuple(ext.lower() for ext in extensions)
    return [p for p in paths if str(p).lower().endswith(extensions)]
