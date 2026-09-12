"""
video_processor.py
-------------------
Извлечение кадров из видеофайла через заданный интервал в секундах.

Ограничение/особенность: извлечение делается по номеру кадра, вычисленному
из FPS видео (frame_idx = round(fps * timestamp)). Для видео с переменным
FPS (variable frame rate, распространено в файлах с телефонов) реальные
временные метки извлечённых кадров могут отличаться от запрошенных на
доли секунды. Для большинства задач разметки датасета это несущественно,
но стоит иметь в виду при работе с точной синхронизацией по времени.
"""

import os
from dataclasses import dataclass
from typing import List

import cv2
from PIL import Image

from config import MAX_VIDEO_FRAMES
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedFrame:
    frame: Image.Image
    timestamp_sec: float
    frame_index: int


class VideoFrameExtractionError(Exception):
    """Ошибка при чтении или разборе видеофайла."""


def extract_frames(video_path: str, interval_sec: float) -> List[ExtractedFrame]:
    """
    Извлекает кадры из видео каждые interval_sec секунд, начиная с 0-й секунды.

    Бросает VideoFrameExtractionError, если файл не удаётся открыть/прочитать
    (например, повреждённый файл или неподдерживаемый кодек — в таком случае
    у пользователя, скорее всего, не установлен нужный кодек в сборке OpenCV
    или нужен ffmpeg).
    """
    if interval_sec <= 0:
        raise ValueError("Интервал извлечения кадров должен быть положительным числом секунд")

    if not os.path.exists(video_path):
        raise VideoFrameExtractionError(f"Файл не найден: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoFrameExtractionError(
            f"Не удалось открыть видеофайл: {video_path}. "
            "Проверьте, что файл не повреждён и кодек поддерживается OpenCV/ffmpeg."
        )

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not fps or fps <= 0:
            # Иногда контейнеры не сообщают корректный FPS. Не можем надёжно
            # сопоставить секунды с номером кадра — явно сообщаем об этом,
            # а не тихо используем произвольное значение по умолчанию.
            raise VideoFrameExtractionError(
                "Не удалось определить FPS видео (метаданные повреждены или отсутствуют). "
                "Разметка по интервалу в секундах невозможна для этого файла."
            )

        duration_sec = total_frames / fps if total_frames > 0 else 0
        logger.info(
            "Видео: fps=%.2f, кадров=%d, длительность≈%.1fс, интервал=%.1fс",
            fps, total_frames, duration_sec, interval_sec,
        )

        extracted: List[ExtractedFrame] = []
        timestamp = 0.0

        while True:
            frame_idx = round(timestamp * fps)
            if total_frames > 0 and frame_idx >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = cap.read()
            if not ok:
                # Достигнут конец видео или повреждённый кадр — прекращаем,
                # но не считаем это фатальной ошибкой, если что-то уже извлекли.
                logger.warning("Не удалось прочитать кадр №%d, прекращаем извлечение", frame_idx)
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            extracted.append(
                ExtractedFrame(
                    frame=Image.fromarray(frame_rgb),
                    timestamp_sec=round(timestamp, 3),
                    frame_index=frame_idx,
                )
            )

            if len(extracted) >= MAX_VIDEO_FRAMES:
                logger.warning(
                    "Достигнут лимит MAX_VIDEO_FRAMES=%d, дальнейшие кадры не извлекаются. "
                    "Увеличьте интервал или измените лимит в config.py для длинных видео.",
                    MAX_VIDEO_FRAMES,
                )
                break

            timestamp += interval_sec

        if not extracted:
            raise VideoFrameExtractionError(
                "Из видео не удалось извлечь ни одного кадра. "
                "Проверьте корректность файла и заданный интервал."
            )

        return extracted
    finally:
        cap.release()
