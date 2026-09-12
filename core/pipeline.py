"""
pipeline.py
-----------
Объединяет GroundingDINODetector и SAMSegmenter в единый пайплайн
"текст -> боксы -> маски", а также инкапсулирует структуру результата,
которая используется во всех остальных частях приложения (визуализация,
экспорт аннотаций, ручное редактирование).
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
from PIL import Image

from core.detector import GroundingDINODetector
from core.segmenter import SAMSegmenter
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """Один найденный и сегментированный объект."""
    box: List[float]        # [x0, y0, x1, y1] в пикселях
    score: float             # уверенность детекции (0..1)
    label: str                # класс / фрагмент текстового запроса
    mask: np.ndarray          # бинарная маска (H, W), значения 0/1


@dataclass
class ImageAnnotationResult:
    """Результат обработки одного изображения."""
    image: Image.Image
    detections: List[Detection] = field(default_factory=list)
    source_path: str = ""


class GroundedSegmentationPipeline:
    """
    Основной класс для получения масок объектов, соответствующих
    текстовому запросу, на одном изображении.
    """

    def __init__(self):
        self.detector = GroundingDINODetector()
        self.segmenter = SAMSegmenter()

    def warmup(self) -> None:
        """Заранее загружает веса обеих моделей (используется при старте сервера)."""
        self.detector.load()
        self.segmenter.load()

    def process_image(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
        source_path: str = "",
    ) -> ImageAnnotationResult:
        """
        Полный цикл обработки одного изображения: детекция боксов по тексту,
        затем сегментация внутри каждого бокса.
        """
        image = image.convert("RGB")

        boxes, scores, labels = self.detector.detect(
            image, text_prompt, box_threshold=box_threshold, text_threshold=text_threshold
        )

        if not boxes:
            logger.info("По запросу '%s' объекты не найдены на изображении", text_prompt)
            return ImageAnnotationResult(image=image, detections=[], source_path=source_path)

        masks = self.segmenter.segment_boxes(image, boxes)

        detections = [
            Detection(box=boxes[i], score=scores[i], label=labels[i], mask=masks[i])
            for i in range(len(boxes))
        ]
        return ImageAnnotationResult(image=image, detections=detections, source_path=source_path)
