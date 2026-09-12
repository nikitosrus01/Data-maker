"""
segmenter.py
------------
Обёртка над Segment Anything Model (SAM) для получения масок сегментации
по bounding box, найденным GroundingDINO (режим "box prompt").

Используется реализация из HuggingFace Transformers (SamModel + SamProcessor).
"""

from typing import List

import numpy as np
import torch
from PIL import Image

from config import DEVICE, SAM_MODEL_ID
from utils.logger import get_logger

logger = get_logger(__name__)


class SAMSegmenter:
    """Сегментатор объектов на базе Segment Anything, с промптом в виде box."""

    def __init__(self, model_id: str = SAM_MODEL_ID, device: str = DEVICE):
        self.device = device
        self.model_id = model_id
        self._processor = None
        self._model = None
        logger.info("Инициализация SAMSegmenter (модель=%s, устройство=%s)", model_id, device)

    def load(self) -> None:
        """Ленивая загрузка весов (см. комментарий в detector.py)."""
        if self._model is not None:
            return

        try:
            from transformers import SamModel, SamProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Библиотека transformers не установлена или не содержит SAM. "
                "Выполните: pip install -U transformers"
            ) from exc

        logger.info("Загрузка весов SAM: %s", self.model_id)
        self._processor = SamProcessor.from_pretrained(self.model_id)
        self._model = SamModel.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()
        logger.info("SAM загружен успешно")

    def segment_boxes(self, image: Image.Image, boxes: List[List[float]]) -> List[np.ndarray]:
        """
        Принимает изображение и список bounding box в пиксельных координатах,
        возвращает список бинарных масок (по одной на каждый box), выбирая
        для каждого box маску с наивысшей предсказанной IoU-оценкой.

        Пустой входной список boxes -> пустой список масок (штатный случай:
        по тексту ничего не найдено).
        """
        self.load()

        if len(boxes) == 0:
            return []

        # SAM в этой реализации ожидает боксы, сгруппированные по изображению:
        # List[List[List[float]]] -> для одного изображения это [boxes]
        inputs = self._processor(image, input_boxes=[boxes], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        masks = self._processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        # masks[0].shape == (num_boxes, num_candidate_masks, H, W)
        iou_scores = outputs.iou_scores.cpu()  # (batch=1, num_boxes, num_candidate_masks)

        per_box_masks = masks[0]
        per_box_scores = iou_scores[0]

        result_masks = []
        for box_idx in range(per_box_masks.shape[0]):
            best_candidate = int(torch.argmax(per_box_scores[box_idx]).item())
            mask = per_box_masks[box_idx, best_candidate].numpy().astype(np.uint8)
            result_masks.append(mask)

        logger.info("Сегментация завершена для %d боксов", len(boxes))
        return result_masks
