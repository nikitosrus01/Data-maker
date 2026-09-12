"""
detector.py
-----------
Обёртка над GroundingDINO для детекции объектов по произвольному
текстовому запросу ("открытый словарь", open-vocabulary detection).

Используется реализация из библиотеки HuggingFace Transformers
(AutoModelForZeroShotObjectDetection), что избавляет от необходимости
собирать кастомные CUDA-расширения оригинального репозитория GroundingDINO.

ВАЖНО (ограничение): GroundingDINO чувствителен к формулировке запроса.
  - Запрос должен быть на английском языке для лучшего качества (модель
    обучена на англоязычных парах "изображение-текст"); русскоязычные
    запросы могут детектироваться значительно хуже или не детектироваться
    вовсе. Это существенное ограничение, о котором нужно явно предупредить
    пользователя в интерфейсе.
  - Несколько классов в одном запросе разделяются точкой, например:
    "airplane. car. person." — так и детектор, и маркировка результатов
    по классам будут работать корректнее, чем с одной фразой без разделителей.
"""

from typing import List, Tuple

import torch
from PIL import Image

from config import DEVICE, GROUNDING_DINO_MODEL_ID, MAX_DETECTIONS_PER_IMAGE
from utils.logger import get_logger

logger = get_logger(__name__)


class GroundingDINODetector:
    """Детектор объектов по тексту на базе GroundingDINO."""

    def __init__(self, model_id: str = GROUNDING_DINO_MODEL_ID, device: str = DEVICE):
        self.device = device
        self.model_id = model_id
        self._processor = None
        self._model = None
        logger.info("Инициализация GroundingDINODetector (модель=%s, устройство=%s)", model_id, device)

    def load(self) -> None:
        """
        Ленивая загрузка весов модели. Загрузка вынесена из __init__,
        чтобы приложение могло запуститься и показать интерфейс, даже если
        веса ещё не скачаны / сеть временно недоступна — ошибка будет
        показана пользователю только в момент первого запроса, а не при
        старте сервера.
        """
        if self._model is not None:
            return

        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except ImportError as exc:
            raise RuntimeError(
                "Библиотека transformers не установлена или установлена без "
                "поддержки GroundingDINO. Выполните: pip install -U transformers"
            ) from exc

        logger.info("Загрузка весов GroundingDINO: %s", self.model_id)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()
        logger.info("GroundingDINO загружен успешно")

    @staticmethod
    def _normalize_prompt(text_prompt: str) -> str:
        """
        GroundingDINO ожидает запрос в нижнем регистре, с точкой в конце
        каждого класса. Приводим пользовательский ввод к этому формату.
        """
        text = text_prompt.strip().lower()
        if not text:
            raise ValueError("Текстовый запрос не может быть пустым")
        if not text.endswith("."):
            text += "."
        return text

    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Tuple[List[List[float]], List[float], List[str]]:
        """
        Возвращает три списка одинаковой длины:
          boxes  - [[x0, y0, x1, y1], ...] в пиксельных координатах исходного изображения
          scores - уверенность модели для каждого бокса
          labels - фрагмент текста запроса, которому соответствует детекция

        Пустые списки означают, что по заданному запросу и порогам ничего
        не найдено — это штатная ситуация, а не ошибка.
        """
        self.load()

        normalized_prompt = self._normalize_prompt(text_prompt)

        inputs = self._processor(images=image, text=normalized_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # В разных версиях transformers параметр порога детекции по боксу
        # назывался по-разному:
        #   - старые версии:  box_threshold=...
        #   - новые версии (см. PR transformers, переименование в 2024):
        #                     threshold=...
        # Пробуем оба варианта, чтобы не привязываться к конкретной версии
        # библиотеки, установленной у пользователя.
        post_process_kwargs = dict(
            input_ids=inputs["input_ids"],
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],  # (height, width)
        )
        try:
            results = self._processor.post_process_grounded_object_detection(
                outputs, box_threshold=box_threshold, **post_process_kwargs
            )[0]
        except TypeError:
            try:
                results = self._processor.post_process_grounded_object_detection(
                    outputs, threshold=box_threshold, **post_process_kwargs
                )[0]
            except TypeError as exc:
                raise RuntimeError(
                    "Несовместимая версия transformers для post-обработки GroundingDINO: "
                    "не подходит ни 'box_threshold', ни 'threshold'. "
                    "Обновите библиотеку: pip install -U transformers. "
                    f"Исходная ошибка: {exc}"
                ) from exc

        boxes = results["boxes"].cpu().tolist()
        scores = results["scores"].cpu().tolist()

        # В разных версиях ключ с текстовыми метками может называться
        # "labels" или "text_labels" — проверяем оба варианта.
        labels = results.get("text_labels") or results.get("labels") or [text_prompt] * len(boxes)
        if isinstance(labels, torch.Tensor):
            labels = [str(l) for l in labels.tolist()]

        if len(boxes) > MAX_DETECTIONS_PER_IMAGE:
            logger.warning(
                "Найдено %d объектов, что превышает лимит MAX_DETECTIONS_PER_IMAGE=%d. "
                "Оставляем %d объектов с наибольшей уверенностью. Возможно, запрос "
                "слишком общий (например, одно слово, соответствующее фону).",
                len(boxes), MAX_DETECTIONS_PER_IMAGE, MAX_DETECTIONS_PER_IMAGE,
            )
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            order = order[:MAX_DETECTIONS_PER_IMAGE]
            boxes = [boxes[i] for i in order]
            scores = [scores[i] for i in order]
            labels = [labels[i] for i in order]

        logger.info("Детекция завершена: найдено %d объектов по запросу '%s'", len(boxes), text_prompt)
        return boxes, scores, labels
