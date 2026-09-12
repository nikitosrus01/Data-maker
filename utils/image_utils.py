"""
image_utils.py
---------------
Функции для:
  - наложения масок и bounding box на изображение (визуализация);
  - конвертации бинарной маски в полигон (для формата COCO);
  - слияния маски, полученной от модели, с ручными правками пользователя
    (кисть/ластик), нарисованными поверх в редакторе.
"""

from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from utils.logger import get_logger

logger = get_logger(__name__)

# Фиксированная, но разнообразная палитра для инстансов, чтобы соседние
# объекты визуально не сливались друг с другом.
_PALETTE = [
    (255, 99, 71), (60, 179, 113), (65, 105, 225), (255, 215, 0),
    (238, 130, 238), (0, 206, 209), (255, 140, 0), (154, 205, 50),
    (219, 112, 147), (72, 61, 139),
]


def _color_for_index(idx: int) -> Tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


def overlay_masks_and_boxes(
    image: Image.Image,
    masks: List[np.ndarray],
    boxes: List[List[float]],
    labels: List[str],
    scores: List[float],
    mask_alpha: float = 0.45,
) -> Image.Image:
    """
    Возвращает копию изображения с наложенными полупрозрачными масками,
    bounding box и подписями (класс + уверенность модели).

    masks  - список бинарных масок формы (H, W), значения {0, 1}
    boxes  - список [x0, y0, x1, y1] в пиксельных координатах
    labels - текстовые подписи для каждого объекта
    scores - уверенность детекции для каждого объекта
    """
    base = np.array(image.convert("RGB"))
    overlay = base.copy()

    for i, mask in enumerate(masks):
        color = _color_for_index(i)
        colored_mask = np.zeros_like(base)
        colored_mask[mask.astype(bool)] = color
        overlay = np.where(
            mask[..., None].astype(bool),
            (overlay * (1 - mask_alpha) + colored_mask * mask_alpha).astype(np.uint8),
            overlay,
        )

    result = overlay.copy()
    for i, box in enumerate(boxes):
        color = _color_for_index(i)
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        cv2.rectangle(result, (x0, y0), (x1, y1), color, 2)

        label_text = f"{labels[i]} {scores[i]:.2f}" if i < len(labels) else ""
        if label_text:
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x0, max(0, y0 - th - 6)), (x0 + tw + 4, y0), color, -1)
            cv2.putText(
                result, label_text, (x0 + 2, max(12, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

    return Image.fromarray(result)


def mask_to_polygons(mask: np.ndarray, min_area: float = 4.0) -> List[List[float]]:
    """
    Конвертирует бинарную маску в список полигонов формата COCO
    ([x1, y1, x2, y2, ...]). Один объект после ручного редактирования
    иногда распадается на несколько несвязных областей — возвращаем все,
    что больше min_area, отбрасывая шумовые вкрапления.
    """
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        contour = contour.flatten().tolist()
        if len(contour) >= 6:  # минимум 3 точки (x, y) для валидного полигона
            polygons.append(contour)
    return polygons


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Вычисляет bounding box [x0, y0, x1, y1] для бинарной маски."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def merge_manual_edit(
    original_mask: np.ndarray,
    edited_layer: np.ndarray,
    add_color_threshold: int = 200,
    erase_color_threshold: int = 50,
) -> np.ndarray:
    """
    Объединяет автоматическую маску с ручными правками пользователя.

    Ожидается, что edited_layer - это RGBA/RGB слой из графического редактора,
    где:
      - светлые (белые) мазки кисти  -> добавить область к маске;
      - тёмные (чёрные) мазки ластика -> убрать область из маски.

    Такой подход прост и предсказуем для пользователя, но требует, чтобы
    UI использовал именно белую кисть и чёрный ластик (см. app.py).
    Всё остальное (нейтральный/прозрачный фон) не изменяет исходную маску.
    """
    if edited_layer.ndim == 3:
        gray = cv2.cvtColor(edited_layer[..., :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = edited_layer

    result = original_mask.copy().astype(bool)

    add_region = gray >= add_color_threshold
    erase_region = gray <= erase_color_threshold

    result[add_region] = True
    result[erase_region] = False

    return result.astype(np.uint8)


def resize_mask_to_image(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    target_size = (width, height). Приводит маску к размеру изображения,
    например после того, как ручное редактирование велось на превью
    меньшего разрешения.
    """
    w, h = target_size
    if mask.shape[:2] == (h, w):
        return mask
    resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return resized
