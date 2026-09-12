"""
exporters.py
-------------
Сохранение результатов разметки в трёх популярных форматах:

  - COCO JSON     - один общий файл annotations_coco.json на весь датасет,
                    включает и bounding box, и полигоны сегментации.
  - YOLO txt      - по одному .txt файлу на изображение (только bounding box,
                    формат YOLO не хранит полигоны сегментации в базовой версии).
  - Pascal VOC    - по одному .xml файлу на изображение (только bounding box).

Важное ограничение: и YOLO (детекционный формат), и классический Pascal VOC
не поддерживают произвольные полигоны сегментации в "коробочном" виде,
поэтому для этих двух форматов сохраняются только bounding box, а маски
экспортируются только через COCO (segmentation) и как PNG-превью.
"""

import json
import os
import xml.etree.ElementTree as ET
from typing import Dict, List
from xml.dom import minidom

from PIL import Image

from core.pipeline import ImageAnnotationResult
from utils.image_utils import mask_bbox, mask_to_polygons
from utils.logger import get_logger

logger = get_logger(__name__)


def _build_class_index(results: List[ImageAnnotationResult]) -> Dict[str, int]:
    """Строит стабильное отображение "класс -> целочисленный id" по всем результатам."""
    classes = sorted({det.label for r in results for det in r.detections})
    return {name: idx for idx, name in enumerate(classes)}


# ---------------------------------------------------------------------------
# COCO
# ---------------------------------------------------------------------------
def export_coco(results: List[ImageAnnotationResult], output_dir: str) -> str:
    """
    Сохраняет один файл annotations_coco.json в формате COCO
    (images / annotations / categories), а также сами изображения в output_dir/images.
    """
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    class_index = _build_class_index(results)
    categories = [{"id": idx, "name": name} for name, idx in class_index.items()]

    coco = {"images": [], "annotations": [], "categories": categories}
    ann_id = 1

    for img_id, result in enumerate(results, start=1):
        filename = f"image_{img_id:06d}.jpg"
        result.image.convert("RGB").save(os.path.join(images_dir, filename), quality=95)

        width, height = result.image.size
        coco["images"].append({
            "id": img_id, "file_name": filename, "width": width, "height": height,
        })

        for det in result.detections:
            polygons = mask_to_polygons(det.mask)
            if not polygons:
                # Маска пуста после ручного редактирования (пользователь стёр
                # весь объект) — пропускаем, чтобы не писать невалидные аннотации.
                continue

            x0, y0, x1, y1 = mask_bbox(det.mask)
            bbox_w, bbox_h = max(0, x1 - x0), max(0, y1 - y0)
            area = float((det.mask > 0).sum())

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_index[det.label],
                "segmentation": polygons,
                "bbox": [x0, y0, bbox_w, bbox_h],
                "area": area,
                "iscrowd": 0,
                "score": det.score,
            })
            ann_id += 1

    out_path = os.path.join(output_dir, "annotations_coco.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    logger.info("COCO-аннотации сохранены: %s (изображений=%d, объектов=%d)",
                out_path, len(results), ann_id - 1)
    return out_path


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------
def export_yolo(results: List[ImageAnnotationResult], output_dir: str) -> str:
    """
    Сохраняет по одному .txt файлу на изображение в формате YOLO:
    "class_id x_center y_center width height" (все значения нормализованы 0..1).
    Дополнительно создаёт classes.txt с именами классов по порядку id.
    """
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    class_index = _build_class_index(results)

    for img_id, result in enumerate(results, start=1):
        filename_base = f"image_{img_id:06d}"
        result.image.convert("RGB").save(os.path.join(images_dir, f"{filename_base}.jpg"), quality=95)

        width, height = result.image.size
        lines = []
        for det in result.detections:
            x0, y0, x1, y1 = mask_bbox(det.mask)
            if x1 <= x0 or y1 <= y0:
                continue  # пустая маска после ручной правки

            x_center = ((x0 + x1) / 2) / width
            y_center = ((y0 + y1) / 2) / height
            box_w = (x1 - x0) / width
            box_h = (y1 - y0) / height

            class_id = class_index[det.label]
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        with open(os.path.join(labels_dir, f"{filename_base}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    with open(os.path.join(output_dir, "classes.txt"), "w", encoding="utf-8") as f:
        for name, _ in sorted(class_index.items(), key=lambda kv: kv[1]):
            f.write(name + "\n")

    logger.info("YOLO-аннотации сохранены в %s (изображений=%d)", labels_dir, len(results))
    return labels_dir


# ---------------------------------------------------------------------------
# Pascal VOC
# ---------------------------------------------------------------------------
def export_voc(results: List[ImageAnnotationResult], output_dir: str) -> str:
    """Сохраняет по одному .xml файлу на изображение в формате Pascal VOC."""
    images_dir = os.path.join(output_dir, "images")
    annotations_dir = os.path.join(output_dir, "Annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    for img_id, result in enumerate(results, start=1):
        filename_base = f"image_{img_id:06d}"
        image_filename = f"{filename_base}.jpg"
        result.image.convert("RGB").save(os.path.join(images_dir, image_filename), quality=95)

        width, height = result.image.size

        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "filename").text = image_filename

        size_el = ET.SubElement(annotation, "size")
        ET.SubElement(size_el, "width").text = str(width)
        ET.SubElement(size_el, "height").text = str(height)
        ET.SubElement(size_el, "depth").text = "3"

        for det in result.detections:
            x0, y0, x1, y1 = mask_bbox(det.mask)
            if x1 <= x0 or y1 <= y0:
                continue

            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = det.label
            ET.SubElement(obj, "confidence").text = f"{det.score:.4f}"
            bbox_el = ET.SubElement(obj, "bndbox")
            ET.SubElement(bbox_el, "xmin").text = str(x0)
            ET.SubElement(bbox_el, "ymin").text = str(y0)
            ET.SubElement(bbox_el, "xmax").text = str(x1)
            ET.SubElement(bbox_el, "ymax").text = str(y1)

        xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="  ")
        with open(os.path.join(annotations_dir, f"{filename_base}.xml"), "w", encoding="utf-8") as f:
            f.write(xml_str)

    logger.info("Pascal VOC аннотации сохранены в %s (изображений=%d)", annotations_dir, len(results))
    return annotations_dir


EXPORTERS = {
    "COCO": export_coco,
    "YOLO": export_yolo,
    "Pascal VOC": export_voc,
}


def export_all(results: List[ImageAnnotationResult], output_dir: str, formats: List[str]) -> List[str]:
    """Запускает все выбранные пользователем экспортёры и возвращает список путей результатов."""
    paths = []
    for fmt in formats:
        exporter = EXPORTERS.get(fmt)
        if exporter is None:
            logger.warning("Неизвестный формат экспорта: %s, пропускаем", fmt)
            continue
        paths.append(exporter(results, output_dir))
    return paths
