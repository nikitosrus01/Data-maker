"""
app.py
------
Веб-интерфейс приложения на Gradio для быстрой разметки датасета
с помощью текстовых подсказок (GroundingDINO + SAM).

Запуск: python app.py
"""

import os
import traceback
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

from config import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
    DEFAULT_FRAME_INTERVAL_SEC,
    DEVICE,
    OUTPUT_DIR,
    SUPPORTED_EXPORT_FORMATS,
)
from core.pipeline import GroundedSegmentationPipeline, ImageAnnotationResult, Detection
from core.video_processor import extract_frames, VideoFrameExtractionError
from core.exporters import export_all
from utils.file_utils import create_run_dir, zip_directory, cleanup_dir
from utils.image_utils import overlay_masks_and_boxes, merge_manual_edit, resize_mask_to_image
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Модели загружаются один раз и переиспользуются между запросами всех
# пользователей интерфейса (важно: это разделяемое состояние процесса,
# не привязанное к конкретной сессии Gradio).
# ---------------------------------------------------------------------------
_pipeline: Optional[GroundedSegmentationPipeline] = None


def get_pipeline() -> GroundedSegmentationPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Первая инициализация моделей (устройство: %s)...", DEVICE)
        _pipeline = GroundedSegmentationPipeline()
    return _pipeline


def _friendly_error(exc: Exception) -> str:
    """Формирует сообщение об ошибке для пользователя интерфейса (без стектрейса)."""
    logger.error("Ошибка обработки: %s\n%s", exc, traceback.format_exc())
    return f"⚠ Произошла ошибка: {exc}"


# ---------------------------------------------------------------------------
# Вкладка 1: одно изображение
# ---------------------------------------------------------------------------
def handle_single_image(
    image: Optional[Image.Image],
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
):
    if image is None:
        return None, "Сначала загрузите изображение.", None, gr.update(choices=[], value=None)
    if not text_prompt or not text_prompt.strip():
        return None, "Введите текстовый запрос (например, 'airplane').", None, gr.update(choices=[], value=None)

    try:
        pipeline = get_pipeline()
        result = pipeline.process_image(image, text_prompt, box_threshold, text_threshold)
    except Exception as exc:  # noqa: BLE001 - хотим показать пользователю любую ошибку
        return None, _friendly_error(exc), None, gr.update(choices=[], value=None)

    if not result.detections:
        annotated = result.image
        status = f"По запросу «{text_prompt}» объекты не найдены. Попробуйте снизить пороги или изменить формулировку (на английском)."
        return annotated, status, None, gr.update(choices=[], value=None)

    annotated = overlay_masks_and_boxes(
        result.image,
        [d.mask for d in result.detections],
        [d.box for d in result.detections],
        [d.label for d in result.detections],
        [d.score for d in result.detections],
    )
    status = f"Найдено объектов: {len(result.detections)}"
    choices = [f"{i}: {d.label} ({d.score:.2f})" for i, d in enumerate(result.detections)]
    return annotated, status, result, gr.update(choices=choices, value=choices[0] if choices else None)


def refresh_single_preview(result: Optional[ImageAnnotationResult]):
    """Перерисовывает превью после ручной правки маски."""
    if result is None or not result.detections:
        return None
    return overlay_masks_and_boxes(
        result.image,
        [d.mask for d in result.detections],
        [d.box for d in result.detections],
        [d.label for d in result.detections],
        [d.score for d in result.detections],
    )


def load_mask_editor(result: Optional[ImageAnnotationResult], selected: Optional[str]):
    """Готовит редактор масок: базовое изображение + текущая маска выбранного объекта."""
    if result is None or selected is None:
        return None
    idx = int(selected.split(":")[0])
    det = result.detections[idx]

    base = np.array(result.image.convert("RGB"))
    mask_preview = base.copy()
    mask_preview[det.mask.astype(bool)] = (255, 255, 255)
    mask_preview[~det.mask.astype(bool)] = (mask_preview[~det.mask.astype(bool)] * 0.3).astype(np.uint8)

    # gr.ImageEditor принимает словарь {"background": ..., "layers": [...], "composite": ...}
    return {"background": Image.fromarray(mask_preview), "layers": [], "composite": None}


def apply_manual_edit(
    result: Optional[ImageAnnotationResult],
    selected: Optional[str],
    editor_value: dict,
):
    """
    Применяет ручную правку (кисть/ластик) к маске выбранного объекта.
    Ожидается, что пользователь рисует БЕЛЫМ цветом, чтобы добавить область
    к маске, и ЧЁРНЫМ, чтобы удалить область из маски.
    """
    if result is None or selected is None or editor_value is None:
        return result, refresh_single_preview(result)

    idx = int(selected.split(":")[0])
    det = result.detections[idx]

    composite = editor_value.get("composite")
    if composite is None:
        return result, refresh_single_preview(result)

    edited_array = np.array(composite.convert("RGB"))
    edited_array = resize_mask_to_image(edited_array, result.image.size) if edited_array.ndim == 2 else edited_array

    new_mask = merge_manual_edit(det.mask, edited_array)
    result.detections[idx] = Detection(box=det.box, score=det.score, label=det.label, mask=new_mask)

    return result, refresh_single_preview(result)


def export_single_image(result: Optional[ImageAnnotationResult], formats: List[str]):
    if result is None:
        return None, "Нет результатов для сохранения. Сначала выполните разметку."
    if not formats:
        return None, "Выберите хотя бы один формат экспорта."

    run_dir = create_run_dir("single")
    try:
        export_all([result], run_dir, formats)
        zip_path = zip_directory(run_dir, os.path.join(OUTPUT_DIR, "single_image_annotations.zip"))
        return zip_path, f"Сохранено в форматах: {', '.join(formats)}"
    except Exception as exc:  # noqa: BLE001
        return None, _friendly_error(exc)
    finally:
        cleanup_dir(run_dir)


# ---------------------------------------------------------------------------
# Вкладка 2: пакетная обработка изображений
# ---------------------------------------------------------------------------
def handle_batch_images(
    files: Optional[List],
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    formats: List[str],
    progress: gr.Progress = gr.Progress(),
):
    if not files:
        return None, "Загрузите одно или несколько изображений.", []
    if not text_prompt or not text_prompt.strip():
        return None, "Введите текстовый запрос.", []
    if not formats:
        return None, "Выберите хотя бы один формат экспорта.", []

    pipeline = get_pipeline()
    results: List[ImageAnnotationResult] = []
    previews = []
    errors = []

    for i, file_obj in enumerate(progress.tqdm(files, desc="Обработка изображений")):
        path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        try:
            image = Image.open(path).convert("RGB")
            result = pipeline.process_image(image, text_prompt, box_threshold, text_threshold, source_path=path)
            results.append(result)

            if result.detections:
                annotated = overlay_masks_and_boxes(
                    result.image,
                    [d.mask for d in result.detections],
                    [d.box for d in result.detections],
                    [d.label for d in result.detections],
                    [d.score for d in result.detections],
                )
            else:
                annotated = result.image
            previews.append((annotated, f"{os.path.basename(path)}: {len(result.detections)} объектов"))
        except Exception as exc:  # noqa: BLE001 - одна ошибка не должна прерывать всю пачку
            logger.error("Ошибка при обработке файла %s: %s", path, exc)
            errors.append(f"{os.path.basename(path)}: {exc}")

    if not results:
        return None, "Не удалось обработать ни одного файла.\n" + "\n".join(errors), []

    run_dir = create_run_dir("batch_images")
    try:
        export_all(results, run_dir, formats)
        zip_path = zip_directory(run_dir, os.path.join(OUTPUT_DIR, "batch_annotations.zip"))
    except Exception as exc:  # noqa: BLE001
        return None, _friendly_error(exc), previews
    finally:
        cleanup_dir(run_dir)

    status = f"Обработано изображений: {len(results)} из {len(files)}."
    if errors:
        status += "\nОшибки:\n" + "\n".join(errors)

    return zip_path, status, previews


# ---------------------------------------------------------------------------
# Вкладка 3: видео
# ---------------------------------------------------------------------------
def handle_video(
    video_path: Optional[str],
    interval_sec: float,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    formats: List[str],
    progress: gr.Progress = gr.Progress(),
):
    if not video_path:
        return None, "Загрузите видеофайл.", []
    if not text_prompt or not text_prompt.strip():
        return None, "Введите текстовый запрос.", []
    if not formats:
        return None, "Выберите хотя бы один формат экспорта.", []

    try:
        frames = extract_frames(video_path, interval_sec)
    except (VideoFrameExtractionError, ValueError) as exc:
        return None, _friendly_error(exc), []

    pipeline = get_pipeline()
    results: List[ImageAnnotationResult] = []
    previews = []

    for extracted in progress.tqdm(frames, desc="Разметка кадров"):
        try:
            result = pipeline.process_image(
                extracted.frame, text_prompt, box_threshold, text_threshold,
                source_path=f"t={extracted.timestamp_sec}s",
            )
            results.append(result)

            if result.detections:
                annotated = overlay_masks_and_boxes(
                    result.image,
                    [d.mask for d in result.detections],
                    [d.box for d in result.detections],
                    [d.label for d in result.detections],
                    [d.score for d in result.detections],
                )
            else:
                annotated = result.image
            previews.append((annotated, f"t={extracted.timestamp_sec}с: {len(result.detections)} объектов"))
        except Exception as exc:  # noqa: BLE001
            logger.error("Ошибка при обработке кадра t=%.2fs: %s", extracted.timestamp_sec, exc)

    if not results:
        return None, "Не удалось разметить ни одного кадра.", []

    run_dir = create_run_dir("video")
    try:
        export_all(results, run_dir, formats)
        zip_path = zip_directory(run_dir, os.path.join(OUTPUT_DIR, "video_annotations.zip"))
    except Exception as exc:  # noqa: BLE001
        return None, _friendly_error(exc), previews
    finally:
        cleanup_dir(run_dir)

    status = f"Извлечено и размечено кадров: {len(results)} (интервал {interval_sec}с)."
    return zip_path, status, previews


# ---------------------------------------------------------------------------
# Сборка интерфейса
# ---------------------------------------------------------------------------
def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Разметка датасета по текстовым подсказкам") as demo:
        gr.Markdown(
            """
            # 🏷️ Разметка датасета по текстовым подсказкам
            Автоматическая детекция и сегментация объектов на изображениях и видео
            с помощью **GroundingDINO** (детекция по тексту) и **Segment Anything** (сегментация).

            ⚠️ **Важно:** текстовые запросы работают надёжнее всего **на английском языке**
            (например, `airplane`, `car`, `person`), так как модель детекции обучена на
            англоязычных данных. Для нескольких классов сразу разделяйте их точкой:
            `airplane. car. person.`
            """
        )

        with gr.Tabs():
            # ------------------------- Вкладка: одно изображение -------------------------
            with gr.Tab("Одно изображение"):
                state_result = gr.State(value=None)

                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(type="pil", label="Изображение")
                        prompt = gr.Textbox(label="Текстовый запрос", placeholder="airplane")
                        with gr.Row():
                            box_thr = gr.Slider(0.05, 0.95, value=DEFAULT_BOX_THRESHOLD, step=0.05, label="Порог детекции (box)")
                            text_thr = gr.Slider(0.05, 0.95, value=DEFAULT_TEXT_THRESHOLD, step=0.05, label="Порог соответствия тексту")
                        run_btn = gr.Button("🔍 Разметить", variant="primary")
                        status_box = gr.Textbox(label="Статус", interactive=False)

                    with gr.Column(scale=1):
                        output_preview = gr.Image(type="pil", label="Результат", interactive=False)
                        object_selector = gr.Dropdown(label="Выбрать объект для ручной правки", choices=[], interactive=True)

                gr.Markdown(
                    "### Ручная коррекция маски выбранного объекта\n"
                    "Рисуйте **белым**, чтобы добавить область к маске, "
                    "**чёрным**, чтобы удалить область из маски, затем нажмите «Применить правку»."
                )
                with gr.Row():
                    mask_editor = gr.ImageEditor(label="Редактор маски", type="pil")
                    apply_edit_btn = gr.Button("✏️ Применить правку")

                with gr.Row():
                    export_formats_single = gr.CheckboxGroup(
                        SUPPORTED_EXPORT_FORMATS, value=["COCO"], label="Форматы экспорта"
                    )
                    export_btn_single = gr.Button("💾 Сохранить разметку")
                download_single = gr.File(label="Скачать архив с разметкой")

                run_btn.click(
                    handle_single_image,
                    inputs=[input_image, prompt, box_thr, text_thr],
                    outputs=[output_preview, status_box, state_result, object_selector],
                )
                object_selector.change(
                    load_mask_editor, inputs=[state_result, object_selector], outputs=[mask_editor]
                )
                apply_edit_btn.click(
                    apply_manual_edit,
                    inputs=[state_result, object_selector, mask_editor],
                    outputs=[state_result, output_preview],
                )
                export_btn_single.click(
                    export_single_image,
                    inputs=[state_result, export_formats_single],
                    outputs=[download_single, status_box],
                )

            # ------------------------- Вкладка: пакетная обработка -------------------------
            with gr.Tab("Пакетная обработка изображений"):
                gr.Markdown(
                    "Загрузите несколько изображений — ко всем будет применён один и тот же "
                    "текстовый запрос. Ручное редактирование масок в пакетном режиме не поддерживается: "
                    "используйте вкладку «Одно изображение» для точечной коррекции."
                )
                batch_files = gr.Files(label="Изображения", file_types=["image"])
                batch_prompt = gr.Textbox(label="Текстовый запрос", placeholder="car. person.")
                with gr.Row():
                    batch_box_thr = gr.Slider(0.05, 0.95, value=DEFAULT_BOX_THRESHOLD, step=0.05, label="Порог детекции (box)")
                    batch_text_thr = gr.Slider(0.05, 0.95, value=DEFAULT_TEXT_THRESHOLD, step=0.05, label="Порог соответствия тексту")
                batch_formats = gr.CheckboxGroup(SUPPORTED_EXPORT_FORMATS, value=["COCO"], label="Форматы экспорта")
                batch_run_btn = gr.Button("🔍 Обработать пачку", variant="primary")
                batch_status = gr.Textbox(label="Статус", interactive=False)
                batch_download = gr.File(label="Скачать архив с разметкой")
                batch_gallery = gr.Gallery(label="Превью результатов", columns=4)

                batch_run_btn.click(
                    handle_batch_images,
                    inputs=[batch_files, batch_prompt, batch_box_thr, batch_text_thr, batch_formats],
                    outputs=[batch_download, batch_status, batch_gallery],
                )

            # ------------------------- Вкладка: видео -------------------------
            with gr.Tab("Видео"):
                gr.Markdown(
                    "Кадры извлекаются равномерно с заданным интервалом (в секундах), "
                    "начиная с начала видео, и размечаются тем же способом, что и изображения."
                )
                video_input = gr.Video(label="Видеофайл")
                interval_input = gr.Number(value=DEFAULT_FRAME_INTERVAL_SEC, label="Интервал извлечения кадров (сек)", precision=1)
                video_prompt = gr.Textbox(label="Текстовый запрос", placeholder="airplane")
                with gr.Row():
                    video_box_thr = gr.Slider(0.05, 0.95, value=DEFAULT_BOX_THRESHOLD, step=0.05, label="Порог детекции (box)")
                    video_text_thr = gr.Slider(0.05, 0.95, value=DEFAULT_TEXT_THRESHOLD, step=0.05, label="Порог соответствия тексту")
                video_formats = gr.CheckboxGroup(SUPPORTED_EXPORT_FORMATS, value=["COCO"], label="Форматы экспорта")
                video_run_btn = gr.Button("🔍 Извлечь и разметить кадры", variant="primary")
                video_status = gr.Textbox(label="Статус", interactive=False)
                video_download = gr.File(label="Скачать архив с разметкой")
                video_gallery = gr.Gallery(label="Превью размеченных кадров", columns=4)

                video_run_btn.click(
                    handle_video,
                    inputs=[video_input, interval_input, video_prompt, video_box_thr, video_text_thr, video_formats],
                    outputs=[video_download, video_status, video_gallery],
                )

        gr.Markdown(
            f"---\nУстройство вычислений: **{DEVICE.upper()}**"
            + ("" if DEVICE == "cuda" else " ⚠️ GPU не обнаружен, обработка будет заметно медленнее.")
        )

    return demo


if __name__ == "__main__":
    interface = build_interface()
    # Предзагрузка моделей при старте — первый запрос пользователя не будет
    # включать долгую загрузку весов. Можно закомментировать для более
    # быстрого старта сервера, если веса будут грузиться лениво.
    try:
        get_pipeline().warmup()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не удалось предзагрузить модели при старте: %s. "
                        "Модели будут загружены при первом запросе.", exc)

    interface.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
