from __future__ import annotations

from pathlib import Path

OCR_DISABLED_MESSAGE = "OCR未启用：当前系统仅预留多模态OCR扩展接口，默认不解析扫描图片。"


def ocr_image_to_text(path: Path) -> str:
    if not path.exists():
        return OCR_DISABLED_MESSAGE
    return OCR_DISABLED_MESSAGE


def ocr_pdf_pages(path: Path) -> list[str]:
    if not path.exists():
        return []
    return []
