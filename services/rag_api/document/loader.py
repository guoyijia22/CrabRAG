from __future__ import annotations

from pathlib import Path

from charset_normalizer import from_bytes
from docx import Document

from services.rag_api.document.cleaner import clean_text
from services.rag_api.exceptions import DOC_LOAD_ERROR_MESSAGE, DocumentLoadError


SUPPORTED_SUFFIXES = {".docx", ".txt", ".pdf"}


def read_txt_file(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    match = from_bytes(data).best()
    if match is None:
        raise DocumentLoadError(DOC_LOAD_ERROR_MESSAGE)
    return str(match)


def read_docx_file(path: Path) -> str:
    document = Document(str(path))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    table_cells: list[str] = []
    for table in document.tables:
        for row in table.rows:
            table_cells.append(" | ".join(cell.text.strip() for cell in row.cells if cell.text.strip()))
    return "\n".join(paragraphs + table_cells)


def read_pdf_file(path: Path) -> str:
    pymupdf = _load_pymupdf()
    document = None
    try:
        document = pymupdf.open(str(path))
        text = "\n".join((page.get_text("text") or "").strip() for page in document)
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoadError(DOC_LOAD_ERROR_MESSAGE) from exc
    finally:
        if document is not None:
            document.close()
    if not text.strip():
        raise DocumentLoadError(DOC_LOAD_ERROR_MESSAGE)
    return text


def _load_pymupdf():
    try:
        import pymupdf  # type: ignore

        return pymupdf
    except Exception:
        pass
    try:
        import fitz  # type: ignore

        if not hasattr(fitz, "open"):
            raise RuntimeError("imported fitz is not PyMuPDF")
        return fitz
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoadError(DOC_LOAD_ERROR_MESSAGE) from exc


def load_documents(kb_dir: Path) -> list[dict]:
    if not kb_dir.exists() or not kb_dir.is_dir():
        raise DocumentLoadError(DOC_LOAD_ERROR_MESSAGE)
    files = sorted(path for path in kb_dir.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        raise DocumentLoadError(DOC_LOAD_ERROR_MESSAGE)
    documents: list[dict] = []
    for path in files:
        try:
            if path.suffix.lower() == ".docx":
                raw = read_docx_file(path)
            elif path.suffix.lower() == ".pdf":
                raw = read_pdf_file(path)
            else:
                raw = read_txt_file(path)
            content = clean_text(raw)
            if content:
                documents.append({"source_file": path.name, "source_path": str(path), "content": content})
        except Exception as exc:  # noqa: BLE001
            raise DocumentLoadError(DOC_LOAD_ERROR_MESSAGE) from exc
    if not documents:
        raise DocumentLoadError(DOC_LOAD_ERROR_MESSAGE)
    return documents
