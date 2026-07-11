from __future__ import annotations

import base64
import binascii
import json
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from services.rag_api.paths import docs_dirs_env_value, resolve_project_root

PROJECT_ROOT = resolve_project_root(Path(__file__).resolve().parents[2])
APP_SETTINGS_PATH = PROJECT_ROOT / "data" / "app_settings.json"
CONFIG_MD_PATH = PROJECT_ROOT / "Config.md"
SIDEBAR_IMAGE_PATH = PROJECT_ROOT / "data" / "ui" / "sidebar-image.bin"
SIDEBAR_IMAGE_META_PATH = PROJECT_ROOT / "data" / "ui" / "sidebar-image.json"
SIDEBAR_IMAGE_URL = "/api/app-assets/sidebar-image"
MAX_SIDEBAR_IMAGE_BYTES = 10 * 1024 * 1024

UiTheme = Literal["red_white", "blue_white", "classic_green"]
UiLanguage = Literal["en", "zh"]

DEFAULT_SYSTEM_NAME = "CrabRAG"
LEGACY_DEFAULT_SYSTEM_NAMES = {
    "QueryBaseLab 通用基础查询",
    "QueryBasePortableLab 通用基础查询",
    "CrabRAG 通用基础查询",
}
DEFAULT_KNOWLEDGE_BASE_NAME = "通用基础查询知识库"
DEFAULT_UI_THEME: UiTheme = "red_white"
DEFAULT_UI_LANGUAGE: UiLanguage = "en"
DEFAULT_NO_MATCH_RESPONSE = "暂无相关知识库依据，无法为您解答"
DEFAULT_OUT_OF_SCOPE_RESPONSE = "当前问题不属于本系统配置的查询范围，无法为您解答。"

DEFAULT_BUSINESS_SCOPE_DESCRIPTION = (
    "General knowledge base assistant for local documents. "
    "通用本地知识库助手，用于基于已配置文档进行检索、问答、知识图谱分析和可溯源回答。"
)

DEFAULT_IN_SCOPE_KEYWORDS: list[str] = []

DEFAULT_OUT_OF_SCOPE_KEYWORDS = ["股票", "Stock"]


def default_knowledge_base_dirs() -> list[str]:
    env_value = docs_dirs_env_value()
    values = [item.strip() for item in env_value.split(";") if item.strip()] if env_value else []
    return _normalize_dirs(values)


class AppSettings(BaseModel):
    system_name: str = Field(default=DEFAULT_SYSTEM_NAME, min_length=4, max_length=40)
    knowledge_base_name: str = Field(default=DEFAULT_KNOWLEDGE_BASE_NAME, min_length=2, max_length=30)
    ui_theme: UiTheme = DEFAULT_UI_THEME
    ui_language: UiLanguage = DEFAULT_UI_LANGUAGE
    sidebar_image_url: str = ""
    knowledge_base_dirs: list[str] = Field(default_factory=default_knowledge_base_dirs)
    common_questions: list[str] = Field(default_factory=list, max_length=10)
    business_scope_description: str = DEFAULT_BUSINESS_SCOPE_DESCRIPTION
    in_scope_keywords: list[str] = Field(default_factory=lambda: DEFAULT_IN_SCOPE_KEYWORDS.copy())
    out_of_scope_keywords: list[str] = Field(default_factory=lambda: DEFAULT_OUT_OF_SCOPE_KEYWORDS.copy())
    scope_min_score: float = Field(default=0, ge=0.0, le=1.0)
    out_of_scope_response: str = DEFAULT_OUT_OF_SCOPE_RESPONSE
    no_match_response: str = DEFAULT_NO_MATCH_RESPONSE

    @field_validator("common_questions", "in_scope_keywords", "out_of_scope_keywords", mode="after")
    @classmethod
    def _normalize_list(cls, values: list[str], info: ValidationInfo) -> list[str]:
        result: list[str] = []
        for item in values:
            value = str(item).strip()
            if value and value not in result:
                result.append(value)
        return result[:10] if info.field_name == "common_questions" else result

    @field_validator("knowledge_base_dirs", mode="after")
    @classmethod
    def _normalize_knowledge_base_dirs(cls, values: list[str]) -> list[str]:
        return _normalize_dirs(values)

    @field_validator("business_scope_description", "out_of_scope_response", "no_match_response", mode="after")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized


class SidebarImageUpload(BaseModel):
    filename: str
    content_type: str = ""
    data_base64: str


def load_app_settings() -> AppSettings:
    if APP_SETTINGS_PATH.exists():
        try:
            settings = AppSettings.model_validate_json(APP_SETTINGS_PATH.read_text(encoding="utf-8"))
            upgraded = _upgrade_legacy_system_name(settings)
            if upgraded != settings:
                return save_app_settings(upgraded)
            return settings
        except Exception:
            return AppSettings()
    migrated = _migrate_from_config_md()
    return save_app_settings(migrated)


def save_app_settings(settings: AppSettings) -> AppSettings:
    normalized = AppSettings.model_validate(settings.model_dump())
    APP_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    APP_SETTINGS_PATH.write_text(json.dumps(normalized.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def save_sidebar_image(upload: SidebarImageUpload) -> AppSettings:
    data = _decode_sidebar_image(upload.data_base64)
    content_type = _validate_sidebar_image(upload.filename, data)
    SIDEBAR_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SIDEBAR_IMAGE_PATH.write_bytes(data)
    SIDEBAR_IMAGE_META_PATH.write_text(
        json.dumps(
            {
                "filename": Path(upload.filename).name,
                "content_type": content_type,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    settings = load_app_settings().model_copy(update={"sidebar_image_url": SIDEBAR_IMAGE_URL})
    return save_app_settings(settings)


def read_sidebar_image_asset() -> tuple[bytes, str]:
    if not SIDEBAR_IMAGE_PATH.exists():
        raise HTTPException(status_code=404, detail="未配置侧边栏展示图片")
    data = SIDEBAR_IMAGE_PATH.read_bytes()
    content_type = _read_sidebar_image_content_type(data)
    return data, content_type


def read_public_app_config() -> dict:
    settings = load_app_settings()
    return {
        "system_name": settings.system_name,
        "knowledge_base_name": settings.knowledge_base_name,
        "ui_theme": settings.ui_theme,
        "ui_language": settings.ui_language,
        "sidebar_image_url": settings.sidebar_image_url,
        "knowledge_base_dirs": settings.knowledge_base_dirs,
        "common_questions": settings.common_questions,
    }


def update_system_name(name: str) -> str:
    settings = load_app_settings().model_copy(update={"system_name": name.strip()})
    return save_app_settings(settings).system_name


def update_knowledge_base_name(name: str) -> str:
    settings = load_app_settings().model_copy(update={"knowledge_base_name": name.strip()})
    return save_app_settings(settings).knowledge_base_name


def update_common_questions(questions: list[str]) -> list[str]:
    settings = load_app_settings().model_copy(update={"common_questions": questions})
    return save_app_settings(settings).common_questions


def effective_knowledge_base_dirs(settings: AppSettings | None = None) -> list[Path]:
    active = settings or load_app_settings()
    return [Path(item).resolve() for item in active.knowledge_base_dirs]


def _migrate_from_config_md() -> AppSettings:
    if not CONFIG_MD_PATH.exists():
        return AppSettings()
    try:
        text = CONFIG_MD_PATH.read_text(encoding="utf-8")
    except OSError:
        return AppSettings()
    return AppSettings(
        system_name=_normalize_legacy_system_name(_parse_scalar(text, "system_name") or DEFAULT_SYSTEM_NAME),
        knowledge_base_name=_parse_scalar(text, "knowledge_base_name") or DEFAULT_KNOWLEDGE_BASE_NAME,
        common_questions=_parse_common_questions(text),
    )


def _upgrade_legacy_system_name(settings: AppSettings) -> AppSettings:
    normalized = _normalize_legacy_system_name(settings.system_name)
    if normalized == settings.system_name:
        return settings
    return settings.model_copy(update={"system_name": normalized})


def _normalize_legacy_system_name(name: str) -> str:
    return DEFAULT_SYSTEM_NAME if name in LEGACY_DEFAULT_SYSTEM_NAMES else name


def _parse_scalar(text: str, key: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(f"{key}:"):
            continue
        _, _, value = stripped.partition(":")
        return value.strip().strip("\"'")
    return ""


def _parse_common_questions(text: str) -> list[str]:
    questions: list[str] = []
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if not in_block:
            if stripped == "common_questions:":
                in_block = True
            continue
        if not stripped:
            continue
        if not stripped.startswith("- "):
            break
        question = stripped[2:].strip().strip("\"'")
        if question and question not in questions:
            questions.append(question)
    return questions[:10]


def _normalize_dirs(values: list[str]) -> list[str]:
    result: list[str] = []
    for item in values:
        value = str(item).strip()
        if not value:
            continue
        normalized = str(Path(value).expanduser().resolve())
        if normalized not in result:
            result.append(normalized)
    if result:
        return result
    return []


def _decode_sidebar_image(data_base64: str) -> bytes:
    value = data_base64.strip()
    if "," in value:
        value = value.split(",", 1)[1]
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("图片数据不是有效的 base64") from exc
    if len(data) > MAX_SIDEBAR_IMAGE_BYTES:
        raise ValueError("图片大小不能超过 10 MB")
    return data


def _validate_sidebar_image(filename: str, data: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in {".gif", ".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".jprg", ".webp", ".bmp"}:
        raise ValueError("不支持的图片格式")
    content_type = _detect_image_content_type(data)
    expected = _expected_content_type_for_extension(extension)
    if not content_type or content_type != expected:
        raise ValueError("图片格式与文件扩展名不匹配或文件已损坏")
    return content_type


def _read_sidebar_image_content_type(data: bytes) -> str:
    content_type = ""
    if SIDEBAR_IMAGE_META_PATH.exists():
        try:
            meta = json.loads(SIDEBAR_IMAGE_META_PATH.read_text(encoding="utf-8"))
            content_type = str(meta.get("content_type") or "")
        except Exception:
            content_type = ""
    return content_type or _detect_image_content_type(data) or "application/octet-stream"


def _expected_content_type_for_extension(extension: str) -> str:
    if extension == ".png":
        return "image/png"
    if extension == ".gif":
        return "image/gif"
    if extension in {".jpg", ".jpeg", ".jpe", ".jfif", ".jprg"}:
        return "image/jpeg"
    if extension == ".webp":
        return "image/webp"
    if extension == ".bmp":
        return "image/bmp"
    return ""


def _detect_image_content_type(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    return ""
