from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator

PROJECT_ROOT = Path(os.getenv("ELCQA_ROOT") or Path(__file__).resolve().parents[2]).resolve()
APP_SETTINGS_PATH = PROJECT_ROOT / "data" / "app_settings.json"
CONFIG_MD_PATH = PROJECT_ROOT / "Config.md"

UiTheme = Literal["red_white", "blue_white", "classic_green"]

DEFAULT_SYSTEM_NAME = "QueryBaseLab 通用基础查询"
DEFAULT_KNOWLEDGE_BASE_NAME = "通用基础查询知识库"
DEFAULT_UI_THEME: UiTheme = "red_white"
DEFAULT_NO_MATCH_RESPONSE = "暂无相关知识库依据，无法为您解答"
DEFAULT_OUT_OF_SCOPE_RESPONSE = "当前问题不属于本系统配置的查询范围，无法为您解答。"

DEFAULT_BUSINESS_SCOPE_DESCRIPTION = "面向本地知识库文档的通用基础查询、检索增强问答、知识图谱分析和可溯源回答。"

DEFAULT_IN_SCOPE_KEYWORDS = [
    "查询",
    "知识库",
    "文档",
    "规范",
    "流程",
    "规则",
    "条款",
    "制度",
    "政策",
    "材料",
    "审核",
    "办理",
    "资费",
    "故障",
    "投诉",
    "法律",
    "公司法",
    "通信行业",
]

DEFAULT_OUT_OF_SCOPE_KEYWORDS = [
    "天气",
    "股票",
    "彩票",
    "星座",
    "菜谱",
    "旅游攻略",
    "电影推荐",
    "游戏攻略",
    "医疗诊断",
    "个人理财",
]


class AppSettings(BaseModel):
    system_name: str = Field(default=DEFAULT_SYSTEM_NAME, min_length=4, max_length=40)
    knowledge_base_name: str = Field(default=DEFAULT_KNOWLEDGE_BASE_NAME, min_length=2, max_length=30)
    ui_theme: UiTheme = DEFAULT_UI_THEME
    common_questions: list[str] = Field(default_factory=list, max_length=10)
    business_scope_description: str = DEFAULT_BUSINESS_SCOPE_DESCRIPTION
    in_scope_keywords: list[str] = Field(default_factory=lambda: DEFAULT_IN_SCOPE_KEYWORDS.copy())
    out_of_scope_keywords: list[str] = Field(default_factory=lambda: DEFAULT_OUT_OF_SCOPE_KEYWORDS.copy())
    scope_min_score: float = Field(default=0.25, ge=0.0, le=1.0)
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

    @field_validator("business_scope_description", "out_of_scope_response", "no_match_response", mode="after")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized


def load_app_settings() -> AppSettings:
    if APP_SETTINGS_PATH.exists():
        try:
            return AppSettings.model_validate_json(APP_SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return AppSettings()
    migrated = _migrate_from_config_md()
    return save_app_settings(migrated)


def save_app_settings(settings: AppSettings) -> AppSettings:
    normalized = AppSettings.model_validate(settings.model_dump())
    APP_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    APP_SETTINGS_PATH.write_text(json.dumps(normalized.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def read_public_app_config() -> dict:
    settings = load_app_settings()
    return {
        "system_name": settings.system_name,
        "knowledge_base_name": settings.knowledge_base_name,
        "ui_theme": settings.ui_theme,
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


def _migrate_from_config_md() -> AppSettings:
    if not CONFIG_MD_PATH.exists():
        return AppSettings()
    try:
        text = CONFIG_MD_PATH.read_text(encoding="utf-8")
    except OSError:
        return AppSettings()
    return AppSettings(
        system_name=_parse_scalar(text, "system_name") or DEFAULT_SYSTEM_NAME,
        knowledge_base_name=_parse_scalar(text, "knowledge_base_name") or DEFAULT_KNOWLEDGE_BASE_NAME,
        common_questions=_parse_common_questions(text),
    )


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
