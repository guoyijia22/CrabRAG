from __future__ import annotations

from typing import Any

DEFAULT_COMMON_QUESTIONS = [
    "企业客户办理政企专线需要提供哪些材料？",
    "政企专线从申请到开通需要经过哪些步骤？",
    "100M 政企专线对应的资费套餐是什么？",
    "哪些情况属于业务开办一票否决？",
    "企业客户办理地址迁移时，是否需要重新进行合规审核？",
    "专线中断属于什么等级故障？报修流程是什么？",
    "客户存在欠费时能不能直接办理销户？",
]

CATEGORY_QUESTION_TEMPLATES = {
    "客户准入": "企业客户办理政企专线需要提供哪些材料？",
    "办理流程": "政企专线从申请到开通需要经过哪些步骤？",
    "资费咨询": "100M 政企专线对应的资费套餐是什么？",
    "合规审核": "哪些情况属于业务开办一票否决？",
    "业务变更": "企业客户办理地址迁移时，是否需要重新进行合规审核？",
    "故障报修": "专线中断属于什么等级故障？报修流程是什么？",
    "退订销户": "客户存在欠费时能不能直接办理销户？",
    "公司法律法规": "根据公司法，企业客户主体资格核验应关注哪些信息？",
    "投诉处理": "政企专线客户投诉后应按什么流程处理？",
}


def generate_common_questions(category_payload: dict[str, Any]) -> list[str]:
    names = [item.get("name", "") for item in category_payload.get("items", []) if item.get("name")]
    questions: list[str] = []
    for name in names:
        question = CATEGORY_QUESTION_TEMPLATES.get(name) or f"{name}相关规范中有哪些常见办理要求？"
        _append_unique(questions, question)

    for question in DEFAULT_COMMON_QUESTIONS:
        if len(questions) >= 5:
            break
        _append_unique(questions, question)

    return questions[:10]


def _append_unique(items: list[str], item: str) -> None:
    value = item.strip()
    if value and value not in items:
        items.append(value)
