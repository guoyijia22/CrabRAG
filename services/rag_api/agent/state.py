from typing import Any, Dict, List, Optional, TypedDict


class QAState(TypedDict, total=False):
    session_id: str
    question: str
    effective_question: str
    history: List[Dict[str, str]]
    intent: str
    question_type: str
    retrieval_mode: str
    selected_tool: str
    tool_choice_reason: str
    entities: List[str]
    business_scope: Dict[str, Any]
    retrieved_chunks: List[Dict[str, Any]]
    relation_paths: List[Dict[str, Any]]
    answer: str
    references: List[Dict[str, Any]]
    trace: List[Dict[str, Any]]
    error: Optional[str]
