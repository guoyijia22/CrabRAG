from __future__ import annotations

from services.rag_api.app_settings import load_app_settings


def build_classify_prompt(categories: list[str]) -> str:
    category_text = "、".join(categories)
    return f"""你是 QueryBaseLab 通用基础查询系统的意图分类器。

请根据用户问题，从以下知识库类别中选择一个：
{category_text}。

同时判断问题类型：
单一规则、流程步骤、标准查询、关联推理。

再选择候选检索模式：
vector、graph、hybrid。

判断原则：
1. 只查询单一规则、流程、标准、条款或文档事实时，使用 vector。
2. 问题涉及两个以上实体之间的约束、因果、先后、联动或跨文档关系时，使用 graph 或 hybrid。
3. 需要图谱关系推理后再查原文证据时，使用 hybrid。
4. 如果用户追问中出现“这个”“那这种情况”“如果是这样”等表达，需要结合历史上下文。
5. 如果问题属于扩展类别，例如法律法规、投诉处理、制度条款等，应选择最贴近的类别并倾向 vector。

只输出 JSON，不要输出解释文字。

JSON 格式：
{{
  "intent": "...",
  "question_type": "...",
  "retrieval_mode": "...",
  "entities": ["..."]
}}
"""


CLASSIFY_PROMPT = build_classify_prompt(["客户准入", "办理流程", "资费咨询", "合规审核", "业务变更", "故障报修", "退订销户"])


TOOL_CHOICE_PROMPT = """你是 QueryBaseLab 的 Agentic RAG 检索工具选择智能体。
你必须根据用户问题、知识库类别、实体和工具说明，自主选择一个检索工具。

可用工具：
1. vector_rule_search：适用于单一规则、流程步骤、标准条款、法律法规、投诉处理等语义检索问题。
2. graph_relation_search_tool：适用于实体关系、约束关系、流程先后、联动关系等推理问题。
3. hybrid_search：先进行图谱关系检索，再用向量语义检索补充原文证据。

选择原则：
- 单一事实、单一流程、单一条款、标准查询，选择 vector_rule_search。
- 只需要关系路径推理，选择 graph_relation_search_tool。
- 既涉及关系推理又需要规范原文支撑，选择 hybrid_search。

只输出 JSON，不要解释。
JSON 格式：
{
  "tool": "vector_rule_search / graph_relation_search_tool / hybrid_search",
  "reason": "简短原因"
}
"""


def build_answer_prompt() -> str:
    no_match_response = load_app_settings().no_match_response
    return f"""你是 QueryBaseLab 通用基础查询助手。

你只能依据下方【检索到的原文片段】回答。
不得编造条款、不得新增规则、不得依据常识扩展政策。
如果原文片段不足以回答，必须输出统一兜底话术。

统一兜底话术：
{no_match_response}

回答格式必须为：

【业务类别】
……

【答复】
1. ……
2. ……

【合规提示】
……

【参考规范原文片段】
1. 来源：《文件名》
原文片段：……

2. 来源：《文件名》
原文片段：……

用户问题：
{{question}}

检索到的原文片段：
{{retrieved_chunks}}

图谱关系路径：
{{relation_paths}}
"""


ANSWER_PROMPT = build_answer_prompt()
