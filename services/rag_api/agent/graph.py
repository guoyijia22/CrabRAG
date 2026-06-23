from __future__ import annotations

from langgraph.graph import END, StateGraph

from services.rag_api.agent.nodes import classify_intent_node, choose_retrieval_tool_node, generate_answer_node, retrieve_node
from services.rag_api.agent.state import QAState

workflow = StateGraph(QAState)
workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("agent_tool_choice", choose_retrieval_tool_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate_answer", generate_answer_node)
workflow.set_entry_point("classify_intent")
workflow.add_edge("classify_intent", "agent_tool_choice")
workflow.add_edge("agent_tool_choice", "retrieve")
workflow.add_edge("retrieve", "generate_answer")
workflow.add_edge("generate_answer", END)
qa_graph = workflow.compile()


def run_qa(state: QAState) -> QAState:
    return qa_graph.invoke(state)
