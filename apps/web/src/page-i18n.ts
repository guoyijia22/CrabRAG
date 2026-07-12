import type { UiLanguage } from "./api/types";

export const pageText = {
  zh: {
    refresh: "刷新", loading: "加载中…", permission: "需要索引管理权限", details: "详情",
    knowledgeTitle: "知识库", knowledgeIntro: "查看知识库目录、重建进度和图谱结构建议。",
    governanceTitle: "索引治理", governanceIntro: "查看当前/上一代索引、复用率、调度、缓存、告警和清理状态。",
    incremental: "增量更新", full: "全量重建", confirmFull: "全量重建会重新计算所有索引，确定继续吗？",
    directory: "读取目录", noDirectory: "请先在设置页面的“知识库读取目录（每行一个）”配置文件目录", noFiles: "无文件", chroma: "Chroma 状态",
    progress: "重建进度", result: "重建结果", schema: "图谱结构建议", confirmSchema: "确认图谱结构",
    activeGeneration: "当前索引代", previousGeneration: "上一索引代", documents: "文档", chunks: "片段", reused: "复用向量", recomputed: "重算向量", dimension: "向量维度",
    scheduler: "调度器", cache: "检索缓存", warnings: "治理告警", governanceWarningCount: "治理告警数", viewGovernance: "查看索引治理", cleanup: "清理状态", rollback: "回滚到上一代", noWarnings: "无治理告警",
    graphTitle: "知识图谱", graphIntro: "展示知识库 API 返回的动态图谱和可溯源证据。", refreshGraph: "刷新图谱", searchNode: "搜索节点", noGraph: "暂无知识图谱", nodes: "个节点", edges: "条边", source: "来源", evidence: "证据", properties: "属性",
    logsTitle: "问答日志", logsIntro: "按分类检查问题、答复、检索证据和执行 Trace。", logCategory: "日志分类", all: "全部", noLogs: "暂无日志", references: "引用", paths: "图谱路径", trace: "Trace",
    evaluationTitle: "RAG 评测对比", evaluationIntro: "基于当前知识库运行评测，比较各 profile 指标并检查逐题证据。", runEvaluation: "运行评测", noEvaluations: "暂无评测记录", history: "评测历史", profileMetrics: "Profile 指标", cases: "评测用例", successRate: "成功率", sourceHitRate: "来源命中率", graphCoverage: "图谱路径覆盖率", latency: "平均耗时", qualityScore: "质量分", recommendation: "建议", error: "错误", answer: "答复", metrics: "指标", questionSet: "题集摘要", fixedDataset: "固定评测集", dynamicDataset: "动态题集（不可用于门禁）", recallAt5: "Recall@5", mrrAt10: "MRR@10", citationPrecision: "引用精确率", citationCoverage: "引用覆盖率", noEvidenceRate: "无证据回答率", aclLeakage: "ACL 泄漏率", invalidLeakage: "失效内容泄漏率", p95Latency: "P95 延迟", modelCalls: "模型调用量", gatePassed: "质量门禁通过", gateFailed: "质量门禁未通过", gateBaseline: "基线对照", gateIneligible: "需使用固定评测集",
  },
  en: {
    refresh: "Refresh", loading: "Loading…", permission: "Index management permission required", details: "Details",
    knowledgeTitle: "Knowledge Base", knowledgeIntro: "Inspect source directories, rebuild progress, and graph schema suggestions.",
    governanceTitle: "Index Governance", governanceIntro: "Inspect active/previous generations, reuse, scheduler, cache, warnings, and cleanup.",
    incremental: "Incremental update", full: "Full rebuild", confirmFull: "A full rebuild recomputes every index. Continue?",
    directory: "Source directories", noDirectory: "Configure file directories in Settings > Knowledge base directories (one per line)", noFiles: "No files", chroma: "Chroma status",
    progress: "Rebuild progress", result: "Rebuild result", schema: "Graph schema suggestion", confirmSchema: "Confirm graph schema",
    activeGeneration: "Active generation", previousGeneration: "Previous generation", documents: "Documents", chunks: "Chunks", reused: "Reused embeddings", recomputed: "Recomputed embeddings", dimension: "Embedding dimension",
    scheduler: "Scheduler", cache: "Retrieval cache", warnings: "Governance warnings", governanceWarningCount: "Governance warning count", viewGovernance: "View index governance", cleanup: "Cleanup", rollback: "Roll back to previous generation", noWarnings: "No governance warnings",
    graphTitle: "Knowledge Graph", graphIntro: "Explore the traceable dynamic graph returned by the knowledge-base API.", refreshGraph: "Refresh graph", searchNode: "Search nodes", noGraph: "No knowledge graph", nodes: " nodes", edges: " edges", source: "Source", evidence: "Evidence", properties: "Properties",
    logsTitle: "Q&A Logs", logsIntro: "Filter questions, answers, retrieval evidence, and execution traces.", logCategory: "Log category", all: "All", noLogs: "No logs yet", references: "References", paths: "Graph paths", trace: "Trace",
    evaluationTitle: "RAG Evaluation", evaluationIntro: "Run evaluations against the active knowledge base and inspect profile metrics and case evidence.", runEvaluation: "Run evaluation", noEvaluations: "No evaluation runs yet", history: "Evaluation history", profileMetrics: "Profile metrics", cases: "Cases", successRate: "Success rate", sourceHitRate: "Source hit rate", graphCoverage: "Graph path coverage", latency: "Average latency", qualityScore: "Quality score", recommendation: "Recommendation", error: "Error", answer: "Answer", metrics: "Metrics", questionSet: "Question-set summary", fixedDataset: "Fixed dataset", dynamicDataset: "Dynamic set (not gate eligible)", recallAt5: "Recall@5", mrrAt10: "MRR@10", citationPrecision: "Citation precision", citationCoverage: "Citation coverage", noEvidenceRate: "No-evidence answer rate", aclLeakage: "ACL leakage", invalidLeakage: "Inactive-content leakage", p95Latency: "P95 latency", modelCalls: "Model calls", gatePassed: "Quality gate passed", gateFailed: "Quality gate failed", gateBaseline: "Baseline reference", gateIneligible: "Fixed dataset required",
  },
} as const;

export function p(language: UiLanguage) {
  return pageText[language];
}

const runtimeEnglish: Record<string, string> = {
  "等待执行": "Waiting",
  "开始重建": "Starting rebuild",
  "完成": "Completed",
  "失败": "Failed",
  "知识库重建完成": "Knowledge-base rebuild completed",
  "知识库重建失败": "Knowledge-base rebuild failed",
  "评测完成": "Evaluation completed",
  "正在准备评测配置": "Preparing evaluation profiles",
  "标准新链路基线": "Standard retrieval baseline",
  "查询扩展": "Query expansion",
  "统一重排": "Unified rerank",
  "多轮追问重写": "Multi-turn follow-up rewrite",
  "多粒度文本索引": "Multi-granularity text index",
  "查询扩展 + 统一重排": "Query expansion + unified rerank",
  "多轮追问重写 + 统一重排": "Multi-turn follow-up rewrite + unified rerank",
  "多粒度文本索引 + 统一重排": "Multi-granularity text index + unified rerank",
  "全增强配置": "All enhancements",
  "效果最好": "Best result",
  "建议适用": "Recommended",
  "不建议适用/有风险": "Not recommended / risky",
  "效果接近": "Close result",
  "来源命中更高": "Higher source hit",
};

export function localizeRuntime(language: UiLanguage, value: unknown): string {
  const raw = String(value ?? "");
  if (language === "zh" || !raw) return raw;
  if (runtimeEnglish[raw]) return runtimeEnglish[raw];
  return raw
    .replaceAll("质量优先综合分", "Quality-first score")
    .replaceAll("成功率", "success rate")
    .replaceAll("来源命中率", "source hit rate")
    .replaceAll("图谱路径覆盖率", "graph path coverage")
    .replaceAll("，", ", ")
    .replaceAll("。", ".");
}
