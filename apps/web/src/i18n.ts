import type { UiLanguage } from "./api/types";

export const translations = {
  zh: {
    chat: "问答",
    knowledge: "知识库",
    graph: "知识图谱",
    logs: "日志",
    settings: "设置",
    evaluation: "评测对比",
    governance: "索引治理",
    editName: "编辑系统名称",
    saveName: "保存系统名称",
    language: "语言",
    pagePending: "页面源码将在后续任务中恢复",
  },
  en: {
    chat: "Q&A",
    knowledge: "Knowledge Base",
    graph: "Knowledge Graph",
    logs: "Logs",
    settings: "Settings",
    evaluation: "Evaluation",
    governance: "Index Governance",
    editName: "Edit system name",
    saveName: "Save system name",
    language: "Language",
    pagePending: "The page source will be restored in the next task",
  },
} as const;

export type TranslationKey = keyof typeof translations.zh;

export function t(language: UiLanguage, key: TranslationKey): string {
  return translations[language][key];
}
