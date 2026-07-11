import { join } from "node:path";

export const DEFAULT_SYSTEM_NAME = "CrabRAG";
export const DEFAULT_KNOWLEDGE_BASE_NAME = "通用基础查询知识库";
export const DEFAULT_UI_THEME = "red_white";
export const DEFAULT_UI_LANGUAGE = "en";

const LEGACY_DEFAULT_SYSTEM_NAMES = new Set([
  "QueryBaseLab 通用基础查询",
  "QueryBasePortableLab 通用基础查询",
  "CrabRAG 通用基础查询",
]);

export interface PublicAppConfig {
  system_name: string;
  knowledge_base_name: string;
  ui_theme: "red_white" | "blue_white" | "classic_green";
  ui_language: "en" | "zh";
  common_questions: string[];
}

export async function readAppConfig(projectRoot: string): Promise<PublicAppConfig> {
  try {
    const payload = JSON.parse(await Bun.file(join(projectRoot, "data", "app_settings.json")).text());
    return {
      system_name: normalizeSystemName(payload.system_name || DEFAULT_SYSTEM_NAME),
      knowledge_base_name: payload.knowledge_base_name || DEFAULT_KNOWLEDGE_BASE_NAME,
      ui_theme: normalizeTheme(payload.ui_theme),
      ui_language: normalizeLanguage(payload.ui_language),
      common_questions: Array.isArray(payload.common_questions) ? payload.common_questions.slice(0, 10) : [],
    };
  } catch {
    return readLegacyConfig(projectRoot);
  }
}

async function readLegacyConfig(projectRoot: string): Promise<PublicAppConfig> {
  try {
    const text = await Bun.file(join(projectRoot, "Config.md")).text();
    return {
      system_name: normalizeSystemName(parseScalar(text, "system_name") || DEFAULT_SYSTEM_NAME),
      knowledge_base_name: parseScalar(text, "knowledge_base_name") || DEFAULT_KNOWLEDGE_BASE_NAME,
      ui_theme: DEFAULT_UI_THEME,
      ui_language: DEFAULT_UI_LANGUAGE,
      common_questions: parseCommonQuestions(text),
    };
  } catch {
    return {
      system_name: DEFAULT_SYSTEM_NAME,
      knowledge_base_name: DEFAULT_KNOWLEDGE_BASE_NAME,
      ui_theme: DEFAULT_UI_THEME,
      ui_language: DEFAULT_UI_LANGUAGE,
      common_questions: [],
    };
  }
}

function normalizeSystemName(value: string): string {
  return LEGACY_DEFAULT_SYSTEM_NAMES.has(value) ? DEFAULT_SYSTEM_NAME : value;
}

function normalizeTheme(value: unknown): PublicAppConfig["ui_theme"] {
  return value === "blue_white" || value === "classic_green" || value === "red_white"
    ? value
    : DEFAULT_UI_THEME;
}

function normalizeLanguage(value: unknown): PublicAppConfig["ui_language"] {
  return value === "zh" || value === "en" ? value : DEFAULT_UI_LANGUAGE;
}

function parseScalar(text: string, key: string): string {
  for (const line of text.split(/\r?\n/)) {
    const match = line.match(new RegExp(`^\\s*${key}\\s*:\\s*(.+?)\\s*$`));
    if (match?.[1]) return match[1].replace(/^["']|["']$/g, "").trim();
  }
  return "";
}

function parseCommonQuestions(text: string): string[] {
  const questions: string[] = [];
  let inBlock = false;
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!inBlock) {
      if (trimmed === "common_questions:") inBlock = true;
      continue;
    }
    if (!trimmed) continue;
    if (!trimmed.startsWith("- ")) break;
    const question = trimmed.slice(2).replace(/^["']|["']$/g, "").trim();
    if (question && !questions.includes(question)) questions.push(question);
  }
  return questions.slice(0, 10);
}
