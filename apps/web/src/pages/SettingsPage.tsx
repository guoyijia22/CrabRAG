import { useEffect, useState, type ReactNode } from "react";

import { getAppSettings, getModelSettings, getRagSettings, putAppSettings, putModelSettings, putRagSettings } from "../api/client";
import type { AppSettings, ModelSettings, ModelSettingsUpdate, RagSettings, UiLanguage, UiTheme } from "../api/types";

interface SettingsPageProps {
  language: UiLanguage;
  onAppSettingsSaved: (settings: AppSettings) => void;
}

const copy = {
  zh: {
    title: "系统设置",
    subtitle: "集中配置系统名称、模型接口、页面色彩、业务边界、兜底话术和 RAG 优化开关。",
    loading: "正在读取设置…",
    loaded: "已加载当前设置",
    loadFailed: "设置读取失败",
    save: "保存设置",
    saving: "正在保存…",
    saved: "设置已保存",
    saveFailed: "设置保存失败",
    reset: "恢复 RAG 默认",
    resetReady: "RAG 默认值已载入，保存后生效",
  },
  en: {
    title: "System Settings",
    subtitle: "Configure system name, model APIs, page theme, business boundaries, fallback messages, and RAG optimization switches.",
    loading: "Loading settings…",
    loaded: "Settings loaded",
    loadFailed: "Failed to load settings",
    save: "Save settings",
    saving: "Saving…",
    saved: "Settings saved",
    saveFailed: "Failed to save settings",
    reset: "Restore RAG defaults",
    resetReady: "RAG defaults loaded; save to apply",
  },
} as const;

const defaultRagSettings: RagSettings = {
  multi_vector_enabled: false,
  hybrid_bm25_enabled: false,
  query_expansion_enabled: false,
  rerank_enabled: false,
  context_rewrite_enabled: false,
  rag_param_tuning_enabled: false,
  chunk_size: 600,
  chunk_overlap: 100,
  top_k: 2,
  min_score: 0.35,
  vector_candidate_k: 8,
  max_context_tokens: 6000,
  bm25_weight: 0.5,
  vector_weight: 0.5,
  rerank_provider: "api",
  rerank_model: "BAAI/bge-reranker-v2-m3",
};

type StatusKey = keyof typeof copy.zh;
interface SettingsStatus { key: StatusKey; detail?: string }

function lines(value: string): string[] {
  return value.split(/\r?\n/).map((item) => item.trim()).filter((item, index, values) => item && values.indexOf(item) === index);
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return <section className="settings-section"><h2>{title}</h2>{children}</section>;
}

function TextField({ label, value, onChange, type = "text", placeholder, disabled }: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  type?: string;
  placeholder?: string;
  disabled?: boolean;
}) {
  return <label className="field"><span>{label}</span><input type={type} value={value} placeholder={placeholder} disabled={disabled} onChange={(event) => onChange(event.target.value)} /></label>;
}

function NumberField({ label, value, onChange, min, max, step = 1 }: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return <label className="field"><span>{label}</span><input type="number" value={value} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} /></label>;
}

function Toggle({ label, checked, onChange, description }: { label: string; checked: boolean; onChange: (checked: boolean) => void; description?: string }) {
  return <label className="toggle-field"><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><span><strong>{label}</strong>{description && <small>{description}</small>}</span></label>;
}

function AppConfiguration({ value, language, onChange }: { value: AppSettings; language: UiLanguage; onChange: (value: AppSettings) => void }) {
  const themeLabels: Record<UiTheme, [string, string]> = {
    red_white: language === "zh" ? ["红色+白色", "正式汇报和通用查询"] : ["Red + white", "Formal reports and general queries"],
    blue_white: language === "zh" ? ["蓝色+白色", "科技和运营看板风格"] : ["Blue + white", "Technology and operations style"],
    classic_green: language === "zh" ? ["绿色+白色", "绿色灰白工作台风格"] : ["Green + white", "Green and neutral workspace"],
  };
  return (
    <>
      <Section title={language === "zh" ? "系统配置" : "System configuration"}>
        <div className="field-grid">
          <TextField label={language === "zh" ? "系统名称" : "System name"} value={value.system_name} onChange={(system_name) => onChange({ ...value, system_name })} />
        </div>
        <label className="field"><span>{language === "zh" ? "知识库读取目录（每行一个）" : "Knowledge base directories (one per line)"}</span><textarea rows={4} value={value.knowledge_base_dirs.join("\n")} onChange={(event) => onChange({ ...value, knowledge_base_dirs: lines(event.target.value) })} /></label>
        <p className="field-help">{language === "zh" ? "支持多个目录并递归读取；保存后需重建知识库生效。支持 txt、docx、pdf、xlsx、xlsm、csv、pptx。" : "Multiple directories are scanned recursively. Rebuild after saving. Supports txt, docx, pdf, xlsx, xlsm, csv, and pptx."}</p>
        <fieldset className="theme-picker">
          <legend>{language === "zh" ? "页面色彩" : "Page theme"}</legend>
          {Object.entries(themeLabels).map(([theme, [name, description]]) => (
            <label className={value.ui_theme === theme ? "theme-card selected" : "theme-card"} key={theme}>
              <input type="radio" name="ui-theme" checked={value.ui_theme === theme} onChange={() => onChange({ ...value, ui_theme: theme as UiTheme })} />
              <span className={`theme-dots ${theme}`} aria-hidden="true"><i /><i /><i /></span>
              <strong>{name}</strong><small>{description}</small>
            </label>
          ))}
        </fieldset>
        <label className="field"><span>{language === "zh" ? "常用问题（每行一条，最多 10 条）" : "Common questions (one per line, max 10)"}</span><textarea rows={4} value={value.common_questions.join("\n")} onChange={(event) => onChange({ ...value, common_questions: lines(event.target.value).slice(0, 10) })} /></label>
      </Section>
      <BusinessConfiguration value={value} language={language} onChange={onChange} />
    </>
  );
}

function BusinessConfiguration({ value, language, onChange }: { value: AppSettings; language: UiLanguage; onChange: (value: AppSettings) => void }) {
  return (
    <Section title={language === "zh" ? "业务边界配置" : "Business boundary configuration"}>
      <label className="field"><span>{language === "zh" ? "业务范围描述" : "Business scope description"}</span><textarea rows={3} value={value.business_scope_description} onChange={(event) => onChange({ ...value, business_scope_description: event.target.value })} /></label>
      <div className="field-grid three">
        <NumberField label={language === "zh" ? "最小业务匹配分" : "Minimum business match score"} value={value.scope_min_score} min={0} max={1} step={0.05} onChange={(scope_min_score) => onChange({ ...value, scope_min_score })} />
        <label className="field"><span>{language === "zh" ? "业务内关键词（每行一条）" : "In-scope keywords (one per line)"}</span><textarea rows={3} value={value.in_scope_keywords.join("\n")} onChange={(event) => onChange({ ...value, in_scope_keywords: lines(event.target.value) })} /></label>
        <label className="field"><span>{language === "zh" ? "业务外关键词（每行一条）" : "Out-of-scope keywords (one per line)"}</span><textarea rows={3} value={value.out_of_scope_keywords.join("\n")} onChange={(event) => onChange({ ...value, out_of_scope_keywords: lines(event.target.value) })} /></label>
      </div>
      <div className="field-grid">
        <label className="field"><span>{language === "zh" ? "业务外答复语" : "Out-of-scope response"}</span><textarea rows={2} value={value.out_of_scope_response} onChange={(event) => onChange({ ...value, out_of_scope_response: event.target.value })} /></label>
        <label className="field"><span>{language === "zh" ? "无依据答复语" : "No-evidence response"}</span><textarea rows={2} value={value.no_match_response} onChange={(event) => onChange({ ...value, no_match_response: event.target.value })} /></label>
      </div>
    </Section>
  );
}

function ModelConfiguration({ value, language, onChange, secrets, onSecretChange }: {
  value: ModelSettings;
  language: UiLanguage;
  onChange: (value: ModelSettings) => void;
  secrets: { llm: string; embedding: string; rerank: string };
  onSecretChange: (name: "llm" | "embedding" | "rerank", value: string) => void;
}) {
  const unavailable = value.local_model_status.missing_count > 0;
  const configured = (set: boolean, source: string, hint: string) => set ? `${language === "zh" ? "已配置" : "Configured"} · ${hint || source}` : language === "zh" ? "未配置" : "Not configured";
  return (
    <Section title={language === "zh" ? "模型接口配置" : "Model API configuration"}>
      <Toggle label={language === "zh" ? "使用本地模型" : "Use local models"} checked={value.use_local_models} onChange={(use_local_models) => onChange({ ...value, use_local_models })} />
      {unavailable && <div className="alert warning">{language === "zh" ? "本地模型能力不可用：请按下方路径补齐模型文件；远程模型模式仍可使用。" : "Local model capability is unavailable. Add the files listed below; remote model mode remains available."}</div>}
      <h3>LLM</h3>
      <div className="field-grid">
        <TextField label={language === "zh" ? "大语言模型 MODEL_NAME" : "LLM MODEL_NAME"} value={value.chat_model} onChange={(chat_model) => onChange({ ...value, chat_model })} />
        <TextField label={language === "zh" ? "大语言模型 Base URL" : "LLM Base URL"} value={value.base_url} onChange={(base_url) => onChange({ ...value, base_url })} />
        <TextField type="password" label={language === "zh" ? "大语言模型 API Key" : "LLM API Key"} value={secrets.llm} placeholder={configured(value.api_key_set, value.api_key_source, value.api_key_hint)} onChange={(secret) => onSecretChange("llm", secret)} />
      </div>
      <h3>Embedding</h3>
      <div className="field-grid">
        <TextField label={language === "zh" ? "Embedding 模型" : "Embedding model"} value={value.embedding_model} onChange={(embedding_model) => onChange({ ...value, embedding_model })} />
        <TextField label={language === "zh" ? "检索 Base URL（Embedding）" : "Retrieval Base URL (Embedding)"} value={value.embedding_base_url} onChange={(embedding_base_url) => onChange({ ...value, embedding_base_url })} />
        <TextField type="password" label={language === "zh" ? "检索 API Key（本地可留空）" : "Retrieval API Key (optional for local)"} value={secrets.embedding} placeholder={configured(value.embedding_api_key_set, value.embedding_api_key_source, value.embedding_api_key_hint)} onChange={(secret) => onSecretChange("embedding", secret)} />
      </div>
      <h3>Rerank</h3>
      <div className="field-grid">
        <TextField label={language === "zh" ? "排序 Base URL（Rerank）" : "Rerank Base URL"} value={value.rerank_base_url} onChange={(rerank_base_url) => onChange({ ...value, rerank_base_url })} />
        <TextField type="password" label={language === "zh" ? "排序 API Key（本地可留空）" : "Rerank API Key (optional for local)"} value={secrets.rerank} placeholder={configured(value.rerank_api_key_set, value.rerank_api_key_source, value.rerank_api_key_hint)} onChange={(secret) => onSecretChange("rerank", secret)} />
      </div>
      {value.local_model_status.models.length > 0 && <div className="model-status-list">{value.local_model_status.models.map((model) => <details key={model.key}><summary>{model.name} · {model.present ? language === "zh" ? "本地模型文件已检测到" : "Local model files detected" : language === "zh" ? "缺失本地模型文件" : "Missing local model files"}</summary><div className="model-download-detail"><span>{language === "zh" ? "存放目录：" : "Save to:"}</span><code>{model.expected_dir}</code><span>{language === "zh" ? "缺失文件：" : "Missing files:"}</span><div>{model.missing_files.join(", ") || "-"}</div><a href={language === "zh" ? model.download_urls.zh : model.download_urls.en} target="_blank" rel="noreferrer">{language === "zh" ? "下载地址（ModelScope）" : "Download (Hugging Face)"}</a></div></details>)}</div>}
    </Section>
  );
}

function RagConfiguration({ value, language, onChange }: { value: RagSettings; language: UiLanguage; onChange: (value: RagSettings) => void }) {
  const toggle = (key: keyof RagSettings, checked: boolean) => onChange({ ...value, [key]: checked });
  const number = (key: keyof RagSettings, next: number) => onChange({ ...value, [key]: next });
  return (
    <Section title={language === "zh" ? "RAG 检索优化与参数" : "RAG retrieval optimization and parameters"}>
      <div className="toggle-grid">
        <Toggle label={language === "zh" ? "多粒度文本索引" : "Multi-granularity text index"} checked={value.multi_vector_enabled} onChange={(checked) => toggle("multi_vector_enabled", checked)} />
        <Toggle label="BM25 + Vector" checked={value.hybrid_bm25_enabled} onChange={(checked) => toggle("hybrid_bm25_enabled", checked)} />
        <Toggle label={language === "zh" ? "查询扩展" : "Query expansion"} checked={value.query_expansion_enabled} onChange={(checked) => toggle("query_expansion_enabled", checked)} />
        <Toggle label={language === "zh" ? "统一重排" : "Unified rerank"} checked={value.rerank_enabled} onChange={(checked) => toggle("rerank_enabled", checked)} />
        <Toggle label={language === "zh" ? "多轮追问重写" : "Multi-turn follow-up rewrite"} checked={value.context_rewrite_enabled} onChange={(checked) => toggle("context_rewrite_enabled", checked)} />
        <Toggle label={language === "zh" ? "参数调优" : "Parameter tuning"} checked={value.rag_param_tuning_enabled} onChange={(checked) => toggle("rag_param_tuning_enabled", checked)} />
      </div>
      <div className="field-grid three">
        <NumberField label="chunk_size" value={value.chunk_size} min={200} max={1200} onChange={(next) => number("chunk_size", next)} />
        <NumberField label="chunk_overlap" value={value.chunk_overlap} min={0} max={300} onChange={(next) => number("chunk_overlap", next)} />
        <NumberField label={language === "zh" ? "最终引用数量 top_k" : "Final citation count top_k"} value={value.top_k} min={1} max={10} onChange={(next) => number("top_k", next)} />
        <NumberField label="min_score" value={value.min_score} min={0} max={1} step={0.05} onChange={(next) => number("min_score", next)} />
        <NumberField label={language === "zh" ? "候选数量" : "Candidate count"} value={value.vector_candidate_k} min={2} max={50} onChange={(next) => number("vector_candidate_k", next)} />
        <NumberField label={language === "zh" ? "上下文 Token 预算" : "Context token budget"} value={value.max_context_tokens} min={100} max={50000} onChange={(next) => number("max_context_tokens", next)} />
        <NumberField label="BM25 weight" value={value.bm25_weight} min={0} max={1} step={0.05} onChange={(next) => number("bm25_weight", next)} />
        <NumberField label="Vector weight" value={value.vector_weight} min={0} max={1} step={0.05} onChange={(next) => number("vector_weight", next)} />
        <TextField label="Rerank model" value={value.rerank_model} onChange={(rerank_model) => onChange({ ...value, rerank_model })} />
      </div>
      <label className="field"><span>Rerank provider</span><select value={value.rerank_provider} onChange={(event) => onChange({ ...value, rerank_provider: event.target.value as RagSettings["rerank_provider"] })}><option value="api">API</option><option value="local_onnx">Local ONNX</option></select></label>
    </Section>
  );
}

export function SettingsPage({ language, onAppSettingsSaved }: SettingsPageProps) {
  const text = copy[language];
  const [app, setApp] = useState<AppSettings | null>(null);
  const [model, setModel] = useState<ModelSettings | null>(null);
  const [rag, setRag] = useState<RagSettings | null>(null);
  const [secrets, setSecrets] = useState({ llm: "", embedding: "", rerank: "" });
  const [status, setStatus] = useState<SettingsStatus>({ key: "loading" });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let active = true;
    setStatus({ key: "loading" });
    Promise.all([getAppSettings(), getModelSettings(), getRagSettings()])
      .then(([appPayload, modelPayload, ragPayload]) => {
        if (!active) return;
        setApp(appPayload); setModel(modelPayload); setRag(ragPayload); setStatus({ key: "loaded" });
      })
      .catch((reason: unknown) => active && setStatus({ key: "loadFailed", detail: reason instanceof Error ? reason.message : String(reason) }));
    return () => { active = false; };
  }, []);

  async function save() {
    if (!app || !model || !rag) return;
    setSaving(true);
    setStatus({ key: "saving" });
    const modelUpdate: ModelSettingsUpdate = {
      use_local_models: model.use_local_models,
      api_key: secrets.llm || null,
      base_url: model.base_url,
      openai_compatible: true,
      chat_model: model.chat_model,
      embedding_provider: model.use_local_models ? "local_onnx" : "api",
      embedding_api_key: secrets.embedding || null,
      embedding_base_url: model.embedding_base_url,
      embedding_openai_compatible: true,
      embedding_model: model.embedding_model,
      embedding_onnx_model_file: model.embedding_onnx_model_file,
      rerank_api_key: secrets.rerank || null,
      rerank_base_url: model.rerank_base_url,
      rerank_onnx_model_file: model.rerank_onnx_model_file,
    };
    try {
      await Promise.all([putAppSettings(app), putModelSettings(modelUpdate), putRagSettings(rag)]);
      const [freshApp, freshModel, freshRag] = await Promise.all([getAppSettings(), getModelSettings(), getRagSettings()]);
      setApp(freshApp); setModel(freshModel); setRag(freshRag); setSecrets({ llm: "", embedding: "", rerank: "" });
      onAppSettingsSaved(freshApp);
      setStatus({ key: "saved" });
    } catch (reason) {
      setStatus({ key: "saveFailed", detail: reason instanceof Error ? reason.message : String(reason) });
    } finally {
      setSaving(false);
    }
  }

  return (
    <main className="settings-page page-card">
      <div className="page-heading">
        <div><span className="eyebrow">CrabRAG · v1.1.0</span><h1>{text.title}</h1><p>{text.subtitle}</p></div>
        <div className="page-actions"><button type="button" onClick={() => { setRag({ ...defaultRagSettings }); setStatus({ key: "resetReady" }); }} disabled={!rag || saving}>{text.reset}</button><button className="primary-button" type="button" onClick={save} disabled={!app || !model || !rag || saving}>{saving ? text.saving : text.save}</button></div>
      </div>
      <div className="alert" role="status">{text[status.key]}{status.detail ? `: ${status.detail}` : ""}</div>
      {app && <AppConfiguration value={app} language={language} onChange={setApp} />}
      {model && <ModelConfiguration value={model} language={language} onChange={setModel} secrets={secrets} onSecretChange={(name, value) => setSecrets((current) => ({ ...current, [name]: value }))} />}
      {rag && <RagConfiguration value={rag} language={language} onChange={setRag} />}
    </main>
  );
}
