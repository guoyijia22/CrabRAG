import { useEffect, useMemo, useState } from "react";

import { getCategories, getHealth, postChat, uploadSidebarImage } from "../api/client";
import type { AppSettings, CategoriesResponse, ChatResponse, HealthResponse, RelationPath, TraceItem, UiLanguage } from "../api/types";
import { MarkdownContent } from "../components/MarkdownContent";

interface ChatPageProps {
  settings: AppSettings;
  onSettingsChange: (settings: AppSettings) => void;
}

const labels = {
  zh: {
    businessCategory: "业务类别",
    status: "知识库状态",
    docs: "文档目录",
    categories: "分类",
    changeImage: "更换侧边栏图片",
    intro: "基于本地知识库、Chroma、GraphRAG 与 LangGraph 的可溯源问答",
    input: "输入问题",
    placeholder: "请输入需要查询的问题…",
    send: "发送",
    loading: "正在检索知识库…",
    answer: "回答",
    generation: "索引代",
    evidence: "引用依据",
    paths: "关系路径",
    trace: "执行轨迹",
    sourcePath: "来源路径",
    noQuestion: "请输入问题后再发送",
  },
  en: {
    businessCategory: "Business category",
    status: "Knowledge base status",
    docs: "Document directories",
    categories: "Categories",
    changeImage: "Change sidebar image",
    intro: "Traceable Q&A powered by the local knowledge base, Chroma, GraphRAG, and LangGraph",
    input: "Enter a question",
    placeholder: "Ask a question about the configured knowledge base…",
    send: "Send",
    loading: "Searching the knowledge base…",
    answer: "Answer",
    generation: "Index generation",
    evidence: "References",
    paths: "Relation paths",
    trace: "Execution trace",
    sourcePath: "Source path",
    noQuestion: "Enter a question before sending",
  },
} as const;

function imageDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("Unable to read image"));
    reader.readAsDataURL(file);
  });
}

function categoryNames(payload: CategoriesResponse | null): string[] {
  if (!payload) return [];
  if (payload.categories?.length) return payload.categories;
  return (payload.items || []).map((item) => item.name || "").filter(Boolean);
}

function pathText(path: RelationPath): string {
  const value = path.path ?? path.relation_path;
  if (Array.isArray(value)) return value.map(String).join(" → ");
  if (typeof value === "string") return value;
  return JSON.stringify(value ?? path);
}

function traceText(item: TraceItem): string {
  return item.detail || item.message || JSON.stringify(item);
}

export function ChatPage({ settings, onSettingsChange }: ChatPageProps) {
  const language: UiLanguage = settings.ui_language;
  const text = labels[language];
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [categories, setCategories] = useState<CategoriesResponse | null>(null);
  const [question, setQuestion] = useState("");
  const [sessionId, setSessionId] = useState<string>();
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);

  const sidebarImage = useMemo(
    () => settings.sidebar_image_url ? `${settings.sidebar_image_url}?v=${Date.now()}` : sidebarDefaultImage(),
    [settings.sidebar_image_url],
  );

  useEffect(() => {
    let active = true;
    const refreshCategories = () => {
      getCategories()
        .then((payload) => { if (active) setCategories(payload); })
        .catch((reason: unknown) => { if (active) setError(reason instanceof Error ? reason.message : String(reason)); });
    };
    Promise.all([getHealth(), getCategories()])
      .then(([healthPayload, categoryPayload]) => {
        if (!active) return;
        setHealth(healthPayload);
        setCategories(categoryPayload);
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : String(reason));
      });
    window.addEventListener("crabrag:knowledge-base-rebuilt", refreshCategories);
    return () => { active = false; window.removeEventListener("crabrag:knowledge-base-rebuilt", refreshCategories); };
  }, []);

  async function submit() {
    const normalized = question.trim();
    if (!normalized) {
      setError(text.noQuestion);
      return;
    }
    setLoading(true);
    setError("");
    try {
      const payload = await postChat({ question: normalized, ...(sessionId ? { session_id: sessionId } : {}) });
      setSessionId(payload.session_id);
      setResponse(payload);
      if (payload.error) setError(payload.error);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setLoading(false);
    }
  }

  async function changeImage(file: File | undefined) {
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      const data = await imageDataUrl(file);
      const saved = await uploadSidebarImage(file.name, file.type, data);
      onSettingsChange(saved);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setUploading(false);
    }
  }

  const names = categoryNames(categories);
  return (
    <main className="chat-layout">
      <aside className="sidebar-card">
        <h2>{text.businessCategory}</h2>
        <div className="status-stack">
          <strong>{text.status}</strong>
          <span>{text.docs}: {health?.docs_dir_exists === true ? "ok" : health?.docs_dir_exists === false ? "missing" : "unknown"}</span>
          <span>Chroma: {String(health?.chroma ?? "unknown")}</span>
          <span>LLM: {String(health?.llm_api ?? "unknown")}</span>
        </div>
        <div className="category-list" aria-label={text.categories}>
          {names.map((name) => <span className="tag" key={name}>{name}</span>)}
        </div>
        <img className="sidebar-image" src={sidebarImage} alt={language === "zh" ? "侧边栏展示图" : "Sidebar image"} />
        <label className="upload-button">
          {uploading ? "…" : text.changeImage}
          <input type="file" accept="image/png,image/jpeg,image/gif,image/webp,image/bmp" disabled={uploading} onChange={(event) => changeImage(event.target.files?.[0])} />
        </label>
      </aside>

      <section className="chat-panel">
        <div className="chat-hero">
          <span className="eyebrow">CrabRAG · {__CRABRAG_VERSION_LABEL__}</span>
          <h1>{settings.system_name}</h1>
          <p>{text.intro}</p>
          <div className="category-list">{names.map((name) => <span className="tag" key={name}>{name}</span>)}</div>
        </div>

        {settings.common_questions.length > 0 && (
          <div className="quick-questions">
            {settings.common_questions.map((item) => (
              <button type="button" key={item} onClick={() => setQuestion(item)}>{item}</button>
            ))}
          </div>
        )}

        <div className="composer">
          <label htmlFor="chat-question">{text.input}</label>
          <textarea id="chat-question" rows={4} value={question} placeholder={text.placeholder} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.altKey) { event.preventDefault(); void submit(); }
          }} />
          <button className="primary-button" type="button" disabled={loading} onClick={submit}>{loading ? text.loading : text.send}</button>
        </div>

        {error && <div className="alert error" role="alert">{error}</div>}
        {response && (
          <article className="answer-card">
            <div className="answer-heading">
              <h2>{text.answer}</h2>
              {response.index_generation && <span className="generation-chip"><small>{text.generation}</small>{response.index_generation}</span>}
            </div>
            <MarkdownContent value={response.answer} />
            <div className="answer-metadata">
              {response.intent && <span>{response.intent}</span>}
              {response.retrieval_mode && <span>{response.retrieval_mode}</span>}
              {response.question_type && <span>{response.question_type}</span>}
            </div>
            {!!response.references?.length && (
              <section className="evidence-section">
                <h3>{text.evidence}</h3>
                {response.references.map((reference, index) => (
                  <article className="evidence-item" key={`${reference.document_id || reference.source_file || "reference"}-${index}`}>
                    <p>{String(reference.content || reference.text || "")}</p>
                    {(reference.source_file || reference.source) && (
                      <details open>
                        <summary>{text.sourcePath}</summary>
                        <code>{String(reference.source_file || reference.source)}</code>
                      </details>
                    )}
                    {typeof reference.score === "number" && <small>score {reference.score.toFixed(3)}</small>}
                  </article>
                ))}
              </section>
            )}
            {!!response.relation_paths?.length && (
              <section className="evidence-section">
                <h3>{text.paths}</h3>
                {response.relation_paths.map((path, index) => <p className="path-line" key={index}>{pathText(path)}</p>)}
              </section>
            )}
            {!!response.trace?.length && (
              <details className="trace-section" open>
                <summary>{text.trace}</summary>
                <ol>{response.trace.map((item, index) => <li key={index}><strong>{item.node || `#${index + 1}`}</strong> {traceText(item)}</li>)}</ol>
              </details>
            )}
          </article>
        )}
      </section>
    </main>
  );
}

export function sidebarDefaultImage(){return`/picture/crab.png`}
