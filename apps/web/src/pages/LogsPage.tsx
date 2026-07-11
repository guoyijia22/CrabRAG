import { useEffect, useState } from "react";

import { getLogs } from "../api/client";
import type { LogItem, UiLanguage } from "../api/types";
import { p } from "../page-i18n";

export function LogsPage({ language }: { language: UiLanguage }) {
  const text = p(language);
  const [items, setItems] = useState<LogItem[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [category, setCategory] = useState("");
  const [selected, setSelected] = useState<LogItem>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function load(nextCategory = category) {
    setBusy(true); setError(""); setSelected(undefined);
    try {
      const payload = await getLogs(nextCategory);
      setItems(payload.items || []);
      setCategories((current) => [...new Set([...current, ...payload.items.map((item) => String(item.intent || item.category || "")).filter(Boolean)])]);
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(false); }
  }
  useEffect(() => { void load(""); }, []);

  async function chooseCategory(next: string) { setCategory(next); await load(next); }

  return <main className="page-card">
    <div className="page-heading"><div><span className="eyebrow">Audit</span><h1>{text.logsTitle}</h1><p>{text.logsIntro}</p></div><div className="page-actions"><button type="button" onClick={() => load()} disabled={busy}>{text.refresh}</button></div></div>
    {error && <div className="alert error" role="alert">{error}</div>}
    <section className="filter-row"><label><span>{text.logCategory}</span><select aria-label={text.logCategory} value={category} onChange={(event) => chooseCategory(event.target.value)}><option value="">{text.all}</option>{categories.map((item) => <option key={item} value={item}>{item}</option>)}</select></label></section>
    {!busy && !error && items.length === 0 ? <div className="empty-state large">{text.noLogs}</div> : <div className="master-detail">
      <section className="item-list" aria-label={text.logsTitle}>{items.map((item, index) => <button aria-label={item.question || "-"} key={item.id || item.session_id || `${item.time || item.created_at}-${index}`} type="button" className={selected === item ? "selected" : ""} onClick={() => setSelected(item)}><strong>{item.question || "-"}</strong><small>{item.time || item.created_at || ""} · {item.intent || item.category || "-"}</small></button>)}</section>
      <aside className="detail-panel">{selected ? <LogDetails item={selected} language={language} /> : <p className="empty-state">{language === "zh" ? "选择一条日志查看详情" : "Select a log to inspect details"}</p>}</aside>
    </div>}
  </main>;
}

function LogDetails({ item, language }: { item: LogItem; language: UiLanguage }) {
  const text = p(language);
  const sources = (item.references || []).length ? item.references || [] : (item.sources || []).map((source_file) => ({ source_file, source: "", content: "", text: "" }));
  return <><h2>{text.details}</h2><h3>{item.question}</h3><p>{item.answer || "-"}</p>{item.error && <div className="alert error">{item.error}</div>}<div className="tag-row"><span className="tag">{item.intent || item.category || "-"}</span><span className="tag">{item.retrieval_mode || "-"}</span>{(item.entities || []).map((entity) => <span className="tag" key={entity}>{entity}</span>)}</div>
    <h3>{text.references}</h3>{sources.length ? sources.map((reference, index) => <article className="evidence-item" key={index}><strong>{reference.source_file || reference.source || "-"}</strong><p>{reference.content || reference.text || ""}</p></article>) : <p className="empty-state">-</p>}
    <h3>{text.paths}</h3>{(item.relation_paths || []).length ? <ul>{(item.relation_paths || []).map((path, index) => <li key={index}>{formatUnknown(path.path ?? path.relation_path)}</li>)}</ul> : <p className="empty-state">-</p>}
    <h3>{text.trace}</h3>{(item.trace || []).length ? <ol>{(item.trace || []).map((trace, index) => <li key={index}><strong>{trace.node || "step"}</strong> · {trace.detail || trace.message || formatUnknown(trace)}</li>)}</ol> : <p className="empty-state">-</p>}
  </>;
}

function formatUnknown(value: unknown): string {
  if (Array.isArray(value)) return value.join(" → ");
  if (typeof value === "string") return value;
  return JSON.stringify(value ?? "");
}
