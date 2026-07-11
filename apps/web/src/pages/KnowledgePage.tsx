import { useEffect, useState } from "react";

import {
  getActiveIngest, getGraphSchemaSuggestion, getHealth, getIndexStatus, getIngestProgress,
  getIngestResult, postIndexRollback, postIngest, putGraphSchema,
} from "../api/client";
import type { GraphSchema, HealthResponse, IndexStatus, IngestProgress, IngestResult, UiLanguage } from "../api/types";
import { useTaskPoller } from "../hooks/useTaskPoller";
import { localizeRuntime, p } from "../page-i18n";

interface KnowledgePageProps { language: UiLanguage; governanceOnly?: boolean }

const errorText = (reason: unknown) => reason instanceof Error ? reason.message : String(reason);

function value(object: Record<string, unknown> | undefined, key: string, fallback: unknown = "-") {
  return object?.[key] ?? fallback;
}

export function KnowledgePage({ language, governanceOnly = false }: KnowledgePageProps) {
  const text = p(language);
  const [health, setHealth] = useState<HealthResponse>();
  const [index, setIndex] = useState<IndexStatus>();
  const [schema, setSchema] = useState<GraphSchema>();
  const [progress, setProgress] = useState<IngestProgress>();
  const [result, setResult] = useState<IngestResult>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const { watch: watchIngest } = useTaskPoller<IngestProgress, IngestResult>({
    loadProgress: getIngestProgress,
    loadCompleted: getIngestResult,
    onProgress: setProgress,
    onCompleted: (completed) => {
      setResult(completed);
      window.dispatchEvent(new Event("crabrag:knowledge-base-rebuilt"));
    },
    onFailed: (failed) => { if (failed.error) setError(failed.error); },
    onError: (reason) => setError(errorText(reason)),
  });

  async function load() {
    setError("");
    const [healthResult, activeResult, schemaResult, indexResult] = await Promise.allSettled([
      getHealth(), getActiveIngest(), getGraphSchemaSuggestion(), getIndexStatus(),
    ]);
    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    if (schemaResult.status === "fulfilled") setSchema(schemaResult.value);
    if (indexResult.status === "fulfilled") setIndex(indexResult.value);
    const failure = [activeResult, schemaResult, indexResult].find((item) => item.status === "rejected");
    if (failure?.status === "rejected") setError(errorText(failure.reason));
    if (activeResult.status === "fulfilled") {
      const current = activeResult.value.active || activeResult.value.last_success || undefined;
      setProgress(current);
      if (activeResult.value.active?.run_id) void watchIngest(activeResult.value.active.run_id);
    }
  }

  useEffect(() => { void load(); }, []);

  async function rebuild(full: boolean) {
    if (full && !window.confirm(text.confirmFull)) return;
    setBusy(true); setError(""); setResult(undefined);
    try {
      const started = await postIngest(full);
      setProgress(started);
      await watchIngest(started.run_id);
    } catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  }

  async function confirmSchema() {
    if (!schema) return;
    setBusy(true); setError("");
    try { setSchema(await putGraphSchema(schema)); }
    catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  }

  async function rollback() {
    setBusy(true); setError("");
    try { setIndex({ ...index, ...(await postIndexRollback()) }); }
    catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  }

  const stats = index?.active?.stats;
  const docsDirs = Array.isArray(health?.docs_dirs) ? health.docs_dirs.map(String) : [];
  const warnings = index?.active?.warnings || [];
  return (
    <main className="page-card">
      <div className="page-heading">
        <div><span className="eyebrow">{__CRABRAG_VERSION_LABEL__}</span><h1>{governanceOnly ? text.governanceTitle : text.knowledgeTitle}</h1><p>{governanceOnly ? text.governanceIntro : text.knowledgeIntro}</p></div>
        <div className="page-actions">
          {!governanceOnly && <><button type="button" onClick={() => rebuild(false)} disabled={busy}>{text.incremental}</button><button type="button" onClick={() => rebuild(true)} disabled={busy}>{text.full}</button></>}
          <button type="button" onClick={load} disabled={busy}>{text.refresh}</button>
        </div>
      </div>
      {error && <div className="alert error" role="alert">{error}</div>}
      {!governanceOnly && <>
        <section className="dashboard-grid two">
          <article className="dashboard-panel"><h2>{text.directory}</h2>{docsDirs.length ? <ul>{docsDirs.map((dir) => <li key={dir}><code>{dir}</code></li>)}</ul> : <p className="empty-state">{text.noDirectory}</p>}</article>
          <article className="dashboard-panel"><h2>{text.chroma}</h2><strong className="large-value">{String(health?.chroma ?? "-")}</strong><p>{health?.docs_dir_has_files === false ? text.noFiles : ""}</p></article>
        </section>
        {(progress || result) && <section className="dashboard-panel"><h2>{result ? text.result : text.progress}</h2>{progress && <><progress max="100" value={progress.percent || 0} /><p>{localizeRuntime(language, progress.current_step || progress.message)} · {progress.percent || 0}%</p>{progress.error && <div className="alert error">{localizeRuntime(language, progress.error)}</div>}</>}{result && <MetricGrid values={[[text.activeGeneration, result.generation_id], [text.documents, result.document_count], [text.chunks, result.chunk_count], [text.reused, result.reused_embedding_count], [text.recomputed, result.embedded_chunk_count], [text.dimension, result.embedding_dimension], [`${text.graphTitle} · ${language === "zh" ? "节点" : "nodes"}`, result.graph_node_count], [`${text.graphTitle} · ${language === "zh" ? "边" : "edges"}`, result.graph_edge_count]]} />}</section>}
        <section className="dashboard-panel"><div className="section-heading"><h2>{text.schema}</h2>{schema && <button type="button" onClick={confirmSchema} disabled={busy}>{text.confirmSchema}</button>}</div>{schema ? <><div className="tag-row">{(schema.entity_types || []).map((item) => <span className="tag" key={item}>{item}</span>)}</div><dl className="property-list">{[...(schema.node_fields || []), ...(schema.edge_fields || [])].map((field) => <div key={`${field.key}-${field.label}`}><dt>{field.label || field.key}</dt><dd><code>{field.key}</code>{field.type ? ` · ${field.type}` : ""}</dd></div>)}</dl></> : <p className="empty-state">-</p>}</section>
      </>}
      <section className="dashboard-panel governance-panel">
        <div className="section-heading"><h2>{text.governanceTitle}</h2>{index?.can_rollback && <button className="danger-button" type="button" onClick={rollback} disabled={busy}>{text.rollback}</button>}</div>
        <MetricGrid values={[[text.activeGeneration, index?.active_generation], [text.previousGeneration, index?.previous_generation], [text.documents, value(stats, "document_count")], [text.chunks, value(stats, "chunk_count")], [text.reused, value(stats, "reused_embedding_count")], [text.recomputed, value(stats, "embedded_chunk_count")], [text.dimension, value(stats, "embedding_dimension")]]} />
        <div className="dashboard-grid three"><JsonPanel title={text.scheduler} value={index?.scheduler} /><JsonPanel title={text.cache} value={index?.cache} /><JsonPanel title={text.cleanup} value={(index?.scheduler?.last_cleanup as Record<string, unknown> | undefined) || {}} /></div>
        <h3>{text.warnings}</h3>{warnings.length ? <ul className="warning-list">{warnings.map((warning, i) => <li key={i}>{String(warning.code || "warning")}{warning.path ? ` · ${String(warning.path)}` : ""}</li>)}</ul> : <p className="empty-state">{text.noWarnings}</p>}
      </section>
    </main>
  );
}

function MetricGrid({ values }: { values: Array<[string, unknown]> }) {
  return <div className="metric-grid">{values.map(([label, metric]) => <div key={label}><span>{label}</span><strong>{String(metric ?? "-")}</strong></div>)}</div>;
}

function JsonPanel({ title, value }: { title: string; value: unknown }) {
  return <article className="mini-panel"><h3>{title}</h3><pre>{JSON.stringify(value || {}, null, 2)}</pre></article>;
}
