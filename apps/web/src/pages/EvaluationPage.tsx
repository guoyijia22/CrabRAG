import { useEffect, useState } from "react";

import { getActiveEvaluation, getEvaluation, getEvaluationProgress, getEvaluations, postEvaluationRun } from "../api/client";
import type { EvaluationProgress, EvaluationRun, EvaluationRunSummary, UiLanguage } from "../api/types";
import { useTaskPoller } from "../hooks/useTaskPoller";
import { localizeRuntime, p } from "../page-i18n";

export function EvaluationPage({ language }: { language: UiLanguage }) {
  const text = p(language);
  const [runs, setRuns] = useState<EvaluationRunSummary[]>([]);
  const [progress, setProgress] = useState<EvaluationProgress>();
  const [detail, setDetail] = useState<EvaluationRun>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const { watch } = useTaskPoller<EvaluationProgress, EvaluationRun>({
    loadProgress: getEvaluationProgress,
    loadCompleted: (runId) => getEvaluation(runId),
    onProgress: setProgress,
    onCompleted: setDetail,
    onFailed: (failed) => { if (failed.error) setError(failed.error); },
    onError: (reason) => setError(errorText(reason)),
  });

  async function load() {
    setError("");
    const [listResult, activeResult] = await Promise.allSettled([getEvaluations(), getActiveEvaluation()]);
    if (listResult.status === "fulfilled") setRuns(listResult.value.items || []);
    if (activeResult.status === "fulfilled") {
      setProgress(activeResult.value);
      if (activeResult.value.status !== "idle" && activeResult.value.run_id) void watch(activeResult.value.run_id);
    }
    const failure = [listResult, activeResult].find((item) => item.status === "rejected");
    if (failure?.status === "rejected") setError(errorText(failure.reason));
  }
  useEffect(() => { void load(); }, []);

  async function run() {
    setBusy(true); setError(""); setDetail(undefined);
    try {
      const started = await postEvaluationRun();
      setProgress(started);
      if (!started.run_id) throw new Error(language === "zh" ? "评测任务未返回 run_id" : "Evaluation did not return a run_id");
      await watch(started.run_id);
    } catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  }

  async function openRun(runId: string) {
    setBusy(true); setError("");
    try { setDetail(await getEvaluation(runId)); }
    catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  }

  return <main className="page-card">
    <div className="page-heading"><div><span className="eyebrow">Quality</span><h1>{text.evaluationTitle}</h1><p>{text.evaluationIntro}</p></div><div className="page-actions"><button className="primary-button" type="button" onClick={run} disabled={busy}>{text.runEvaluation}</button><button type="button" onClick={load} disabled={busy}>{text.refresh}</button></div></div>
    {error && <div className="alert error" role="alert">{error}</div>}
    {progress && progress.status !== "idle" && <section className="dashboard-panel"><h2>{language === "zh" ? "评测进度" : "Evaluation progress"}</h2><progress max="100" value={progress.percent || 0} /><p>{localizeRuntime(language, progress.current_question || progress.message || progress.current_profile || "-")} · {progress.percent || 0}%</p></section>}
    <section className="dashboard-panel"><h2>{text.history}</h2>{runs.length ? <div className="evaluation-history">{runs.map((run) => <button type="button" key={run.run_id} onClick={() => openRun(run.run_id)}><strong>{run.run_id}</strong><small>{run.created_at || ""} · {run.profile_count || 0} profiles · {run.question_count || 0} cases</small></button>)}</div> : <p className="empty-state">{text.noEvaluations}</p>}</section>
    {detail && <EvaluationDetail detail={detail} language={language} />}
  </main>;
}

function EvaluationDetail({ detail, language }: { detail: EvaluationRun; language: UiLanguage }) {
  const text = p(language);
  const overall = detail.summary || {};
  const profiles = detail.profiles || [];
  return <section className="evaluation-results">
    <article className="dashboard-panel"><h2>{text.questionSet}</h2><p><strong>{localizeRuntime(language, overall.best_profile_name || overall.best_profile_id || "-")}</strong></p><p>{localizeRuntime(language, overall.best_reason || "-")}</p><pre>{JSON.stringify(detail.question_generation || {}, null, 2)}</pre></article>
    <h2>{text.profileMetrics}</h2>{profiles.map((profile) => {
      const summary = object(profile.summary); const cases = array(profile.cases);
      return <article className="dashboard-panel" key={String(profile.id || profile.name)}><div className="section-heading"><div><h3>{localizeRuntime(language, profile.name || profile.id || "profile")}</h3><small>{strings(profile.enabled_switches).join(", ")}</small></div><span className="score-chip">{percent(summary.quality_score)}</span></div>
        <MetricRows values={[[text.successRate, percent(summary.success_rate)], [text.sourceHitRate, percent(summary.source_hit_rate)], [text.graphCoverage, percent(summary.graph_path_coverage)], [text.latency, `${number(summary.avg_latency_ms)} ms`], [text.recommendation, localizeRuntime(language, summary.recommendation || "-")]]} />
        <h3>{text.cases}</h3><div className="case-list">{cases.map((caseItem, index) => <CaseDetails key={String(caseItem.question_id || index)} item={caseItem} language={language} />)}</div>
      </article>;
    })}
  </section>;
}

function CaseDetails({ item, language }: { item: Record<string, unknown>; language: UiLanguage }) {
  const text = p(language);
  return <details><summary><button type="button" onClick={(event) => { event.preventDefault(); event.currentTarget.closest("details")?.toggleAttribute("open"); }}>{String(item.question || "-")}</button></summary>
    <div className="case-content">{item.error ? <div className="alert error"><strong>{text.error}:</strong> {String(item.error)}</div> : null}<h4>{text.answer}</h4><p>{String(item.answer || "-")}</p><h4>{text.metrics}</h4><pre>{JSON.stringify(item.metrics || {}, null, 2)}</pre><h4>{text.references}</h4>{array(item.references).map((reference, index) => <article className="evidence-item" key={index}><strong>{String(reference.source_file || reference.source || "-")}</strong><p>{String(reference.content || reference.text || "")}</p></article>)}<h4>{text.paths}</h4>{array(item.relation_paths).map((path, index) => <p key={index}>{display(path.path || path.relation_path)}</p>)}<h4>{text.trace}</h4>{array(item.trace).map((trace, index) => <div className="trace-line" key={index}><strong>{String(trace.node || "step")}</strong><span>{String(trace.detail || trace.message || display(trace.output || trace))}</span></div>)}</div>
  </details>;
}

function MetricRows({ values }: { values: Array<[string, string]> }) { return <div className="metric-grid">{values.map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>; }
function object(value: unknown): Record<string, unknown> { return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {}; }
function array(value: unknown): Array<Record<string, unknown>> { return Array.isArray(value) ? value.map(object) : []; }
function strings(value: unknown): string[] { return Array.isArray(value) ? value.map(String) : []; }
function number(value: unknown): number { const parsed = Number(value); return Number.isFinite(parsed) ? parsed : 0; }
function percent(value: unknown): string { return `${(number(value) * 100).toFixed(1)}%`; }
function display(value: unknown): string { return Array.isArray(value) ? value.join(" → ") : typeof value === "string" ? value : JSON.stringify(value || {}); }
function errorText(reason: unknown): string { return reason instanceof Error ? reason.message : String(reason); }
