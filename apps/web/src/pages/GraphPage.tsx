import { useEffect, useMemo, useState } from "react";

import { getGraph } from "../api/client";
import type { GraphEdge, GraphNode, GraphResponse, UiLanguage } from "../api/types";
import { p } from "../page-i18n";

interface Point { x: number; y: number }

export function crGraphDiskRingCount(count: number) {
  return count <= 12 ? 2 : count <= 35 ? 3 : count <= 90 ? 5 : count <= 180 ? 6 : 7;
}

export function crGraphRadialDiskLayout(nodes: GraphNode[], width = 900, height = 560): Map<string, Point> {
  const positions = new Map<string, Point>();
  if (!nodes.length) return positions;
  const center = { x: width / 2, y: height / 2 };
  const ringCount = Math.min(crGraphDiskRingCount(nodes.length), Math.max(1, Math.ceil(Math.sqrt(nodes.length / 3))));
  nodes.forEach((node, index) => {
    if (index === 0) { positions.set(node.id, center); return; }
    const ring = 1 + ((index - 1) % ringCount);
    const members = nodes.filter((_, candidate) => candidate > 0 && 1 + ((candidate - 1) % ringCount) === ring);
    const order = members.findIndex((item) => item.id === node.id);
    const angle = -Math.PI / 2 + (Math.PI * 2 * order) / Math.max(1, members.length);
    const radius = ring * (Math.min(width, height) * 0.42 / ringCount);
    positions.set(node.id, { x: center.x + Math.cos(angle) * radius, y: center.y + Math.sin(angle) * radius });
  });
  return positions;
}

const sourceLabels: Record<string, { zh: string; en: string }> = {
  dynamic_graph: { zh: "知识库动态图谱", en: "Dynamic knowledge graph" },
  subgraph: { zh: "问答命中子图", en: "Q&A matched subgraph" },
  empty_graph: { zh: "暂无知识图谱", en: "No knowledge graph" },
  static_graph: { zh: "内置基础图谱", en: "Built-in base graph" },
};

export function GraphPage({ language }: { language: UiLanguage }) {
  const text = p(language);
  const [graph, setGraph] = useState<GraphResponse>();
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<GraphNode | GraphEdge>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function load() {
    setBusy(true); setError(""); setSelected(undefined);
    try { setGraph(await getGraph()); }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(false); }
  }
  useEffect(() => { void load(); }, []);

  const positions = useMemo(() => crGraphRadialDiskLayout(graph?.nodes || []), [graph]);
  const normalized = query.trim().toLocaleLowerCase();
  const source = String(graph?.stats.graph_source || "");
  const sourceLabel = sourceLabels[source]?.[language] || String(graph?.stats.graph_source_label || source || "-");
  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];

  return <main className="page-card">
    <div className="page-heading"><div><span className="eyebrow">GraphRAG</span><h1>{text.graphTitle}</h1><p>{text.graphIntro}</p></div><div className="page-actions"><button type="button" onClick={load} disabled={busy}>{text.refreshGraph}</button></div></div>
    {error && <div className="alert error" role="alert">{error}</div>}
    <section className="graph-toolbar"><label><span>{text.searchNode}</span><input type="search" aria-label={text.searchNode} value={query} onChange={(event) => setQuery(event.target.value)} /></label><div className="tag-row">{source !== "empty_graph" && <span className="tag">{sourceLabel}</span>}<span className="tag">{`${nodes.length}${language === "zh" ? " " : ""}${text.nodes}`}</span><span className="tag">{`${edges.length}${language === "zh" ? " " : ""}${text.edges}`}</span></div></section>
    {!busy && !error && nodes.length === 0 ? <div className="empty-state large">{text.noGraph}</div> : <div className="graph-layout">
      <svg data-testid="radial-disk-graph" className="graph-canvas" viewBox="0 0 900 560" role="img" aria-label={text.graphTitle}>
        {edges.map((edge) => {
          const start = positions.get(edge.source); const end = positions.get(edge.target); if (!start || !end) return null;
          return <g key={edge.id} role="button" tabIndex={0} onClick={() => setSelected(edge)} onKeyDown={(event) => event.key === "Enter" && setSelected(edge)}>
            <line x1={start.x} y1={start.y} x2={end.x} y2={end.y} />
            <text className="edge-label" x={(start.x + end.x) / 2} y={(start.y + end.y) / 2} onClick={() => setSelected(edge)}>{edge.label}</text>
          </g>;
        })}
        {nodes.map((node) => {
          const point = positions.get(node.id)!;
          const focused = !normalized || node.label.toLocaleLowerCase().includes(normalized) || node.id.toLocaleLowerCase().includes(normalized);
          return <g key={node.id} data-testid={`graph-node-${node.id}`} data-focused={focused ? "true" : "false"} className="graph-node" opacity={focused ? 1 : .2} role="button" tabIndex={0} onClick={() => setSelected(node)} onKeyDown={(event) => event.key === "Enter" && setSelected(node)}>
            <circle cx={point.x} cy={point.y} r={node === selected ? 28 : 22} />
            <text x={point.x} y={point.y + 38} textAnchor="middle" onClick={() => setSelected(node)}>{node.label}</text>
          </g>;
        })}
      </svg>
      <aside className="detail-panel"><h2>{text.details}</h2>{selected ? <><strong>{selected.label}</strong><dl className="property-list">{Object.entries(selected.properties || {}).map(([key, item]) => <div key={key}><dt>{key === "source_file" || key === "source_files" ? text.source : key === "evidence" ? text.evidence : key}</dt><dd>{Array.isArray(item) ? item.join(", ") : String(item ?? "-")}</dd></div>)}</dl></> : <p className="empty-state">{language === "zh" ? "点击节点或边查看详情" : "Select a node or edge to inspect details"}</p>}</aside>
    </div>}
  </main>;
}
