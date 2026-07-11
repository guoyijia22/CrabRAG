const PANEL_ID = "crabrag-governance-panel";
let loading = false;
let lastLoadedAt = 0;

function isKnowledgePage() {
  const active = document.querySelector("nav button.active");
  const label = active?.textContent?.trim() ?? "";
  return label.includes("Knowledge") || label.includes("知识库");
}

function panel() {
  let element = document.getElementById(PANEL_ID);
  if (element) return element;
  element = document.createElement("section");
  element.id = PANEL_ID;
  element.className = "crabrag-governance-panel";
  element.innerHTML = '<div class="crabrag-governance-header"><div><h2>Index Governance / 索引治理</h2><p>Atomic generations, document versions, cache and scheduled activation.</p></div><button type="button" data-action="refresh">Refresh</button></div><div data-content class="crabrag-governance-content">Loading…</div>';
  element.querySelector('[data-action="refresh"]').addEventListener("click", () => refresh(true));
  document.querySelector(".app")?.appendChild(element);
  return element;
}

function textRow(label, value) {
  const row = document.createElement("div");
  row.className = "crabrag-governance-row";
  const title = document.createElement("span");
  title.textContent = label;
  const content = document.createElement("strong");
  content.textContent = value ?? "-";
  row.append(title, content);
  return row;
}

function generationCard(title, generation) {
  const card = document.createElement("article");
  card.className = "crabrag-governance-card";
  const heading = document.createElement("h3");
  heading.textContent = title;
  card.append(
    heading,
    textRow("Generation", generation?.generation_id),
    textRow("Published", generation?.published_at),
    textRow("Documents", String(generation?.stats?.document_count ?? 0)),
    textRow("Chunks", String(generation?.stats?.chunk_count ?? 0)),
    textRow("Embeddings reused", String(generation?.stats?.reused_embedding_count ?? 0)),
    textRow("Embeddings rebuilt", String(generation?.stats?.embedded_chunk_count ?? 0))
  );
  return card;
}

function render(payload) {
  const content = panel().querySelector("[data-content]");
  content.replaceChildren();
  const grid = document.createElement("div");
  grid.className = "crabrag-governance-grid";
  grid.append(generationCard("Active / 当前", payload.active), generationCard("Previous / 上一代", payload.previous));
  const operations = document.createElement("article");
  operations.className = "crabrag-governance-card";
  const heading = document.createElement("h3");
  heading.textContent = "Operations / 运行状态";
  operations.append(
    heading,
    textRow("Next activation", payload.scheduler?.next_activation_at),
    textRow("Scheduler", payload.scheduler?.running ? "running" : "stopped"),
    textRow("Cache entries", String(payload.cache?.size ?? 0)),
    textRow("Cache hits / misses", `${payload.cache?.hits ?? 0} / ${payload.cache?.misses ?? 0}`),
    textRow("Last cleanup", payload.scheduler?.last_cleanup_at)
  );
  if (payload.can_rollback) {
    const rollback = document.createElement("button");
    rollback.type = "button";
    rollback.className = "crabrag-governance-rollback";
    rollback.textContent = "Rollback to previous / 回滚上一代";
    rollback.addEventListener("click", rollbackGeneration);
    operations.appendChild(rollback);
  }
  grid.appendChild(operations);
  content.appendChild(grid);
  const warnings = payload.active?.warnings ?? [];
  if (warnings.length) {
    const warning = document.createElement("div");
    warning.className = "crabrag-governance-warning";
    warning.textContent = `High-risk auto-public warnings / 自动公开告警: ${warnings.length} (${warnings.map(item => item.path).filter(Boolean).join(", ")})`;
    content.appendChild(warning);
  }
}

async function refresh(force = false) {
  if (loading || !isKnowledgePage()) return;
  if (!force && Date.now() - lastLoadedAt < 30000) return;
  loading = true;
  const content = panel().querySelector("[data-content]");
  try {
    const response = await fetch("/api/index/status");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
    lastLoadedAt = Date.now();
  } catch (error) {
    content.textContent = `Index governance unavailable: ${error.message}`;
  } finally {
    loading = false;
  }
}

async function rollbackGeneration() {
  if (!window.confirm("Rollback to the previous verified index generation? / 确认回滚到上一成功索引代？")) return;
  const response = await fetch("/api/index/rollback", { method: "POST" });
  if (!response.ok) {
    window.alert(`Rollback failed: HTTP ${response.status}`);
    return;
  }
  lastLoadedAt = 0;
  await refresh(true);
}

function syncVisibility() {
  const element = panel();
  const shouldHide = !isKnowledgePage();
  if (element.hidden !== shouldHide) element.hidden = shouldHide;
  if (!element.hidden) refresh();
}

const style = document.createElement("style");
style.textContent = `
  .crabrag-governance-panel{margin:18px;padding:18px;border:1px solid var(--border,#ddd);border-radius:10px;background:var(--surface,#fff);color:var(--text,#222)}
  .crabrag-governance-header{display:flex;align-items:center;justify-content:space-between;gap:16px}.crabrag-governance-header h2{margin:0 0 4px}.crabrag-governance-header p{margin:0;color:var(--muted,#666)}
  .crabrag-governance-header button,.crabrag-governance-rollback{border:1px solid var(--border,#ccc);border-radius:6px;padding:8px 12px;background:var(--surface-soft,#f7f7f7);color:inherit;cursor:pointer}
  .crabrag-governance-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:16px}.crabrag-governance-card{padding:14px;border:1px solid var(--border,#ddd);border-radius:8px}.crabrag-governance-card h3{margin:0 0 10px}
  .crabrag-governance-row{display:flex;justify-content:space-between;gap:12px;padding:5px 0}.crabrag-governance-row span{color:var(--muted,#666)}.crabrag-governance-row strong{overflow-wrap:anywhere;text-align:right}
  .crabrag-governance-warning{margin-top:12px;padding:10px;border-radius:6px;background:#fff3cd;color:#7a4b00}.crabrag-governance-rollback{margin-top:12px;width:100%}@media(max-width:900px){.crabrag-governance-grid{grid-template-columns:1fr}}
`;
document.head.appendChild(style);

new MutationObserver(syncVisibility).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["class"] });
window.addEventListener("DOMContentLoaded", syncVisibility);
syncVisibility();
