import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App } from "../App";
import { appSettings, mockApi } from "../test/fixtures";

const health = {
  web: "ok", rag_service: "ok", docs_dir_exists: true, docs_dir_has_files: true,
  docs_dirs: ["D:/docs"], chroma: "ok:24", active_generation: "gen-2",
};
const indexStatus = {
  active_generation: "gen-2", previous_generation: "gen-1", can_rollback: true,
  active: { stats: { document_count: 4, chunk_count: 24, reused_embedding_count: 18, embedded_chunk_count: 6, embedding_dimension: 1024 }, warnings: [{ code: "AUTO_PUBLIC_DOCUMENT", path: "policy.md" }] },
  cache: { size: 3, hits: 12 }, scheduler: { running: true, next_activation_at: "2026-07-12 02:00:00", last_cleanup: { deleted: 2 } },
};

describe("knowledge base and index governance", () => {
  test("loads directory, schema suggestion and governed generation details without collection controls", async () => {
    const fetchMock = mockApi({
      "/api/app-settings": appSettings,
      "/api/health": health,
      "/api/ingest/active": { active: null, last_success: null },
      "/api/graph/schema/suggestion": { status: "pending", entity_types: ["制度"], node_fields: [{ key: "risk", label: "风险等级" }] },
      "/api/index/status": indexStatus,
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "知识库" }));
    expect(await screen.findByText("D:/docs")).not.toBeNull();
    expect(screen.getByText("ok:24")).not.toBeNull();
    expect(screen.getByText("gen-2")).not.toBeNull();
    expect(screen.getByText("gen-1")).not.toBeNull();
    expect(screen.getByText("18")).not.toBeNull();
    expect(screen.getByText("1024")).not.toBeNull();
    expect(screen.getByText("风险等级")).not.toBeNull();
    expect(screen.queryByText("知识库集合")).toBeNull();
    expect(screen.queryByText("Knowledge base collection")).toBeNull();
  });

  test("starts incremental and confirmed full rebuilds, reads result, confirms schema and rolls back", async () => {
    const confirm = vi.fn(() => true);
    vi.stubGlobal("confirm", confirm);
    const fetchMock = mockApi({
      "/api/health": health,
      "/api/ingest/active": { active: null, last_success: null },
      "/api/graph/schema/suggestion": { status: "pending", entity_types: ["制度"] },
      "/api/index/status": indexStatus,
      "/api/ingest/run": { run_id: "ingest-1", status: "queued", percent: 0 },
      "/api/ingest/full": { run_id: "ingest-2", status: "queued", percent: 0 },
      "/api/ingest/ingest-1/progress": { run_id: "ingest-1", status: "completed", percent: 100, current_step: "完成" },
      "/api/ingest/ingest-2/progress": { run_id: "ingest-2", status: "completed", percent: 100, current_step: "完成" },
      "/api/ingest/ingest-1": { generation_id: "gen-3", document_count: 5, chunk_count: 30, reused_embedding_count: 24, embedded_chunk_count: 6 },
      "/api/ingest/ingest-2": { generation_id: "gen-4", document_count: 5, chunk_count: 31, reused_embedding_count: 0, embedded_chunk_count: 31 },
      "/api/graph/schema": { status: "active", entity_types: ["制度"] },
      "/api/index/rollback": { status: "rolled_back", active_generation: "gen-1", previous_generation: "gen-2" },
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "知识库" }));

    await user.click(await screen.findByRole("button", { name: "增量更新" }));
    expect(await screen.findByText("gen-3")).not.toBeNull();
    await user.click(screen.getByRole("button", { name: "全量重建" }));
    expect(confirm).toHaveBeenCalledOnce();
    expect(await screen.findByText("gen-4")).not.toBeNull();
    await user.click(screen.getByRole("button", { name: "确认图谱结构" }));
    await user.click(screen.getByRole("button", { name: "回滚到上一代" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith("/api/graph/schema", expect.objectContaining({ method: "PUT" }));
      expect(fetchMock).toHaveBeenCalledWith("/api/index/rollback", expect.objectContaining({ method: "POST" }));
    });
  });

  test("shows localized permission denial and hides administrator actions", async () => {
    const fetchMock = mockApi({
      "/api/health": health,
      "/api/ingest/active": new Response(JSON.stringify({ detail: "需要索引管理权限" }), { status: 403 }),
      "/api/graph/schema/suggestion": new Response(JSON.stringify({ detail: "需要索引管理权限" }), { status: 403 }),
      "/api/index/status": new Response(JSON.stringify({ detail: "index management permission required" }), { status: 403 }),
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "知识库" }));

    expect(await screen.findByText("需要索引管理权限")).not.toBeNull();
    expect(screen.queryByRole("button", { name: "回滚到上一代" })).toBeNull();
  });
});
