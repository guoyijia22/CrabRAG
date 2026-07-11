import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App } from "../App";
import { appSettings, mockApi } from "../test/fixtures";

const detail = {
  run_id: "eval-2", created_at: "2026-07-11 11:00:00", question_count: 1, profile_count: 1,
  question_generation: { source: "llm", question_count: 1, category_count: 1, generated_at: "2026-07-11 10:59:00" },
  summary: { best_profile_name: "全增强配置", best_reason: "来源命中更高" },
  profiles: [{ id: "enhanced", name: "全增强配置", enabled_switches: ["rerank_enabled"], summary: { success_rate: 1, source_hit_rate: 1, graph_path_coverage: 0.5, avg_latency_ms: 120, quality_score: 0.92, recommendation: "建议适用" }, cases: [{ question_id: "q1", question: "如何回滚？", answer: "选择上一代", category: "制度", references: [{ source_file: "guide.md", content: "回滚步骤" }], relation_paths: [{ path: "索引 -> 回滚 -> 上一代" }], trace: [{ node: "retrieve", detail: "hit" }], metrics: { source_hit: true }, error: "模拟错误" }] }],
};

describe("evaluation", () => {
  test("loads active and history, runs and polls a result with profile metrics and case evidence", async () => {
    const fetchMock = mockApi({
      "/api/evaluations": { items: [{ run_id: "eval-1", created_at: "2026-07-10", profile_count: 2, question_count: 3, summary: { best_profile_name: "基线" } }] },
      "/api/evaluations/active": { status: "idle" },
      "/api/evaluations/run": { run_id: "eval-2", status: "queued", percent: 0 },
      "/api/evaluations/eval-2/progress": { run_id: "eval-2", status: "completed", percent: 100, current_profile: "enhanced", current_question: "如何回滚？", message: "评测完成" },
      "/api/evaluations/eval-2": detail,
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "评测对比" }));
    expect(await screen.findByText("eval-1")).not.toBeNull();
    await user.click(screen.getByRole("button", { name: "运行评测" }));
    expect(await screen.findByText("92.0%")).not.toBeNull();
    expect(screen.getByText("来源命中更高")).not.toBeNull();
    await user.click(screen.getByRole("button", { name: "如何回滚？" }));
    expect(screen.getByText("模拟错误")).not.toBeNull();
    expect(screen.getByText("guide.md")).not.toBeNull();
    expect(screen.getByText("hit")).not.toBeNull();
    expect(document.body.textContent).not.toMatch(/300-500K|60\s*分钟|60\s*minutes/i);
  });

  test("renders English empty and permission error states", async () => {
    const fetchMock = mockApi({ "/api/evaluations": { items: [] }, "/api/evaluations/active": { status: "idle" } });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "English" }));
    await user.click(screen.getByRole("button", { name: "Evaluation" }));
    expect(await screen.findByText("No evaluation runs yet")).not.toBeNull();

    fetchMock.mockImplementation(async (input) => {
      const url = String(input);
      if (url === "/api/evaluations/run") return new Response(JSON.stringify({ detail: "index management permission required" }), { status: 403 });
      return new Response(JSON.stringify({ items: [] }), { status: 200 });
    });
    await user.click(screen.getByRole("button", { name: "Run evaluation" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("index management permission required"));
  });

  test("localizes backend profile, recommendation and progress text in English", async () => {
    vi.stubGlobal("fetch", mockApi({
      "/api/app-settings": { ...appSettings, ui_language: "en" },
      "/api/evaluations": { items: [] },
      "/api/evaluations/active": { status: "idle" },
      "/api/evaluations/run": { run_id: "eval-2", status: "queued", percent: 0 },
      "/api/evaluations/eval-2/progress": { run_id: "eval-2", status: "completed", percent: 100, message: "评测完成" },
      "/api/evaluations/eval-2": detail,
    }));
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "Evaluation" }));
    await user.click(screen.getByRole("button", { name: "Run evaluation" }));

    expect(await screen.findByText("All enhancements", { selector: "h3" })).not.toBeNull();
    expect(screen.getByText("Recommended")).not.toBeNull();
    expect(screen.getByText(/Evaluation completed/)).not.toBeNull();
  });

  test("reports active evaluation polling failures and stops after unmount", async () => {
    const fetchMock = mockApi({
      "/api/evaluations": { items: [] },
      "/api/evaluations/active": { run_id: "eval-error", status: "running", percent: 10 },
      "/api/evaluations/eval-error/progress": new Response(JSON.stringify({ detail: "evaluation poll unavailable" }), { status: 503 }),
    });
    vi.stubGlobal("fetch", fetchMock);
    const view = render(<App />);
    await userEvent.setup().click(await screen.findByRole("button", { name: "评测对比" }));

    expect((await screen.findByRole("alert")).textContent).toContain("evaluation poll unavailable");
    view.unmount();
    await new Promise((resolve) => setTimeout(resolve, 1050));
    expect(fetchMock.mock.calls.filter(([url]) => url === "/api/evaluations/eval-error/progress")).toHaveLength(1);
  });

  test("cancels a running evaluation timer when the page unmounts", async () => {
    const fetchMock = mockApi({
      "/api/evaluations": { items: [] },
      "/api/evaluations/active": { run_id: "eval-running", status: "running", percent: 10 },
      "/api/evaluations/eval-running/progress": { run_id: "eval-running", status: "running", percent: 20 },
    });
    vi.stubGlobal("fetch", fetchMock);
    const view = render(<App />);
    await userEvent.setup().click(await screen.findByRole("button", { name: "评测对比" }));
    await waitFor(() => expect(fetchMock.mock.calls.filter(([url]) => url === "/api/evaluations/eval-running/progress")).toHaveLength(1));
    view.unmount();

    await new Promise((resolve) => setTimeout(resolve, 1050));
    expect(fetchMock.mock.calls.filter(([url]) => url === "/api/evaluations/eval-running/progress")).toHaveLength(1);
  });
});
