import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App } from "../App";
import { mockApi } from "../test/fixtures";

describe("query logs", () => {
  test("loads, filters by category and expands traceable log details", async () => {
    const fetchMock = mockApi({
      "/api/logs": { items: [{ session_id: "log-1", time: "2026-07-11 10:00:00", intent: "制度", question: "如何回滚？", answer: "选择上一代。", retrieval_mode: "hybrid", sources: ["guide.md"], entities: ["索引"], error: null }] },
      "/api/logs?intent=%E5%88%B6%E5%BA%A6": { items: [{ id: "log-1", created_at: "2026-07-11 10:00:00", intent: "制度", question: "如何回滚？", answer: "选择上一代。", references: [], trace: [] }] },
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "日志" }));
    expect(await screen.findByText("如何回滚？")).not.toBeNull();
    await user.click(screen.getByRole("button", { name: "如何回滚？" }));
    expect(screen.getByText("选择上一代。")).not.toBeNull();
    expect(screen.getByText("guide.md")).not.toBeNull();
    await user.selectOptions(screen.getByLabelText("日志分类"), "制度");
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/logs?intent=%E5%88%B6%E5%BA%A6", undefined));
  });

  test("shows empty and permission error states", async () => {
    vi.stubGlobal("fetch", mockApi({ "/api/logs": { items: [] } }));
    const user = userEvent.setup();
    const first = render(<App />);
    await user.click(await screen.findByRole("button", { name: "日志" }));
    expect(await screen.findByText("暂无日志")).not.toBeNull();
    first.unmount();

    vi.stubGlobal("fetch", mockApi({ "/api/logs": new Response(JSON.stringify({ detail: "需要索引管理权限" }), { status: 403 }) }));
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "日志" }));
    expect(await screen.findByText("需要索引管理权限")).not.toBeNull();
  });
});
