import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App } from "./App";
import { appSettings, mockApi, modelSettings, ragSettings } from "./test/fixtures";

describe("application shell", () => {
  test("shows the v1.1.0 CrabRAG default and all Chinese and English navigation labels", async () => {
    vi.stubGlobal("fetch", mockApi());
    const user = userEvent.setup();
    render(<App />);

    expect((await screen.findAllByText("CrabRAG")).length).toBeGreaterThan(0);
    expect(screen.getByText("v1.1.0")).not.toBeNull();
    expect(screen.getByRole("button", { name: "问答" })).not.toBeNull();
    expect(screen.getByRole("button", { name: "索引治理" })).not.toBeNull();

    await user.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByRole("button", { name: "Q&A" })).not.toBeNull();
    expect(screen.getByRole("button", { name: "Index Governance" })).not.toBeNull();
  });

  test("uses the bundled crab by default and gives a persisted custom image priority", async () => {
    const fetchMock = mockApi();
    vi.stubGlobal("fetch", fetchMock);
    const { unmount } = render(<App />);

    expect((await screen.findByRole("img", { name: "侧边栏展示图" })).getAttribute("src")).toBe("/picture/crab.png");
    unmount();

    vi.stubGlobal("fetch", mockApi({
      "/api/app-settings": { ...appSettings, sidebar_image_url: "/api/app-assets/sidebar-image" },
    }));
    render(<App />);
    expect((await screen.findByRole("img", { name: "侧边栏展示图" })).getAttribute("src")).toMatch(
      /^\/api\/app-assets\/sidebar-image\?v=\d+$/,
    );
  });
});

describe("chat", () => {
  test("posts the question with a stable session and renders answer evidence with generation metadata", async () => {
    const fetchMock = mockApi({
      "/api/chat": {
        session_id: "session-7",
        index_generation: "gen-20260711",
        intent: "规范",
        question_type: "事实",
        retrieval_mode: "hybrid",
        entities: ["CrabRAG"],
        answer: "**答案**来自规范。",
        references: [{ source_file: "D:/private/spec.md", content: "证据片段", score: 0.91, document_id: "doc-1" }],
        relation_paths: [{ path: ["CrabRAG", "依据", "规范"], source_file: "D:/private/spec.md" }],
        trace: [{ node: "retrieve", detail: "找到证据" }],
        error: null,
      },
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await user.type(await screen.findByLabelText("输入问题"), "什么是规范？");
    await user.click(screen.getByRole("button", { name: "发送" }));

    expect(await screen.findByText("答案", { selector: "strong" })).not.toBeNull();
    expect(screen.getByText("gen-20260711")).not.toBeNull();
    expect(screen.getByText("证据片段")).not.toBeNull();
    expect(screen.getByText("CrabRAG → 依据 → 规范")).not.toBeNull();
    expect(screen.getByText("找到证据")).not.toBeNull();
    const calls = fetchMock.mock.calls.filter(([url]) => url === "/api/chat");
    expect(JSON.parse(String(calls[0][1]?.body))).toEqual({ question: "什么是规范？" });
  });

  test("submits with Enter and refreshes categories after a knowledge-base rebuild", async () => {
    const fetchMock = mockApi({
      "/api/chat": { session_id: "session-enter", answer: "ok", references: [], relation_paths: [], trace: [] },
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);

    const input = await screen.findByLabelText("输入问题");
    await user.type(input, "回车提交{enter}");
    await waitFor(() => expect(fetchMock.mock.calls.filter(([url]) => url === "/api/chat")).toHaveLength(1));
    window.dispatchEvent(new Event("crabrag:knowledge-base-rebuilt"));
    await waitFor(() => expect(fetchMock.mock.calls.filter(([url]) => url === "/api/categories")).toHaveLength(2));
  });
});

describe("settings", () => {
  test("loads and saves app, model and RAG settings without a knowledge-base-name field", async () => {
    const fetchMock = mockApi({
      "/api/app-settings": appSettings,
      "/api/model-settings": modelSettings,
      "/api/settings": ragSettings,
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "设置" }));
    expect(await screen.findByDisplayValue("D:/docs")).not.toBeNull();
    expect(screen.getByText("动态 Top-K")).not.toBeNull();
    expect(screen.getByText("父片段上下文")).not.toBeNull();
    expect(screen.queryByLabelText("知识库名称")).toBeNull();
    expect(screen.queryByText("不得显示的知识库名")).toBeNull();

    const name = screen.getByLabelText("系统名称");
    fireEvent.change(name, { target: { value: "CrabRAG Enterprise" } });
    await user.click(screen.getByRole("button", { name: "保存设置" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/app-settings",
        expect.objectContaining({ method: "PUT" }),
      );
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/model-settings",
        expect.objectContaining({ method: "PUT" }),
      );
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/settings",
        expect.objectContaining({ method: "PUT" }),
      );
    });
    expect(await screen.findByText("设置已保存")).not.toBeNull();
  });

  test("does not render evaluation-cost disabling copy", async () => {
    vi.stubGlobal("fetch", mockApi());
    const user = userEvent.setup();
    const { container } = render(<App />);
    await user.click(await screen.findByRole("button", { name: "评测对比" }));

    expect(within(container).queryByText(/评测.*消耗.*禁用|evaluation.*cost.*disabled/i)).toBeNull();
  });

  test("shows bilingual local model file locations and provider download links", async () => {
    vi.stubGlobal("fetch", mockApi({
      "/api/model-settings": {
        ...modelSettings,
        local_model_status: {
          base_dir: "runtime/models",
          missing_count: 1,
          models: [{ key: "llm", name: "Qwen", present: false, expected_dir: "runtime/models/qwen", required_files: ["model.onnx"], missing_files: ["model.onnx"], download_urls: { zh: "https://www.modelscope.cn/qwen", en: "https://huggingface.co/qwen" } }],
        },
      },
    }));
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "设置" }));

    expect(await screen.findByText("存放目录：")).not.toBeNull();
    expect(screen.getByRole("link", { name: "下载地址（ModelScope）" }).getAttribute("href")).toContain("modelscope.cn");
    await user.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByText("Save to:")).not.toBeNull();
    expect(screen.getByRole("link", { name: "Download (Hugging Face)" }).getAttribute("href")).toContain("huggingface.co");
  });

  test("keeps unsaved settings drafts when the shell language changes", async () => {
    const fetchMock = mockApi();
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "设置" }));

    const name = await screen.findByLabelText("系统名称");
    await user.clear(name);
    await user.type(name, "CrabRAG Draft");
    await user.click(screen.getByRole("button", { name: "English" }));

    expect(await screen.findByDisplayValue("CrabRAG Draft")).not.toBeNull();
    expect(fetchMock.mock.calls.filter(([url, init]) => url === "/api/app-settings" && !init)).toHaveLength(2);
  });
});
