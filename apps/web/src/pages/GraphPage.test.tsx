import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App } from "../App";
import { mockApi } from "../test/fixtures";

const graph = {
  nodes: [
    { id: "n1", label: "CrabRAG", properties: { type: "系统", source_files: ["spec.md"] } },
    { id: "n2", label: "索引治理", properties: { type: "能力", source_files: ["governance.md"] } },
  ],
  edges: [{ id: "e1", source: "n1", target: "n2", label: "包含", properties: { evidence: "支持双代索引", source_file: "governance.md", confidence: 0.9 } }],
  stats: { node_count: 2, edge_count: 1, source_file_count: 2, evidence_source_file_count: 1, graph_source: "dynamic_graph", graph_source_label: "知识库动态图谱" },
};

describe("knowledge graph", () => {
  test("renders API graph in an SVG radial disk, filters focus and shows node and edge evidence", async () => {
    const fetchMock = mockApi({ "/api/graph": graph });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "知识图谱" }));

    expect(await screen.findByText("知识库动态图谱")).not.toBeNull();
    expect(screen.getByText("2 个节点")).not.toBeNull();
    expect(screen.getByTestId("radial-disk-graph").querySelectorAll("circle")).toHaveLength(2);
    await user.type(screen.getByRole("searchbox", { name: "搜索节点" }), "索引");
    expect(screen.getByTestId("graph-node-n2").getAttribute("data-focused")).toBe("true");
    await user.click(screen.getByText("索引治理", { selector: "text" }));
    expect(screen.getByText("governance.md")).not.toBeNull();
    await user.click(screen.getByText("包含", { selector: "text" }));
    expect(screen.getByText("支持双代索引")).not.toBeNull();
  });

  test("refreshes, renders empty and error states, and switches all labels to English", async () => {
    const fetchMock = mockApi({
      "/api/graph": { nodes: [], edges: [], stats: { node_count: 0, edge_count: 0, graph_source: "empty_graph", graph_source_label: "暂无知识图谱" } },
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "English" }));
    await user.click(screen.getByRole("button", { name: "Knowledge Graph" }));
    expect(await screen.findByText("No knowledge graph")).not.toBeNull();
    await user.click(screen.getByRole("button", { name: "Refresh graph" }));
    await waitFor(() => expect(fetchMock.mock.calls.filter(([url]) => url === "/api/graph")).toHaveLength(2));
  });
});
