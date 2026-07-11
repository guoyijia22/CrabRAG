import { postChat } from "./client";

test("the browser API client never adds self-reported CrabRAG identity headers", async () => {
  const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) =>
    new Response(JSON.stringify({ session_id: "s", answer: "ok" })),
  );
  vi.stubGlobal("fetch", fetchMock);

  await postChat({ question: "hello" });

  const [, init] = fetchMock.mock.calls[0];
  const headers = new Headers(init?.headers);
  expect([...headers.keys()].filter((name) => name.startsWith("x-crabrag-"))).toEqual([]);
  expect(headers.get("content-type")).toBe("application/json");
});
