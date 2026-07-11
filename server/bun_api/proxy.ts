export type Fetcher = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

export async function proxyJson(fetcher: Fetcher, input: string, init?: RequestInit): Promise<Response> {
  const response = await fetcher(input, init);
  return Response.json(await response.json(), { status: response.status });
}

export async function proxyBinary(fetcher: Fetcher, input: string, init?: RequestInit): Promise<Response> {
  const response = await fetcher(input, init);
  return new Response(response.body, {
    status: response.status,
    headers: {
      "content-type": response.headers.get("content-type") ?? "application/octet-stream",
    },
  });
}
