from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gateway_injects_configured_identity_and_proxies_index_governance_routes():
    config = (ROOT / "server" / "bun_api" / "config.ts").read_text(encoding="utf-8")
    routes = (ROOT / "server" / "bun_api" / "routes.ts").read_text(encoding="utf-8")
    entrypoint = (ROOT / "server" / "bun_api" / "index.ts").read_text(encoding="utf-8")

    assert "CRABRAG_INTERNAL_TOKEN" in config
    assert 'headers.set("x-crabrag-subject", config.subject)' in config
    assert 'app.get("/index/status"' in routes
    assert 'app.post("/index/rollback"' in routes
    assert 'app.route("/api", createApiRoutes(' in entrypoint


def test_gateway_forwards_trusted_identity_to_all_admin_routes():
    routes = (ROOT / "server" / "bun_api" / "routes.ts").read_text(encoding="utf-8")

    expected_calls = [
        'api("/evaluations/run"), { method: "POST", headers: governed() }',
        'api("/evaluations"), { headers: governed() }',
        'api("/evaluations/active"), { headers: governed() }',
        'api(`/evaluations/${encodeURIComponent(c.req.param("runId"))}/progress`), { headers: governed() }',
        'api(`/evaluations/${encodeURIComponent(c.req.param("runId"))}`), { headers: governed() }',
        'api("/ingest"), { method: "POST", headers: governed() }',
        'api("/ingest/run"), { method: "POST", headers: governed() }',
        'api("/ingest/full"), { method: "POST", headers: governed() }',
        'api("/ingest/active"), { headers: governed() }',
        'api(`/ingest/${encodeURIComponent(c.req.param("runId"))}/progress`), { headers: governed() }',
        'api(`/ingest/${encodeURIComponent(c.req.param("runId"))}`), { headers: governed() }',
        '`${api("/logs")}${new URL(c.req.url).search}`, { headers: governed() }',
        'api("/graph/schema"), { headers: governed() }',
        'api("/graph/schema/suggestion"), { headers: governed() }',
        'api("/graph/schema"), {\n    method: "PUT", headers: governed(true),',
    ]

    for expected in expected_calls:
        assert expected in routes


def test_gateway_bundle_has_maintainable_rebuildable_source_contract():
    package = (ROOT / "package.json").read_text(encoding="utf-8")
    entrypoint = (ROOT / "server" / "bun_api" / "index.ts").read_text(encoding="utf-8")
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")

    for source_name in ("app-config.ts", "config.ts", "index.ts", "proxy.ts", "routes.ts", "static.ts"):
        assert (ROOT / "server" / "bun_api" / source_name).is_file()
    assert '"hono": "4.12.29"' in package
    assert '"build:gateway": "bun build server/bun_api/index.ts --target bun --outfile server/gateway.js"' in package
    assert "if (import.meta.main)" in entrypoint
    assert "server/gateway.js text eol=lf" in attributes


def test_start_scripts_generate_shared_internal_token_when_missing():
    powershell = (ROOT / "run.ps1").read_text(encoding="utf-8")
    shell = (ROOT / "run.sh").read_text(encoding="utf-8")

    assert "CRABRAG_INTERNAL_TOKEN" in powershell
    assert "[guid]::NewGuid()" in powershell
    assert "CRABRAG_INTERNAL_TOKEN" in shell
    assert "secrets.token_urlsafe" in shell


def test_governance_panel_is_loaded_as_readable_standalone_module():
    html = (ROOT / "apps" / "web" / "dist" / "index.html").read_text(encoding="utf-8")
    module = (ROOT / "apps" / "web" / "dist" / "index-governance.js").read_text(encoding="utf-8")

    assert 'src="/index-governance.js"' in html
    assert "crabrag-governance-panel" in module
    assert "/api/index/status" in module
    assert "/api/index/rollback" in module


def test_index_governance_manifest_and_operations_are_documented():
    chinese = (ROOT / "README_ZH.md").read_text(encoding="utf-8")
    english = (ROOT / "README.md").read_text(encoding="utf-8")

    for text in (chinese, english):
        assert ".crabrag-manifest.json" in text
        assert "/api/index/status" in text
        assert "CRABRAG_INTERNAL_TOKEN" in text
        assert "draft" in text and "published" in text and "retired" in text
