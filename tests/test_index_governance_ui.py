from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gateway_injects_configured_identity_and_proxies_index_governance_routes():
    gateway = (ROOT / "server" / "gateway.js").read_text(encoding="utf-8")

    assert "CRABRAG_INTERNAL_TOKEN" in gateway
    assert '"x-crabrag-subject"' in gateway
    assert 'indexRoute.get("/index/status"' in gateway
    assert 'indexRoute.post("/index/rollback"' in gateway
    assert 'app.route("/api", indexRoute)' in gateway


def test_gateway_forwards_trusted_identity_to_all_admin_routes():
    gateway = (ROOT / "server" / "gateway.js").read_text(encoding="utf-8")

    expected_calls = [
        'fetch(`${RAG_BASE_URL}/api/evaluations/run`, { method: "POST", headers: ragHeaders() })',
        'fetch(`${RAG_BASE_URL}/api/evaluations`, { headers: ragHeaders() })',
        'fetch(`${RAG_BASE_URL}/api/evaluations/active`, { headers: ragHeaders() })',
        'fetch(`${RAG_BASE_URL}/api/evaluations/${encodeURIComponent(runId)}/progress`, { headers: ragHeaders() })',
        'fetch(`${RAG_BASE_URL}/api/evaluations/${encodeURIComponent(runId)}`, { headers: ragHeaders() })',
        'fetch(`${RAG_BASE_URL}/api/ingest`, { method: "POST", headers: ragHeaders() })',
        'fetch(`${RAG_BASE_URL}/api/ingest/run`, { method: "POST", headers: ragHeaders() })',
        'fetch(`${RAG_BASE_URL}/api/ingest/full`, { method: "POST", headers: ragHeaders() })',
        'fetch(`${RAG_BASE_URL}/api/ingest/active`, { headers: ragHeaders() })',
        'fetch(`${RAG_BASE_URL}/api/ingest/${encodeURIComponent(runId)}/progress`, { headers: ragHeaders() })',
        'fetch(`${RAG_BASE_URL}/api/ingest/${encodeURIComponent(runId)}`, { headers: ragHeaders() })',
        'fetch(`${RAG_BASE_URL}/api/logs${qs}`, { headers: ragHeaders() })',
        'fetch(`${RAG_BASE_URL}/api/graph/schema`, { headers: ragHeaders() })',
        'fetch(`${RAG_BASE_URL}/api/graph/schema/suggestion`, { headers: ragHeaders() })',
        'fetch(`${RAG_BASE_URL}/api/graph/schema`, {\n    method: "PUT",\n    headers: ragHeaders(true),',
    ]

    for expected in expected_calls:
        assert expected in gateway


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
