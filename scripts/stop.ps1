Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$RunStatePath = Join-Path $Root "data\run.json"
$candidates = @(
    (Join-Path $Root ".venv\Scripts\python.exe"),
    (Join-Path $Root "runtime\python\python.exe")
)
$Python = $null
foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate) { $Python = $candidate; break }
}
if (-not $Python) {
    $command = Get-Command "python" -ErrorAction SilentlyContinue
    if ($null -ne $command) { $Python = $command.Source }
}
if (-not $Python) {
    Write-Error "[CrabRAG] Python runtime not found; refusing to stop unverified processes."
    exit 1
}

& $Python (Join-Path $PSScriptRoot "crabrag_admin.py") stop
exit $LASTEXITCODE
