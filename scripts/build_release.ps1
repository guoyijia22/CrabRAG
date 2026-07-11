param(
    [string]$Version = "",
    [string]$OutputDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
if (-not $Version) {
    $Version = (Get-Content -LiteralPath (Join-Path $Root "VERSION") -Raw).Trim()
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $Root "release"
}

$candidates = @(
    (Join-Path $Root ".venv\Scripts\python.exe"),
    (Join-Path $Root "runtime\python\python.exe")
)
$Python = $null
foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate) {
        $Python = $candidate
        break
    }
}
if (-not $Python) {
    $command = Get-Command "python" -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        $Python = $command.Source
    }
}
if (-not $Python) {
    throw "Python runtime not found. Run install.ps1 first."
}

& $Python (Join-Path $PSScriptRoot "build_release.py") --root $Root --output-dir $OutputDir --version $Version
exit $LASTEXITCODE
