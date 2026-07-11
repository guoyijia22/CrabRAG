Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$RunStatePath = Join-Path $Root "data\run.json"

if (-not (Test-Path -LiteralPath $RunStatePath)) {
    Write-Host "[CrabRAG] No project run state was found."
    exit 0
}

try {
    $state = Get-Content -LiteralPath $RunStatePath -Raw | ConvertFrom-Json
} catch {
    Write-Error "[CrabRAG] Project run state is invalid; refusing to stop unverified processes."
    exit 1
}

$stateRoot = [System.IO.Path]::GetFullPath([string]$state.project_root).TrimEnd('\')
$expectedRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\')
if (-not $stateRoot.Equals($expectedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "[CrabRAG] Run state belongs to another project; refusing to stop processes."
    exit 1
}

$ports = @([int]$state.web_port, [int]$state.api_port)
$stopped = 0
foreach ($processId in @($state.pids)) {
    $processId = [int]$processId
    $ownsExpectedPort = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.OwningProcess -eq $processId -and $_.LocalPort -in $ports } |
        Select-Object -First 1
    if ($null -eq $ownsExpectedPort) {
        continue
    }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
    if ($null -eq $process -or $process.CommandLine -notmatch 'uvicorn|server[\\/]gateway\.js') {
        continue
    }
    Stop-Process -Id $processId -Force
    $stopped += 1
}

Remove-Item -LiteralPath $RunStatePath -Force
Write-Host "[CrabRAG] Stopped $stopped verified service process(es)."
exit 0
