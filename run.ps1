param(
    [int]$WebPort = 3003,
    [int]$ApiPort = 8001,
    [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[CrabRAG] $Message"
}

function Fail {
    param([string]$Message)
    Write-Error "[CrabRAG] $Message"
    exit 1
}

if ($Help) {
    Write-Host "Usage: .\run.ps1 [-WebPort 3003] [-ApiPort 8001]"
    Write-Host "Starts the FastAPI backend and Bun web gateway."
    exit 0
}

$Root = $PSScriptRoot
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$PortablePython = Join-Path $Root "runtime\python\python.exe"
$PortableBun = Join-Path $Root "runtime\bun\bun.exe"
$GatewayPath = Join-Path $Root "server/gateway.js"
$GatewayArgument = '"' + $GatewayPath + '"'
$RunStatePath = Join-Path $Root "data\run.json"

function Resolve-Python {
    if (Test-Path $VenvPython) {
        return $VenvPython
    }
    if (Test-Path $PortablePython) {
        return $PortablePython
    }
    Fail "Python runtime not found. Run .\install.ps1 first."
}

function Resolve-Bun {
    if (Test-Path $PortableBun) {
        return $PortableBun
    }
    $bun = Get-Command "bun" -ErrorAction SilentlyContinue
    if ($null -ne $bun) {
        return $bun.Source
    }
    Fail "Bun was not found. Install Bun from https://bun.sh/docs/installation, then rerun .\run.ps1."
}

function Test-PortFree {
    param([int]$Port)
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse("127.0.0.1"), $Port)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($null -ne $listener) {
            $listener.Stop()
        }
    }
}

if (-not (Test-PortFree -Port $ApiPort)) {
    Fail "API port $ApiPort is already in use."
}
if (-not (Test-PortFree -Port $WebPort)) {
    Fail "Web port $WebPort is already in use."
}

$Python = Resolve-Python
$Bun = Resolve-Bun

$env:CRABRAG_ROOT = $Root
$env:ELCQA_ROOT = $Root
$env:CRABRAG_ENV_FILE = Join-Path $Root "config\.env"
$env:ELCQA_ENV_FILE = $env:CRABRAG_ENV_FILE
$env:RAG_BASE_URL = "http://127.0.0.1:$ApiPort"
$env:PORT = "$WebPort"
if (-not $env:CRABRAG_INTERNAL_TOKEN) {
    $env:CRABRAG_INTERNAL_TOKEN = [guid]::NewGuid().ToString("N")
}
$env:PYTHONUTF8 = "1"
$env:PYTHONNOUSERSITE = "1"
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$Root;$($env:PYTHONPATH)"
} else {
    $env:PYTHONPATH = $Root
}

$apiProcess = $null
$webProcess = $null

try {
    Write-Step "Starting API on http://127.0.0.1:$ApiPort"
    $apiProcess = Start-Process -FilePath $Python -ArgumentList @("-m", "uvicorn", "services.rag_api.main:app", "--host", "127.0.0.1", "--port", "$ApiPort") -WorkingDirectory $Root -PassThru -NoNewWindow

    Start-Sleep -Seconds 2

    Write-Step "Starting web gateway on http://127.0.0.1:$WebPort"
    $webProcess = Start-Process -FilePath $Bun -ArgumentList @($GatewayArgument) -WorkingDirectory $Root -PassThru -NoNewWindow

    $runState = @{
        schema_version = 1
        project_root = $Root
        web_port = $WebPort
        api_port = $ApiPort
        pids = @($apiProcess.Id, $webProcess.Id)
        processes = @(
            @{ pid = $apiProcess.Id; role = "api"; start_identity = $apiProcess.StartTime.ToUniversalTime().Ticks.ToString() },
            @{ pid = $webProcess.Id; role = "web"; start_identity = $webProcess.StartTime.ToUniversalTime().Ticks.ToString() }
        )
        started_at = [DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json
    $runStateTemp = "$RunStatePath.tmp"
    New-Item -ItemType Directory -Path (Split-Path -Parent $RunStatePath) -Force | Out-Null
    [System.IO.File]::WriteAllText($runStateTemp, $runState, [System.Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $runStateTemp -Destination $RunStatePath -Force

    Write-Step "CrabRAG is starting. Open http://127.0.0.1:$WebPort/. Press Ctrl+C to stop."
    while ($true) {
        if ($apiProcess.HasExited) {
            Fail "API process exited with code $($apiProcess.ExitCode)."
        }
        if ($webProcess.HasExited) {
            Fail "Web gateway exited with code $($webProcess.ExitCode)."
        }
        Start-Sleep -Seconds 1
    }
} finally {
    foreach ($process in @($webProcess, $apiProcess)) {
        if ($null -ne $process -and -not $process.HasExited) {
            Write-Step "Stopping process $($process.Id)."
            Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
        }
    }
    Remove-Item -LiteralPath $RunStatePath -Force -ErrorAction SilentlyContinue
}
