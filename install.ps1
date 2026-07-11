param(
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
    Write-Host "Usage: .\install.ps1"
    Write-Host "Creates .venv, installs Python dependencies, runs bun install, and copies config\.env.example to config\.env when missing."
    exit 0
}

if ($PSVersionTable.PSVersion.Major -lt 5) {
    Fail "PowerShell 5.0 or newer is required. Current version: $($PSVersionTable.PSVersion)"
}

$Root = $PSScriptRoot
$VenvDir = Join-Path $Root ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$PortablePython = Join-Path $Root "runtime\python\python.exe"
$Requirements = Join-Path $Root "requirements.txt"
$EnvExample = Join-Path $Root "config\.env.example"
$EnvPath = Join-Path $Root "config\.env"
$PortableBun = Join-Path $Root "runtime\bun\bun.exe"
$PortableBunDir = Join-Path $Root "runtime\bun"
$PackageJson = Join-Path $Root "package.json"
$ReleaseManifest = Join-Path $Root "release-manifest.json"
$BunVersion = "1.3.14"
$BunWindowsX64Sha256 = "0a0620930b6675d7ba440e81f4e0e00d3cfbe096c4b140d3fff02205e9e18922"

function Test-PythonCandidate {
    param(
        [string]$Command,
        [string[]]$Arguments
    )

    $versionCode = "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info < (3, 14) else 1)"
    try {
        & $Command @Arguments -c $versionCode *> $null
        if ($LASTEXITCODE -eq 0) {
            return [pscustomobject]@{ Command = $Command; Arguments = $Arguments }
        }
    } catch {
        return $null
    }
    return $null
}

function Find-Python {
    $candidates = @()
    if (Test-Path $PortablePython) {
        $candidates += [pscustomobject]@{ Command = $PortablePython; Arguments = @() }
    }
    $candidates += @(
        [pscustomobject]@{ Command = "python"; Arguments = @() },
        [pscustomobject]@{ Command = "python3"; Arguments = @() },
        [pscustomobject]@{ Command = "py"; Arguments = @("-3.13") },
        [pscustomobject]@{ Command = "py"; Arguments = @("-3.12") },
        [pscustomobject]@{ Command = "py"; Arguments = @("-3.11") },
        [pscustomobject]@{ Command = "py"; Arguments = @("-3.10") }
    )
    foreach ($candidate in $candidates) {
        $found = Test-PythonCandidate -Command $candidate.Command -Arguments $candidate.Arguments
        if ($null -ne $found) {
            return $found
        }
    }
    Fail "Supported Python 3.10-3.13 was not found. Python 3.14 is not yet supported. Install Python from https://www.python.org/downloads/ and rerun .\install.ps1."
}

function Require-Command {
    param(
        [string]$Name,
        [string]$InstallHint
    )
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        Fail "$Name was not found. $InstallHint"
    }
    return $command.Source
}

function Test-BunVersion {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    try {
        return ((& $Path --version 2>$null | Select-Object -First 1).Trim() -eq $BunVersion)
    } catch {
        return $false
    }
}

function Resolve-BunForInstall {
    if (Test-BunVersion -Path $PortableBun) {
        return $PortableBun
    }
    $command = Get-Command "bun" -ErrorAction SilentlyContinue
    if ($null -ne $command -and (Test-BunVersion -Path $command.Source)) {
        return $command.Source
    }

    Install-PortableBun
    if (Test-BunVersion -Path $PortableBun) {
        return $PortableBun
    }
    Fail "Failed to install project-local Bun $BunVersion."
}

function Install-PortableBun {
    Write-Step "Downloading verified project-local Bun $BunVersion to runtime\bun."
    New-Item -ItemType Directory -Path $PortableBunDir -Force | Out-Null
    $archive = Join-Path ([System.IO.Path]::GetTempPath()) "crabrag-bun-$PID.zip"
    $extractDir = Join-Path ([System.IO.Path]::GetTempPath()) "crabrag-bun-$PID"
    try {
        Invoke-WebRequest -Uri "https://github.com/oven-sh/bun/releases/download/bun-v1.3.14/bun-windows-x64.zip" -OutFile $archive
        $actualSha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualSha256 -ne $BunWindowsX64Sha256) {
            Fail "Downloaded Bun archive checksum mismatch."
        }
        Expand-Archive -LiteralPath $archive -DestinationPath $extractDir -Force
        $bunExe = Get-ChildItem -LiteralPath $extractDir -Recurse -Filter "bun.exe" | Select-Object -First 1
        if ($null -eq $bunExe) {
            Fail "Downloaded Bun archive did not contain bun.exe."
        }
        Copy-Item -LiteralPath $bunExe.FullName -Destination $PortableBun -Force
    } catch {
        Fail "Failed to download project-local Bun."
    } finally {
        if (Test-Path -LiteralPath $archive) {
            Remove-Item -LiteralPath $archive -Force
        }
        if (Test-Path -LiteralPath $extractDir) {
            Remove-Item -LiteralPath $extractDir -Recurse -Force
        }
    }
}

function Use-SafePipIndexIfNeeded {
    param([string]$PythonExe)

    $configuredIndex = ""
    try {
        $configuredIndex = (& $PythonExe -m pip config get global.index-url 2>$null | Select-Object -First 1).Trim()
    } catch {
        $configuredIndex = ""
    }

    $effectiveIndex = $env:PIP_INDEX_URL
    if (-not $effectiveIndex) {
        $effectiveIndex = $configuredIndex
    }

    if ($effectiveIndex -match "^http://" -and -not $env:PIP_TRUSTED_HOST) {
        $env:PIP_INDEX_URL = "https://pypi.org/simple"
        Write-Step "Detected an insecure HTTP pip index. Using https://pypi.org/simple for this install. Set PIP_INDEX_URL to override."
    }
}

Write-Step "Repository root: $Root"

foreach ($dir in @("docs", "data", "logs", "runtime", "runtime\models")) {
    $path = Join-Path $Root $dir
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

if (-not (Test-Path $EnvExample)) {
    Fail "Missing config\.env.example."
}

if (Test-Path $EnvPath) {
    Write-Step "Keeping existing config\.env."
} else {
    Copy-Item -Path $EnvExample -Destination $EnvPath
    Write-Step "Created config\.env from config\.env.example."
}

$python = Find-Python
if (-not (Test-Path $VenvPython)) {
    Write-Step "Creating Python virtual environment in .venv."
    # Equivalent command: python -m venv .venv
    & $python.Command @($python.Arguments + @("-m", "venv", $VenvDir))
    if ($LASTEXITCODE -ne 0) {
        Fail "Failed to create .venv."
    }
} else {
    Write-Step "Reusing existing .venv."
}

Use-SafePipIndexIfNeeded -PythonExe $VenvPython

Write-Step "Installing Python dependencies."
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Fail "Failed to upgrade pip."
}
# Equivalent command: pip install -r requirements.txt
& $VenvPython -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) {
    Fail "Failed to install Python dependencies from requirements.txt."
}

$bun = Resolve-BunForInstall
foreach ($optional in @("node", "npm", "pnpm")) {
    $optionalCommand = Get-Command $optional -ErrorAction SilentlyContinue
    if ($null -ne $optionalCommand) {
        Write-Step "$optional detected: $($optionalCommand.Source)"
    } else {
        Write-Step "$optional not found; it is optional for this bundled frontend."
    }
}

Push-Location $Root
try {
    if (Test-Path -LiteralPath $PackageJson) {
        Write-Step "Installing JavaScript dependencies with Bun."
        # Equivalent command: bun install
        if (Test-Path -LiteralPath $ReleaseManifest) {
            if (-not (Test-Path (Join-Path $Root "bun.lock"))) {
                Fail "Release package is missing bun.lock."
            }
            & $bun install --production --frozen-lockfile
        } elseif (Test-Path (Join-Path $Root "bun.lock")) {
            & $bun install --frozen-lockfile
        } else {
            & $bun install
        }
        if ($LASTEXITCODE -ne 0) {
            Fail "bun install failed."
        }
    } else {
        Write-Step "Skipping JavaScript dependency install because this release uses the bundled gateway."
    }
} finally {
    Pop-Location
}

Write-Step "Running smoke check."
& $VenvPython (Join-Path $Root "scripts\check_env.py")
if ($LASTEXITCODE -ne 0) {
    Fail "Smoke check failed."
}

Write-Step "Install completed. Run .\run.ps1 and open http://127.0.0.1:3003/."
