# Verify-and-guide installer for Project ADA on Windows 10/11.
#
# This script does NOT install OS-level dependencies (CUDA driver, Ollama,
# ffmpeg, Python). It checks for them, prints what is missing with a direct
# install hint, then offers to create a Python venv, install the project,
# and pull the LLM model.
#
# Exit code: 0 on a clean run, even if external deps are missing.
# Non-zero only if a guided step (venv / pip / ollama pull) fails.

# Soft-fail on missing tools during checks; only guided actions use Stop.
$ErrorActionPreference = 'Continue'

$RepoRoot   = Split-Path -Parent $PSScriptRoot
$OllamaUrl  = if ($env:OLLAMA_URL) { $env:OLLAMA_URL } else { 'http://localhost:11434' }
$LlmModel   = 'qwen3:8b'
$ReqPyMajor = 3
$ReqPyMinor = 11

Set-Location -LiteralPath $RepoRoot

# Track whether anything blocks the guided steps.
$script:MissingHard      = $false
$script:HasOllamaCli     = $false
$script:HasOllamaDaemon  = $false
$script:HasModel         = $false

function Write-OK   ([string]$msg)              { Write-Host "  [OK]   $msg" -ForegroundColor Green }
function Write-Miss ([string]$msg, [string]$hint) {
    Write-Host "  [MISS] $msg" -ForegroundColor Red
    if ($hint) { Write-Host "    $hint" -ForegroundColor DarkGray }
}
function Write-Warn2 ([string]$msg, [string]$hint) {
    Write-Host "  [WARN] $msg" -ForegroundColor Yellow
    if ($hint) { Write-Host "    $hint" -ForegroundColor DarkGray }
}
function Write-Header ([string]$title) {
    Write-Host ""
    Write-Host "== $title ==" -ForegroundColor DarkGray
}

function Test-Python {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) {
        Write-Miss "python not in PATH" "winget install Python.Python.3.11  (or https://www.python.org/downloads/)"
        $script:MissingHard = $true
        return
    }
    $verRaw = & python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    if (-not $verRaw) {
        Write-Miss "python present but failed to report version" "Reinstall Python."
        $script:MissingHard = $true
        return
    }
    $parts = $verRaw.Trim().Split('.')
    $major = [int]$parts[0]; $minor = [int]$parts[1]
    if (($major -gt $ReqPyMajor) -or (($major -eq $ReqPyMajor) -and ($minor -ge $ReqPyMinor))) {
        Write-OK "python $verRaw"
    } else {
        Write-Miss "python $verRaw (need >= $ReqPyMajor.$ReqPyMinor)" `
                  "winget install Python.Python.3.11"
        $script:MissingHard = $true
    }
}

function Test-Nvidia {
    $cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $cmd) {
        Write-Miss "nvidia-smi not in PATH" `
                  "Install NVIDIA driver from https://www.nvidia.com/Download/index.aspx (CUDA Toolkit not required - the driver bundles libcuda)."
        return
    }
    $line = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1
    if ($LASTEXITCODE -eq 0 -and $line) {
        Write-OK "nvidia-smi: $line"
    } else {
        Write-Warn2 "nvidia-smi present but query failed" `
                   "GPU might not be visible. Run 'nvidia-smi' manually to diagnose."
    }
}

function Test-Ffmpeg {
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
        Write-OK "ffmpeg"
    } else {
        Write-Miss "ffmpeg" "winget install Gyan.FFmpeg"
    }
}

function Test-OllamaCli {
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        $ver = (& ollama --version 2>$null | Select-Object -First 1)
        Write-OK "ollama CLI ($ver)"
        $script:HasOllamaCli = $true
    } else {
        Write-Miss "ollama CLI" `
                  "winget install Ollama.Ollama  (or https://ollama.com/download/windows)"
    }
}

function Test-OllamaDaemon {
    if (-not $script:HasOllamaCli) { return }
    try {
        $null = Invoke-WebRequest -Uri "$OllamaUrl/api/tags" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
        Write-OK "ollama daemon reachable at $OllamaUrl"
        $script:HasOllamaDaemon = $true
    } catch {
        Write-Warn2 "ollama daemon not reachable at $OllamaUrl" `
                   "Open Ollama from the Start menu - it runs as a tray app."
    }
}

function Test-LlmModel {
    if (-not $script:HasOllamaDaemon) { return }
    $names = (& ollama list 2>$null) | Select-Object -Skip 1 | ForEach-Object { ($_ -split '\s+')[0] }
    if ($names -contains $LlmModel) {
        Write-OK "model $LlmModel present"
        $script:HasModel = $true
    } else {
        Write-Warn2 "model $LlmModel not pulled yet" "Will be offered as a guided step below."
    }
}

function Confirm-YesNo ([string]$question) {
    $reply = Read-Host "`n$question [y/N]"
    return ($reply -match '^(y|yes)$')
}

function Invoke-Venv {
    if (Test-Path -LiteralPath '.venv') {
        Write-OK ".venv already exists"
        return
    }
    if (Confirm-YesNo "Create .venv (python -m venv .venv)?") {
        & python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
        Write-OK ".venv created"
    }
}

function Invoke-PipInstall {
    $pip = Join-Path $RepoRoot '.venv\Scripts\pip.exe'
    if (-not (Test-Path -LiteralPath $pip)) {
        Write-Warn2 "no .venv\Scripts\pip.exe; skipping 'pip install -e .'" "Create the venv first."
        return
    }
    if (Confirm-YesNo "Install project into .venv (pip install -e .)?") {
        & $pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
        & $pip install -e .
        if ($LASTEXITCODE -ne 0) { throw "pip install -e . failed" }
        Write-OK "project installed into .venv"
    }
}

function Invoke-PullModel {
    if (-not $script:HasOllamaDaemon -or $script:HasModel) { return }
    if (Confirm-YesNo "Pull $LlmModel into Ollama now (may download several GB)?") {
        & ollama pull $LlmModel
        if ($LASTEXITCODE -ne 0) { throw "ollama pull failed" }
        Write-OK "$LlmModel pulled"
    }
}

Write-Header "Project ADA - Windows installer"
Write-Host "Repo: $RepoRoot"

Write-Header "System checks"
Test-Python
Test-Nvidia
Test-Ffmpeg
Test-OllamaCli
Test-OllamaDaemon
Test-LlmModel

Write-Header "Guided steps"
if ($script:MissingHard) {
    Write-Host "Python is missing or too old; skipping guided steps."
    Write-Host "Re-run this script after installing the items marked [MISS] above."
    exit 0
}

try {
    $ErrorActionPreference = 'Stop'
    Invoke-Venv
    Invoke-PipInstall
    Invoke-PullModel
} catch {
    Write-Host ""
    Write-Host "Guided step failed: $_" -ForegroundColor Red
    exit 1
} finally {
    $ErrorActionPreference = 'Continue'
}

Write-Header "Next steps"
@'
  1. .\.venv\Scripts\Activate.ps1
  2. ada --once        # single conversational turn
     ada               # interactive loop ('Press Enter to speak')

If the audio device is wrong, list devices with:
  python -c "import sounddevice; print(sounddevice.query_devices())"
and set audio.input_device / audio.output_device in config\default.yaml.
'@ | Write-Host
