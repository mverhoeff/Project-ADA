#!/usr/bin/env bash
# Verify-and-guide installer for Project ADA on Fedora 43+.
#
# This script does NOT install OS-level dependencies (CUDA driver, Ollama,
# ffmpeg). It checks for them, prints what is missing with a direct
# install hint, then offers to create a Python venv, install the project,
# and pull the LLM model.
#
# Exit code: 0 on a clean run, even if external deps are missing.
# Non-zero only if a guided step (venv / pip / ollama pull) fails.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
REQUIRED_PY_MAJOR=3
REQUIRED_PY_MINOR=11
LLM_MODEL="qwen3:8b"

# Terminal colors (only if stdout is a tty).
if [ -t 1 ]; then
    C_OK="\033[32m"; C_MISS="\033[31m"; C_WARN="\033[33m"; C_DIM="\033[2m"; C_RST="\033[0m"
else
    C_OK=""; C_MISS=""; C_WARN=""; C_DIM=""; C_RST=""
fi

ok()   { printf "  ${C_OK}[OK]${C_RST}   %s\n" "$1"; }
miss() { printf "  ${C_MISS}[MISS]${C_RST} %s\n    ${C_DIM}%s${C_RST}\n" "$1" "$2"; }
warn() { printf "  ${C_WARN}[WARN]${C_RST} %s\n    ${C_DIM}%s${C_RST}\n" "$1" "$2"; }
hdr()  { printf "\n${C_DIM}== %s ==${C_RST}\n" "$1"; }

# Run a command with an animated spinner and elapsed-time counter. Output is
# buffered to a tempfile and only shown if the command fails. On non-tty
# stdout (piped/CI), the spinner loop is skipped and we just print the final
# [OK]/[FAIL] line.
#
# Usage: spinner_run "Label" cmd arg1 arg2 ...
spinner_run() {
    local label="$1"; shift
    local logfile rc start elapsed
    logfile=$(mktemp)
    start=$SECONDS
    if [ -t 1 ]; then
        "$@" > "$logfile" 2>&1 &
        local pid=$! frames='|/-\' i=0
        while kill -0 "$pid" 2>/dev/null; do
            printf "\r  ${C_DIM}%s${C_RST} %s (%ds)" \
                   "${frames:$((i%4)):1}" "$label" "$((SECONDS-start))"
            i=$((i+1))
            sleep 0.1
        done
        if wait "$pid"; then rc=0; else rc=$?; fi
    else
        printf "  ... %s\n" "$label"
        if "$@" > "$logfile" 2>&1; then rc=0; else rc=$?; fi
    fi
    elapsed=$((SECONDS-start))
    if [ "$rc" -eq 0 ]; then
        printf "\r  ${C_OK}[OK]${C_RST}   %s (%ds)%s\n" "$label" "$elapsed" "                              "
    else
        printf "\r  ${C_MISS}[FAIL]${C_RST} %s (%ds)%s\n" "$label" "$elapsed" "                              "
        printf "  ${C_DIM}--- output ---${C_RST}\n"
        sed 's/^/    /' "$logfile"
    fi
    rm -f "$logfile"
    return "$rc"
}

# Track missing critical deps so the summary can show whether the user
# can already proceed with the guided steps.
MISSING_HARD=0

check_python() {
    if ! command -v python3 >/dev/null 2>&1; then
        miss "python3" "sudo dnf install python3.11"
        MISSING_HARD=1
        return
    fi
    local ver major minor
    ver="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    major="${ver%%.*}"
    minor="${ver##*.}"
    if [ "$major" -gt "$REQUIRED_PY_MAJOR" ] || \
       { [ "$major" -eq "$REQUIRED_PY_MAJOR" ] && [ "$minor" -ge "$REQUIRED_PY_MINOR" ]; }; then
        ok "python3 ${ver}"
    else
        miss "python3 ${ver} (need >= ${REQUIRED_PY_MAJOR}.${REQUIRED_PY_MINOR})" \
             "sudo dnf install python3.11"
        MISSING_HARD=1
    fi
}

check_nvidia() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        miss "nvidia-smi not in PATH" \
             "Install NVIDIA driver (Fedora: enable RPM Fusion, then 'sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda'). Driver downloads: https://www.nvidia.com/Download/index.aspx"
        return
    fi
    local line
    if line="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n1)"; then
        ok "nvidia-smi: ${line}"
    else
        warn "nvidia-smi present but query failed" \
             "GPU might not be visible. Run 'nvidia-smi' manually to diagnose."
    fi
}

check_ffmpeg() {
    if command -v ffmpeg >/dev/null 2>&1; then
        ok "ffmpeg"
    else
        miss "ffmpeg" "sudo dnf install ffmpeg  (RPM Fusion required)"
    fi
}

check_ollama_cli() {
    if command -v ollama >/dev/null 2>&1; then
        ok "ollama CLI ($(ollama --version 2>/dev/null | head -n1))"
        HAS_OLLAMA_CLI=1
    else
        miss "ollama CLI" "curl -fsSL https://ollama.com/install.sh | sh"
        HAS_OLLAMA_CLI=0
    fi
}

check_ollama_daemon() {
    if [ "${HAS_OLLAMA_CLI:-0}" -eq 0 ]; then
        return
    fi
    if curl -fsS --max-time 3 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
        ok "ollama daemon reachable at ${OLLAMA_URL}"
        HAS_OLLAMA_DAEMON=1
    else
        warn "ollama daemon not reachable at ${OLLAMA_URL}" \
             "Start it with 'systemctl --user start ollama' or 'ollama serve'"
        HAS_OLLAMA_DAEMON=0
    fi
}

check_llm_model() {
    if [ "${HAS_OLLAMA_DAEMON:-0}" -eq 0 ]; then
        return
    fi
    if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "${LLM_MODEL}"; then
        ok "model ${LLM_MODEL} present"
        HAS_MODEL=1
    else
        warn "model ${LLM_MODEL} not pulled yet" "Will be offered as a guided step below."
        HAS_MODEL=0
    fi
}

prompt_yes() {
    local question="$1"
    local reply
    printf "\n%s [y/N] " "$question"
    read -r reply || return 1
    case "$reply" in
        y|Y|yes|YES) return 0 ;;
        *) return 1 ;;
    esac
}

guided_venv() {
    if [ -d ".venv" ]; then
        ok ".venv already exists"
        return
    fi
    if prompt_yes "Create .venv (python3 -m venv .venv)?"; then
        spinner_run "Creating .venv" python3 -m venv .venv
    fi
}

guided_pip_install() {
    if [ ! -x ".venv/bin/pip" ]; then
        warn "no .venv/bin/pip; skipping 'pip install -e .'" \
             "Create the venv first."
        return
    fi
    if prompt_yes "Install project into .venv (pip install -e .)?"; then
        spinner_run "Upgrading pip" .venv/bin/pip install --upgrade pip
        spinner_run "Installing project (pip install -e .)" .venv/bin/pip install -e .
    fi
}

guided_pull_model() {
    if [ "${HAS_OLLAMA_DAEMON:-0}" -eq 0 ] || [ "${HAS_MODEL:-0}" -eq 1 ]; then
        return
    fi
    if prompt_yes "Pull ${LLM_MODEL} into Ollama now (may download several GB)?"; then
        printf "  ${C_DIM}(Ollama will show its own progress bar; this can take 5-10 minutes.)${C_RST}\n"
        ollama pull "${LLM_MODEL}"
        ok "${LLM_MODEL} pulled"
    fi
}

hdr "Project ADA — Linux installer (Fedora 43+)"
printf "Repo: %s\n" "$REPO_ROOT"

hdr "System checks"
check_python
check_nvidia
check_ffmpeg
check_ollama_cli
check_ollama_daemon
check_llm_model

hdr "Guided steps"
if [ "$MISSING_HARD" -eq 1 ]; then
    printf "Python is missing or too old; skipping guided steps.\n"
    printf "Re-run this script after installing the items marked [MISS] above.\n"
    exit 0
fi

guided_venv
guided_pip_install
guided_pull_model

hdr "Next steps"
cat <<EOF
  1. source .venv/bin/activate
  2. ada --once        # single conversational turn
     ada               # interactive loop ('Press Enter to speak')

If the audio device is wrong, list devices with:
  python -c "import sounddevice; print(sounddevice.query_devices())"
and set audio.input_device / audio.output_device in config/default.yaml.
EOF
