@echo off
REM Wrapper so users can run the installer without changing PowerShell's
REM ExecutionPolicy. -ExecutionPolicy Bypass applies to this invocation only;
REM system policy is not modified.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1" %*
