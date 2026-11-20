# Packs the working tree (with uncommitted changes) for deploy/bootstrap.sh.
# Output: %TEMP%\krm-logit-mask.tar.gz (a few MB: code, data, eval sets, results).
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot)
$out = Join-Path $env:TEMP 'krm-logit-mask.tar.gz'
# Windows' own bsdtar: from Git Bash, a bare `tar` is MSYS tar, which reads "C:" as a remote host.
# Pack declared project inputs rather than the repository root: a stale
# sandbox-owned .pytest_cache can be unreadable on Windows and must never make
# a deployment archive silently incomplete.
& "$env:SystemRoot\System32\tar.exe" -czf $out pyproject.toml uv.lock README.md src data experiments deploy web
Get-Item $out | Select-Object FullName, Length
