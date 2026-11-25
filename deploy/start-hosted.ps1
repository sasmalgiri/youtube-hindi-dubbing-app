# VoiceDub hosted service: backend (127.0.0.1:8000) + web (127.0.0.1:3000), restarted if either dies.
# Public traffic arrives only through Cloudflare Tunnel (deploy/cloudflared.yml) to the web port.
# Run by the "VoiceDub" scheduled task (deploy/install-tasks.ps1) or by hand.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $root 'backend\logs'
New-Item -ItemType Directory -Force $logs | Out-Null

$env:VOICEDUB_PROFILE   = 'hosted'
$env:VOICEDUB_HOST      = '127.0.0.1'     # never reachable except through the web proxy
$env:VOICEDUB_PORT      = '8000'
$env:VOICEDUB_TRUST_CF  = '1'             # client address from CF-Connecting-IP (rate limits)
$env:BACKEND_PORT       = '8000'
$env:PYTHONIOENCODING   = 'utf-8'
if (-not $env:VOICEDUB_STATE_DIR) { $env:VOICEDUB_STATE_DIR = Join-Path $root 'backend' }
# Keys and secrets live in backend\.env (loaded by the backend), never in this file.

$python = (Get-Command python).Source
function Start-Backend {
    Start-Process -FilePath $python -ArgumentList 'app.py' -WorkingDirectory (Join-Path $root 'backend') `
        -RedirectStandardOutput (Join-Path $logs 'backend.out.log') -RedirectStandardError (Join-Path $logs 'backend.err.log') `
        -WindowStyle Hidden -PassThru
}
function Start-Web {
    Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', 'npx next start -H 127.0.0.1 -p 3000' -WorkingDirectory (Join-Path $root 'web') `
        -RedirectStandardOutput (Join-Path $logs 'web.out.log') -RedirectStandardError (Join-Path $logs 'web.err.log') `
        -WindowStyle Hidden -PassThru
}

if (-not (Test-Path (Join-Path $root 'web\.next\BUILD_ID'))) {
    Push-Location (Join-Path $root 'web'); npx next build; Pop-Location
}
$b = Start-Backend; $w = Start-Web
"[$(Get-Date -Format s)] started backend pid $($b.Id), web pid $($w.Id)" | Add-Content (Join-Path $logs 'supervisor.log')
while ($true) {
    Start-Sleep -Seconds 10
    if ($b.HasExited) { "[$(Get-Date -Format s)] backend exited ($($b.ExitCode)); restarting" | Add-Content (Join-Path $logs 'supervisor.log'); $b = Start-Backend }
    if ($w.HasExited) { "[$(Get-Date -Format s)] web exited ($($w.ExitCode)); restarting" | Add-Content (Join-Path $logs 'supervisor.log'); $w = Start-Web }
}
