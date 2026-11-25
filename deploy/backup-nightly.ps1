# Nightly backup of the state databases (+ receipts, rights log). Keeps the newest 14.
# Copy VOICEDUB_BACKUP_DIR off this machine (e.g. a synced folder or R2) - a backup on the same disk is not a backup.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$state = if ($env:VOICEDUB_STATE_DIR) { $env:VOICEDUB_STATE_DIR } else { Join-Path $root 'backend' }
$dest  = if ($env:VOICEDUB_BACKUP_DIR) { $env:VOICEDUB_BACKUP_DIR } else { Join-Path $state 'backups' }
Push-Location (Join-Path $root 'backend')
try {
    python -m ops.backup create --state $state --dest $dest
    if ($LASTEXITCODE -ne 0) { throw "backup failed ($LASTEXITCODE)" }
    Get-ChildItem $dest -Directory -Filter 'backup_*' | Sort-Object Name -Descending | Select-Object -Skip 14 |
        ForEach-Object { Remove-Item $_.FullName -Recurse -Force -Confirm:$false }
} finally { Pop-Location }
