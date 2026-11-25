# Registers two scheduled tasks for the current user (run once, in an elevated PowerShell):
#   VoiceDub         - starts deploy\start-hosted.ps1 at log-on and keeps it running
#   VoiceDub Backup  - deploy\backup-nightly.ps1 every day at 03:30
# Remove with: Unregister-ScheduledTask -TaskName 'VoiceDub','VoiceDub Backup' -Confirm:$false
$here = $PSScriptRoot
$ps = (Get-Command powershell).Source
$svc = New-ScheduledTaskAction -Execute $ps -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$here\start-hosted.ps1`""
$set = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable
Register-ScheduledTask -TaskName 'VoiceDub' -Action $svc -Trigger (New-ScheduledTaskTrigger -AtLogOn) -Settings $set -Force | Out-Null
$bk = New-ScheduledTaskAction -Execute $ps -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$here\backup-nightly.ps1`""
Register-ScheduledTask -TaskName 'VoiceDub Backup' -Action $bk -Trigger (New-ScheduledTaskTrigger -Daily -At 3:30am) -Settings (New-ScheduledTaskSettingsSet -StartWhenAvailable) -Force | Out-Null
Write-Output 'Registered: VoiceDub, VoiceDub Backup'
