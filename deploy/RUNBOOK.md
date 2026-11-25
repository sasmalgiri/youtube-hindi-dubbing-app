# VoiceDub hosted service — runbook

One Windows machine runs everything: the backend on `127.0.0.1:8000`, the web app on `127.0.0.1:3000` (it proxies `/api` to the backend), and Cloudflare Tunnel, which carries all public traffic to the web app. Nothing listens on a public port.

## First-time setup

1. **Keys.** Put every key in `backend\.env` (never in chat or git). Then run
   `cd backend; python setup_check.py --live` until it shows no red lines.
   Required in hosted: Razorpay (live), Google TTS (Chirp), SMTP, Google OAuth client id, `VOICEDUB_ADMIN_TOKEN` (any long random string), `VOICEDUB_CORS_ORIGINS=https://app.example.com`.
   Optional: `VOICEDUB_ALERT_EMAIL` (where alerts go), `VOICEDUB_CURRENCIES=INR,USD,EUR,GBP` (once prices are approved), `VOICEDUB_BACKUP_DIR` (a folder that syncs off this machine), `NEXT_PUBLIC_BRAND`.
2. **Build the web app.** `cd web; npx next build`
3. **Tunnel.** Follow the comments in `deploy\cloudflared.yml` (login, create, route DNS, install as a service).
4. **Start on log-on and back up nightly.** In an elevated PowerShell: `deploy\install-tasks.ps1`.
5. **Check.** Open `https://app.example.com/pricing`, sign in, then `https://app.example.com/admin` with the admin token. You should see no active alerts once the first backup has run (run `deploy\backup-nightly.ps1` once by hand).

## Every day (2 minutes)

Open `/admin`. It shows active alerts, the queue, dubs in the last 24 h, failures with their errors, orders and money by currency, disk, last backup, and billing reconciliation (should say **0 problems**). The same checks run every 5 minutes and email `VOICEDUB_ALERT_EMAIL` (at most every 6 h per alert); every alert is also kept in `backend\alerts.jsonl`.

## When an alert fires

| Alert | What to do |
|---|---|
| Disk nearly full | Delete old `backend\outputs\<job>` folders past the retention period; move backups off the disk. |
| Queue stuck | Look at `backend\logs\backend.err.log`. Restart: stop the `VoiceDub` task and start it again. Stuck jobs are requeued once, then released (time refunded). |
| Running job past its lease | The worker died; a restart recovers it as above. |
| N dubs failed in 24 h | Open the failures on `/admin`. A provider outage (Groq, Google TTS) shows the same error on every job — check the provider status page. Customers are not charged for failed dubs. |
| No backup / backup old | Run `deploy\backup-nightly.ps1` by hand; check the `VoiceDub Backup` task's last result. |
| Payment dispute | Answer it in the Razorpay dashboard. The time from that payment has already been removed. |
| Reconciliation problems | Do not edit the databases. Run `python -m billing.reconcile` for detail; the ledger is append-only, fix with a correcting entry. |

## Restore from a backup

1. Stop the `VoiceDub` task.
2. `cd backend; python -m ops.backup restore <BACKUP_DIR>\backup_YYYYMMDD_HHMMSS --target .`
3. `python -m ops.backup verify <same backup> --target .` — must report no differences.
4. Start the task. Jobs that were running are requeued.

## Updating the app

`git pull`, `cd web; npx next build`, then restart the `VoiceDub` task. Never upgrade torch (pinned 2.4.1+cu121; install with `-c backend\constraints.txt`).

## Limits worth knowing

- Request limits per visitor (per minute): 5 sign-in codes, 20 uploads, 30 quotes, 20 checkouts, 600 API calls overall (`backend\ratelimit.py`).
- Uploads up to 2048 MB (`VOICEDUB_MAX_UPLOAD_MB`), but Cloudflare's free plan stops request bodies at 100 MB.
- Each dub can be re-voiced 5 times for free (`VOICEDUB_MAX_REVOICES`); only changed lines are voiced again.
