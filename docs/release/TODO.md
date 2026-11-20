# Implementation TODO

Ordered by the build order in §8 (A → B → C ∥ D → E → F). Each task lists the tracker rows it
closes in **Covers**; `python docs/release/check_alignment.py` fails if a tracker row is covered by
no task or a task names a row that does not exist.

`Needs` = owner input that must arrive before the task can be finished (not started).

## Stage A — inventory, baseline, switches (no owner input needed)

| # | Task | Covers | Needs |
| --- | --- | --- | --- |
| 1 | **Module on/off matrix**: `backend/features.py` registry of every existing mode, engine and option (input modes, pipelines, TTS/ASR/MT engines, story modes, visual transforms, voice clone, trainer forwarding, cookies, local-dub, outputs listing, cache admin); per-profile defaults (`local` / `hosted`); persisted overrides; `GET/PATCH /api/features`; API rejects a switched-off module; UI hides it; Settings → Modules page. Nothing deleted. | X01, A06, L08, R07 | – |
| 2 | Hosted route allowlist driven by the matrix: `deny` routes return 404 in the hosted profile; local-dub, cookies, outputs, cache admin, story GPU, notebook/tunnel off; CORS from config; trainer forwarding impossible when hosted (empty env included) | A06, S02, S03 | – |
| 3 | Baseline: clean install from manifests, `next build`, backend start, `pytest --collect-only`; record versions/counts/skips; classify paid-provider scripts; add GitHub Actions CI (offline tests + web build) | A02, T01 | – |
| 4 | Dependency hygiene: supported Next.js, locked Python deps, CosyVoice gitlink provenance, exclude conflict copies from packaging | S06, L07 | – |
| 5 | Offline licensed fixture set (synthetic voices, generated video, SRT variants) | A03 | – |
| 6 | Finish inventory owner/test columns; keep tracker current | A01, A04 | OQ1 |

## Stage B — private accounts, storage, rights intake

| # | Task | Covers | Needs |
| --- | --- | --- | --- |
| 7 | Accounts + sessions (chosen method), owner column on jobs/links/presets/glossary, enforcement at the service boundary, legacy local history kept, unknown owners private + unassigned with a recovery path; migration with snapshot/rollback | S01, P09, T04, T09 | OQ2 |
| 8 | One intake service for link/upload × SRT role; bounded streamed upload; real probe; track selection; no-speech detection; reference-only link never downloads/charges; drafts survive payment/refresh/sign-in | P01, Q01, S02 | – |
| 9 | Rights declaration per job/batch item (versioned wording, actor, time, media) | L01 | Counsel wording |
| 10 | Versioned job manifest + durable DB/queue/worker processes with leases, heartbeats, cancel, idempotent finalize, persisted events | Q00, S03, R01, P07 | OQ4 |

## Stage C — single-job pipeline and private downloads

| # | Task | Covers | Needs |
| --- | --- | --- | --- |
| 11 | Trace executed path per style/quality; put authz, rights, billing hooks and finalization in shared code on every hosted path | S05 | – |
| 12 | Finalize: decode + silent-output detection + SRT bounds/alignment + checksum; temp → verify → immutable publish; restart reconciliation; results page (download, SRT, private destination, description with Copy, real Saved Links count) | Q05, P08 | – |
| 13 | Single-voice dialogue: verbatim approved target SRT, pronunciation normalization logged separately | P03 | – |
| 14 | Voice catalog with audible previews, capability catalog per direction, interval selection applied to video/captions/cost/export | P02, P10, Q04 | OQ5, provider rights (L06) |

## Stage D — catalog, ledger, Razorpay sandbox (parallel with C)

| # | Task | Covers | Needs |
| --- | --- | --- | --- |
| 15 | Price catalog (integer seconds, minor units) and quote engine bound to account/source/interval/langs/style/quality/price version/tax/expiry | M01, M02, P06 | – |
| 16 | Immutable ledger (grant/reserve/settle/release/refund/adjust, unique business keys, single-transaction reserve) + property tests | M04, T02 | – |
| 17 | Razorpay test mode: server-side orders, signature + payment-state verification, raw-body webhooks, duplicates/out-of-order, refunds to original method, chargebacks | M03, M05 | OQ3 |

## Stage E — connect payment and delivery; recovery, batches, review, history

| # | Task | Covers | Needs |
| --- | --- | --- | --- |
| 18 | Server-side batches with Premium funding, per-child idempotency/retry/cancel, mixed-success restoration | S04, M06 | – |
| 19 | Multi-speaker review (uncertain turns, reassign, regenerate lines) and third-person narration style with cue links and editable preview | P05, P04, Q02, Q03 | – |
| 20 | Failure and concurrency drills + named permanent regression cases | T08, T11 | – |
| 21 | Daily reconciliation, telemetry with redaction and alerts, provider budgets and admission limits | M07, R02, R06 | alert recipient |

## Stage F — qualification, compliance, cost, production

| # | Task | Covers | Needs |
| --- | --- | --- | --- |
| 22 | Evaluation corpus, metrics harness, blinded listening workflow, HQ/Standard gates | Q06, Q07, G02 | OQ6 |
| 23 | Legal controls: voice authority/revocation, complaints workflow, privacy map + deletion trace, AI disclosure marking, terms per market, SBOM + notices | L02, L03, L04, L05, L06 | Counsel, Acct |
| 24 | Media compatibility suite, UI end-to-end journeys, browser and accessibility passes | T05, T06, T07 | devices |
| 25 | Staging integration (labelled results), load, 24 h soak, encrypted backup/restore, deploy/rollback, vertical slice on staging | T03, R03, R04, R05, A05 | OQ4 |
| 26 | Cost/margin model, international matrix, support drills, release gates, first end-to-end acceptance run | M08, M09, T10, G01, G03, G04, G05, G06, G07, E01 | Founder, Pay, Counsel |
