# Release tracker

Source: *Global dubbing app · Engineering and release instructions* (issued 2026-11-20).
Base SHA `dcaa0a0`. Inventory: [INVENTORY.md](INVENTORY.md). Implementation order: [TODO.md](TODO.md).
Earlier dialogue-directive evidence stays in [../dialogue/requirements_matrix.md](../dialogue/requirements_matrix.md).

**Status values:** `planned` · `implementing` · `verified` · `blocked` · `disabled in production`.
"Implemented" alone never closes a row: `verified` needs the evidence column filled with a
reproducible command or artifact.

**W pkg column:** the approved work packages from commercial specification v5 (reproduced in the
2026-11-20 instructions §1): W1 accounts and storage · W2 creator submission · W3 media and worker ·
W4 catalogue and balances · W5 Razorpay · W6 quotes and checkout · W7 results and history ·
W8 Premium batches · W9 quality and paid pilot. `—` = cross-cutting engineering work outside the
packages (inventory, CI, switches, licences). The task/requirement reference column uses P01–P10 for
product requirements and the section row or drill for the rest (Q = §3, M = §4, L = §5, T = §6,
R = §7, G/E = §8).

**Owners:** Eng = implementation agent · Founder · Pay = payment owner · Counsel = qualified legal
reviewer · Acct = accountant · Rev = two fluent listeners · Ops = release/support owner.

## A · First working session (§1)

| Task / req ref | W pkg | Owner | Deps | Files / routes | Acceptance case | Evidence | Status | Blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A01 | — | Eng | – | all | Inventory of routes, pages, entries, workers, providers with purpose/boundary/hosted class | [INVENTORY.md](INVENTORY.md) | implementing | owner + test columns partly "–" |
| A02 | — | Eng | A01 | manifests, `web/`, `.github/workflows` | Clean install, `next build`, backend start, full `pytest --collect-only`; record versions/counts/skips; CI added | `next build` OK, full pytest 396 passed (20m51s) 2026-11-20, local Windows; CI in `.github/workflows/ci.yml` (not yet run on GitHub) | implementing | – |
| A03 | W9 | Eng | A02 | `backend/tests_dialogue/fixtures/` | Small licensed offline fixture set (synthetic speech + generated video) | – | planned | – |
| A04 | — | Eng | – | this file | One tracker row per acceptance item | this file | implementing | W1–W9 (OQ1) |
| A05 | W1–W7 | Eng | S01, M02 | all hosted paths | Vertical slice on staging: sign in → upload owned video → quote → buy test credits → one voice → verified MP4/SRT → private history → receipt | – | planned | OQ2 (auth), OQ3 (Razorpay test keys), OQ4 (staging host) |
| A06 | W1 | Eng | X01 | `backend/features.py`, launch config | Separate desktop/local and hosted profiles; explicit hosted route allowlist; both exercised; legacy local history kept, unknown-owner artifacts private + unassigned | profiles + route gate + CORS in `features.py`/`app.py`; hosted profile not yet run on a host | implementing | – |
| X01 | — | Eng | – | `backend/features.py`, `/api/features`, Settings UI | **Module on/off matrix** (owner request): every existing mode/engine/option is a switchable module; off = hidden in UI **and** rejected by API; nothing deleted; per-profile defaults | `test_features.py` 18 tests + browser run 16/16 (Edge, Playwright) 2026-11-20, local Windows | verified | – |

## S · Confirmed starting points (§1 table)

| Task / req ref | W pkg | Owner | Deps | Files / routes | Acceptance case | Evidence | Status | Blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S01 | W1 | Eng | OQ2 | `jobstore.py`, every `/api/jobs*`, links, presets, glossary | Owner column + enforcement at the service boundary for list/events/retry/edit/billing/files/delete; two accounts disjoint after cold restart | `accounts.py`, owner column + ownership middleware; `test_accounts.py` A/B disjoint after cold restart, 404 on every per-job route 2026-11-20 local; storage still the local disk (no private bucket) | implementing | – |
| S02 | W2 | Eng | X01 | `web/src/app/api/local-dub/route.ts`, `/api/cookies/status` | Removed from hosted surface; reference-only link never downloads, calls a provider or charges; large authorized transfers streamed with limits | – | planned | – |
| S03 | W3 | Eng | OQ4 | `app.py` semaphore, CORS, `HINGLISH_TRAINER_URL` | Durable execution; restricted origins; trainer forwarding impossible in hosted (incl. empty env vars) | – | planned | OQ4 |
| S04 | W8 | Eng | S01, S03 | `useBatchManager.ts`, new batch API | Batches are server records; restart browser + API + one worker mid-batch → each child final state and exact balance | – | planned | – |
| S05 | W3 | Eng | S01 | `dubbing/dialogue/*`, legacy pipelines | Trace which path executes per style/quality; billing, authz, rights, finalization behind shared code on every hosted path | – | planned | – |
| S06 | — | Eng | A02 | `web/package.json`, Python manifests, CosyVoice gitlink | Security-maintained versions, locked builds, gitlink/weights licence verified, installed-artifact smoke test; debug/notebook/tunnel/desktop surfaces unreachable in hosted | – | planned | – |

## P · Product acceptance contract (§2)

| Task / req ref | W pkg | Owner | Deps | Files / routes | Acceptance case | Evidence | Status | Blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P01 | W2 | Eng | S01, L01 | intake service, `page.tsx` | Link or upload × {no SRT, source SRT, approved target SRT}; probe before final quote; expired link, wrong track, duplicate click, interrupted upload, unsupported media; drafts survive payment/refresh/sign-in | – | planned | – |
| P02 | W2, W6 | Eng | P10, M02 | quote API, voice catalog, `page.tsx` | Locales, audible voice preview, full/first-N-minutes, style, quality, price; server-side validation; interval applied to video, captions, cost and export | – | planned | – |
| P03 | W3 | Eng | Q04 | `dubbing/dialogue` single-voice | Two-person argument stays a sequence of lines; approved target SRT spoken verbatim, pronunciation normalization recorded separately | partial: `srtdub`, `run_from_srt` | planned | – |
| P04 | W3 | Eng | Q03 | new narration style | Narrator recounts selected scenes in order at original timings; passages linked to cue IDs; editable preview; exact approved narration saved as SRT | – | planned | – |
| P05 | W3 | Eng | Q02 | `diarization.py`, `speaker_registry.py`, review UI | Stable voices across scenes; uncertain turns shown; preview/reassign/regenerate lines; A/B/A survives restart and edits; no gender/identity guess shown as fact | partial: dialogue tests | planned | – |
| P06 | W2, W6, W9 | Eng | M02, Q07 | job config, quote | Standard / Highest Quality per job; Premium as prepaid batch entitlement; config+voice+language+glossary+quote bound to job; provider loss → wait/cancel/accept revised quote | – | planned | – |
| P07 | W3, W7 | Eng | S03 | events, `jobs/[id]` | Six steps in order with status/time/%; persisted events for reconnect; skipped steps with reasons; "Complete" only after durable validated result | partial | planned | – |
| P08 | W7 | Eng | Q05 | results page | Download video, matching SRT, private destination, collapsed description + Copy, Recent Jobs; description from delivered interval; Saved Links count = delivered versions | partial | planned | – |
| P09 | W1, W7 | Eng | OQ2 | auth, history, balance, receipts | Verification/recovery, session revocation, private history, deletion, balance, receipts; two identities × two devices; consequences shown before destructive actions | email code + Google sign-in, sessions, sign out everywhere, balance page; browser run 9/9 2026-11-20 local; account deletion only closes access (L04 workflow open), no receipts yet (W5) | implementing | SMTP + Google client id for real use |
| P10 | W9 | Eng | Q06 | capability catalog | Directional catalog (ASR, MT, TTS, speakers, style, quality, region, price); billing country ≠ UI language ≠ source ≠ target; first pairs en→es-419, en→pt-BR, pt-BR→es-419, en→hi-en/bn-en | – | planned | OQ5 (pairs confirmed?) |

## Q · Media pipeline and language quality (§3)

| Task / req ref | W pkg | Owner | Deps | Files / routes | Acceptance case | Evidence | Status | Blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q00 | W2, W3 | Eng | – | job manifest | One versioned job manifest (source checksum, tracks, SRT role, interval, target, style, voices, quote, consent refs, model/config versions); cue edit invalidates only dependents; no obsolete cache reuse | – | planned | – |
| Q01 | W2 | Eng | P01 | probe/acquire | Bounded streamed upload; real container/codec/streams/rotation/duration; track selection; early no-speech detection; errors before charge or reversed | – | planned | – |
| Q02 | W3 | Eng | A03 | ASR + diarization | Benchmark local/Groq Whisper, alignment, pyannote per language; anonymous labels; valid supplied captions used; low-confidence → review | partial | planned | – |
| Q03 | W3 | Eng | Q00 | `translation.py` | Whole-script context: register, glossary, relationships, pronunciation, scene summaries; coverage record (no silent tail loss); untrusted media text isolated from instructions | partial | planned | – |
| Q04 | W3 | Eng | L06 | TTS, fit, mix | Commercially cleared stock voices per locale; timing repair, bounded retries, review before unnatural speed; bleed check; cloning only via L02 | partial | planned | L06 (Edge TTS commercial rights) |
| Q05 | W3 | Eng | – | verify/finalize | ffprobe + real decode; duration, language, intelligibility, clipping, missing/repeated lines, SRT encoding/bounds/alignment, **silent output detection**, checksum; temp → verify → immutable publish; restart reconciliation; settle charge only after | partial: `verify.py` | planned | – |
| Q06 | W9 | Eng + Rev | A03 | eval corpus | Versioned rights-cleared clips + rubrics; WER/CER, speaker error, glossary/number/negation, timing, naturalness, edit minutes; reviewer disagreement | – | planned | OQ6 (reviewers) |
| Q07 | W9 | Rev | Q06 | HQ gate | ≥20 clips × 2 blinded listeners per direction × style; HQ ≥70% wins, zero critical errors; Standard ≥18/20 publishable; one-hour export; requalify on model change | – | planned | OQ6 |

## M · Money, Razorpay, Premium batches (§4)

| Task / req ref | W pkg | Owner | Deps | Files / routes | Acceptance case | Evidence | Status | Blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M01 | W4 | Eng | – | catalog | ₹400 = 14,400 Std s; ₹500 = 14,400 Premium s; ₹100 = 3,600 Std s; no renewal; no expiry; Std used first on singles, Premium only on batches; integer seconds, minor-unit money | `billing/catalog.py`, `test_billing.py` 2026-11-20, local Windows | implementing | – |
| M02 | W6 | Eng | M01, S01 | quote API | 15 min Std 1 lang = ₹25; 30 min × 2 langs = 3,600 s; ₹400 − 2.5 h → 5,400 s a year later; +₹100 → 9,000 s; quote binds account/source/interval/langs/style/quality/price version/tax/currency/expiry/max cash; any change → new quote | `billing/quote.py`: ₹25/15 min, 3,600 s, 5,400→9,000 tests pass 2026-11-20, local Windows; no API until S01 | implementing | – |
| M03 | W5 | Eng + Pay | M01 | Razorpay orders/webhooks | Server-side orders; verify signature, payment state, capture, amount, currency, env, order, account; raw-body webhook check; failed/abandoned/delayed/duplicate/out-of-order events; duplicate payment ID under new event ID | – | planned | OQ3 |
| M04 | W4 | Eng | M01 | ledger | Immutable grant/reserve/settle/release/refund/adjust with unique business keys; single-transaction reserve; two tabs/two workers/final-minute race; property tests for conservation | `billing/ledger.py`: immutable triggers, final-minute race (8 threads), 25 random sequences 2026-11-20, local Windows; not wired to jobs | implementing | – |
| M05 | W5 | Eng + Pay | M04 | refunds | Failed/cancelled pre-delivery releases seconds; captured cash refunded to original method with status; settle once; chargeback mid-batch gives no spendable duplicate | – | planned | OQ3 |
| M06 | W8 | Eng | M04, S04 | batch | Itemized child quotes; batch survives logout; per-child idempotency/output/funding/retry/cancel; mixed success restores only failed portions; unfunded or Standard-only batch rejected server-side | – | planned | – |
| M07 | W5 | Eng + Acct | M03 | reconciliation job | Daily match orders, captures, fees, settlements, refunds, balances, delivered jobs; alerts on orphan/duplicate/negative/stale/mismatch; idempotent repair with audit | – | planned | – |
| M08 | W5 | Founder + Pay | M03 | international | Enabled merchant × country × currency × method × tax × refund matrix; unsupported combos declined before collection | – | disabled in production | Razorpay intl activation |
| M09 | W9 | Founder + Acct | OQ2 | cost model | Per direction/style: median and high-cost 1 h job, fully redeemed 4 h pack; stress FX/provider/dense speech/retries; fulfilment reserve | – | planned | real cost data |

## L · Legal, rights, commercial use (§5) — controls by Eng, decisions by owners

| Task / req ref | W pkg | Owner | Deps | Files / routes | Acceptance case | Evidence | Status | Blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | W2 | Eng + Counsel | P01 | rights declaration | Fresh unchecked declaration per job/batch item (actor, time, wording version, media); reference-only YouTube link asks for original media before any charge | – | planned | wording sign-off |
| L02 | W3 | Eng + Counsel | Q04 | voice/identity | Stock voices first; cloning only with speaker authority, revocation, minors, complaints; operator can suspend one voice | – | disabled in production | Counsel |
| L03 | W1 | Eng + Ops | S01 | complaints | Contact, case intake, triage, suspension, notice, appeal, repeat abuse, restoration; §512 assessment if relied on | – | planned | designated agent |
| L04 | W1 | Eng + Counsel | S01 | privacy | Data map intake→providers→caches→backups→logs; retention; deletion trace incl. backup restore; trainer forwarding off; DPDP/GDPR/LGPD notices | – | planned | Counsel |
| L05 | W3 | Eng + Counsel | Q05 | AI disclosure | Machine-readable marking survives final mux; user-visible disclosure; EU AI Act Art. 50 assessment | – | planned | Counsel |
| L06 | W9 | Founder + Counsel | – | terms, providers | Terms/refund/credit/retention/closure per market; **commercial rights for every provider** (Edge TTS consumer endpoint has none documented) | – | blocked | Counsel, provider contracts |
| L07 | — | Eng | A02 | SBOM / licences | Inventory APIs, voices, weights, datasets, libs, FFmpeg build, fonts, music; SBOM + notices from the release; unapproved provider not selectable by request | – | planned | – |
| L08 | — | Eng | X01 | UI + docs copy | Remove "break Content ID" / evasion claims; keep ordinary visual-editing controls | browser run: no Content-ID/fingerprint wording on home 2026-11-20, local Windows | verified | – |

## T · Review, security, test matrix (§6)

| Task / req ref | W pkg | Owner | Deps | Files / routes | Acceptance case | Evidence | Status | Blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T01 | — | Eng | A02 | all | Static/build: lint, types, migrations, dependency/secret/licence scans, SBOM, artifact scan | – | planned | – |
| T02 | W4 | Eng | M04 | money, SRT, permissions | Unit + property tests: quote units, rounding, interval clipping, ledger conservation, consent scope, SRT timing, cue coverage, speaker continuity | money/ledger property tests done; SRT/consent/speaker parts open | implementing | – |
| T03 | W5 | Eng | OQ4 | staging | Integration/contract on real DB/queue/storage/Razorpay test/provider sandbox; every result labelled mocked/sandbox/staging/production | – | planned | OQ4 |
| T04 | W1 | Eng | S01 | every route | Tenant ownership on API/SSE/retries/downloads/previews/signed URLs/caches/admin; CSRF, CORS, keys, TLS, SSRF denial, unprivileged media workers, log redaction (OWASP ASVS) | ownership on API, SSE events, downloads, `/static/jobs`; HttpOnly+Secure+Lax cookie; hashed sessions and codes 2026-11-20 local; CSRF beyond SameSite, signed URLs, admin MFA open | implementing | – |
| T05 | W3 | Eng | A03 | media | Codecs, mono/stereo, multi-track, VFR, rotation, Unicode/RTL, damaged, no audio, long silence, overlap, SRT BOM/CRLF/order/out-of-range, 1 h and max size | – | planned | – |
| T06 | W7 | Eng | A05 | UI | Full journey incl. six input × SRT combinations, styles × qualities per direction, pairwise for the rest | – | planned | – |
| T07 | W7 | Eng | T06 | browsers | Chrome/Edge Win, Safari macOS/iOS, Chrome Android, Firefox; keyboard, screen reader, 200% zoom, reduced motion; WCAG 2.2 AA | – | planned | devices |
| T08 | W3, W8 | Eng | S03, M04 | failure | Kill/restart between durable states, duplicate delivery, cancel races, deletion mid-job, DB/storage failure after provider response, 429/5xx, out-of-order payments → one business effect | – | planned | – |
| T09 | W1 | Eng | S01 | migration | Snapshot current install; forward migration; mixed versions; rollback; restore compares counts/owners/hashes/states/balances; unassigned records private | – | planned | – |
| T10 | W9 | Ops | R05 | support | Reconstruct a job from a redacted reference; help, broken download, refund, takedown, deletion, outage, global checkout disable | – | planned | – |
| T11 | W1, W3, W8 | Eng | all | regressions | Named permanent cases: A/B isolation; incomplete draft marked success; silent MP4; duplicate job/refund events; failed batch child at final Premium second; unpaid upgrade reaching worker; revoked voice after pause; cancel vs complete race; deleted content back after restore; fr/pt-BR/es job routed through en/hi defaults | – | planned | – |

## R · Reliability and operations (§7)

| Task / req ref | W pkg | Owner | Deps | Files / routes | Acceptance case | Evidence | Status | Blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R01 | W3 | Eng | S03 | DB, queue, store, workers | Durable DB/queue/private store; leases, heartbeats, retry ceilings, checkpoint versions, cancel, idempotent finalize; per-account/global limits shown before acceptance | – | planned | OQ4 |
| R02 | W9 | Eng | R01 | telemetry | Queue time, time per source minute, failures by stage/provider/language, review rate, margin; correlation IDs; redaction; alerts with recipient | – | planned | alert recipient |
| R03 | W9 | Eng | R01 | load | Peak + 2× burst incl. long + short jobs; p95 < 2 s status/history/quote | – | planned | targets (Founder) |
| R04 | W9 | Eng | R03 | soak | 24 h realistic mix with outage; no leak, duplicate, lost credit, tenant leak, false success, stuck paid job | – | planned | – |
| R05 | W9 | Eng + Ops | R01 | backup/deploy | Encrypted backup + isolated restore; RPO ≤15 min, RTO ≤4 h; replay payments before checkout reopens; rolling deploy + rollback keeps balances; separate dev/staging/prod keys/buckets/logs | – | planned | – |
| R06 | W5 | Eng | M03 | limits | Provider budget and retry ceiling exhausted in staging → admission pauses, funded jobs wait/refund, operator alerted; low disk, disabled locale, missing voice, revoked key | – | planned | – |
| R07 | — | Eng + Ops | X01 | `features.py`, admin | Immediate stop: pause one capability (provider, voice, locale, checkout, intake) without stopping unrelated work, on data exposure, unauthorized acquisition/voice use, double debit, lost credit, false success, corrupt export or payment integrity failure; restart only after fix + regression + reconciliation | – | planned | – |

## G · Release gates (§8)

| Task / req ref | W pkg | Owner | Deps | Acceptance case | Status | Blocker |
| --- | --- | --- | --- | --- | --- | --- |
| G01 | W9 | Eng + independent reviewer | A–T | Code and security ready | planned | – |
| G02 | W9 | Media lead + Rev | Q07 | Output ready | planned | OQ6 |
| G03 | W9 | Founder + Pay | M | Commercially ready (genuine small purchase + refund) | planned | merchant activation |
| G04 | W9 | Founder + Counsel | L | Legally ready for enabled markets | planned | Counsel |
| G05 | W9 | Ops | R, T10 | Operationally ready | planned | – |
| G06 | W9 | Founder | G01–G05 | Public rollout (canary) | planned | – |
| G07 | W9 | Eng + Ops | G01–G06 | Release packet: inventory + matrix, commit/build digest, CI logs, setup, route/migration review, redacted E2E evidence, listening results, approvals, margin sheet, a11y, load/restore, reconciliation, runbooks, enabled capabilities, known issues, dated go/no-go; skipped/simulated/blocked labelled | planned | – |
| E01 | W9 | Eng | A05, M06 | First end-to-end acceptance run: ₹400 → 10 min → **13,800 s** left; account B sees nothing; ₹500 Premium two children, one fails → **600 s settle, 13,800 left**; replayed notifications change nothing; rerun consumes 600 once; +1 year clock; 1 h real-provider fixture | planned | OQ3, OQ4 |

## Owner decisions (2026-11-20)

| ID | Decision |
| --- | --- |
| OQ1 | W1–W9 defined in the 2026-11-20 instructions §1; applied to every row |
| OQ2 | Google sign-in + email one-time code; no passwords |
| OQ3 | Razorpay **test mode** first (keys in `backend/.env`); live keys after legal/tax rows |
| OQ4 | Pilot on the owner's PC behind a Cloudflare Tunnel (HTTPS; hosted cookies are Secure-only), small invited group; Postgres when more than one worker |
| OQ5 | English → Hindi (incl. Hindi–English) first; other pairs one at a time after the quality gate |
| OQ6 | Two fluent Hindi listeners chosen by the owner (owner may be one) |
| OQ7 | Hosted Standard voice: Google Chirp 3 HD, Sarvam Bulbul as second/fallback; Edge stays local-only |

## Still needed from the owner

| Item | For |
| --- | --- |
| `GOOGLE_OAUTH_CLIENT_ID` (Google Cloud → OAuth consent + Web client, origin = the tunnel URL) | Google button (P09) |
| SMTP settings (`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, `SMTP_FROM`) | real email codes in hosted |
| Razorpay test key id/secret + webhook secret | W5 |
| `GOOGLE_TTS_API_KEY` with Text-to-Speech enabled | Chirp 3 HD voices |
