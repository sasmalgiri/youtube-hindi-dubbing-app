# Translation benchmark — corpus translation_corpus_v1 (36 lines)

Run 2026-11-25 00:09:59. Automatic checks only; naturalness needs the Hindi reviewers.

| Engine | Checks passed | chrF (drift) | Seconds | Error |
|---|---|---|---|---|
| groq | 96.3% | 61.58 | 58.1 |  |

## groq

| Check | Passed |
|---|---|
| fits_slot | 36/36 |
| glossary | 2/5 |
| hindi_script | 34/36 |
| negation | 9/9 |
| not_obeyed | 2/2 |
| numbers | 6/6 |
| present | 36/36 |
| question | 5/5 |

Failures:
- `gls-01`: names/terms not rendered: ['Amazon']
- `gls-04`: only 80% Devanagari; names/terms not rendered: ['Roman']
- `gls-05`: only 77% Devanagari; names/terms not rendered: ['Tokyo']

## Change vs baseline

| Engine | Δ checks | Δ chrF |
|---|---|---|
| groq | +0.7% | +19.34 |
