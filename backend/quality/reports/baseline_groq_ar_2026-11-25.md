# Translation benchmark — en→ar, corpus translation_corpus_v1 (36 lines)

Run 2026-11-25 00:16:33. Automatic checks only; naturalness needs the Hindi reviewers.

| Engine | Checks passed | chrF (drift) | Seconds | Error |
|---|---|---|---|---|
| groq | 99.2% | None | 6.1 |  |

## groq

| Check | Passed |
|---|---|
| fits_slot | 36/36 |
| negation | 8/9 |
| not_obeyed | 2/2 |
| numbers | 6/6 |
| present | 36/36 |
| question | 5/5 |
| target_script | 36/36 |

Failures:
- `neg-03`: negation lost
