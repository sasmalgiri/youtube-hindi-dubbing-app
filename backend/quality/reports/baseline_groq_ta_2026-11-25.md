# Translation benchmark — en→ta, corpus translation_corpus_v1 (36 lines)

Run 2026-11-25 00:16:46. Automatic checks only; naturalness needs the Hindi reviewers.

| Engine | Checks passed | chrF (drift) | Seconds | Error |
|---|---|---|---|---|
| groq | 97.7% | None | 8.3 |  |

## groq

| Check | Passed |
|---|---|
| fits_slot | 36/36 |
| negation | 6/9 |
| not_obeyed | 2/2 |
| numbers | 6/6 |
| present | 36/36 |
| question | 5/5 |
| target_script | 36/36 |

Failures:
- `neg-04`: negation lost
- `long-01`: negation lost
- `inj-02`: negation lost
