# English video to Hindi with consistent character voices

Use **Hindi Dialogue** for films and conversations. Emotion detection and voice
cloning are optional; neither is needed for this workflow.

## Steps in the app

1. Select **Hindi Dialogue** and add the English YouTube URL or local video.
2. Optionally upload an English SRT. For an ordinary transcript, leave
   **Dialogue speaker labels → Detect speakers from English audio** selected.
   Audio analysis still determines who speaks, even with a supplied transcript.
3. Enable speaker detection and choose a working translation engine and voice
   provider in the module settings. Resolve any dependency warning first.
4. Enable **Pause to review the Hindi lines and voices before voicing**.
5. Start the job. The current review pause occurs **after translation and before
   TTS**; this change does not introduce a separate pre-translation cast review.
6. On the job page, listen to each character's original reference clip. Give it
   an optional name, such as “Female lead” or “Brother”. Names persist with the
   job but never change speaker IDs or become spoken text.
7. Merge duplicate speakers, correct individual line assignments, and select a
   distinct Hindi voice per character. Use **Preview** to audition voices.
8. Review Hindi, particularly names, short replies, and flagged mixed-speaker
   cues. Each manually edited Hindi line is protected from automatic shortening
   during duration fitting. Long lines may need manual shortening to fit.
9. Choose **Continue with these changes**. The existing pipeline generates,
   fits, verifies, and mixes the speech over the separated background.
10. Check the report and listen to the result before using it. Reassign or edit
    incorrect lines and use **Re-voice changed lines**. Unchanged synthesis can
    reuse the same job's cache. Export the finished MP4 and Hindi subtitles.

For a first trial, use a short local excerpt or the CLI's `--limit-seconds 180`
option. This release does not add an integrated scene-preview render button.

## Already have a reviewed speaker-labelled SRT?

Choose **Dialogue speaker labels → Use supplied SRT labels (every cue must be
labelled)**. Each English or Hindi cue must start with a numbered label:

```srt
1
00:00:01,200 --> 00:00:03,500
[SPEAKER_00] तुम कहाँ थे?

2
00:00:03,800 --> 00:00:05,500
[SPEAKER_01] मैं बाहर था।
```

This explicitly makes supplied IDs authoritative. Missing labels stop the job
with the cue numbers; the app does not silently mix two incompatible speaker
numbering schemes. Automatic audio mode retains the existing behavior.

The parser removes labels before synthesis. Do not add `[angry]`, `[female]`,
or other direction tags: those are not voice-control instructions.

Audio analysis may still flag a Hindi cue containing multiple voices. Such a
cue is not used as a clean reference recording. Correct its boundaries in the
source SRT and start a new job if it actually contains two characters.

Switching speaker-label policy on resume is rejected: a fresh job is required
because existing corrections refer to the old speaker identities. Character
names, line edits, and voice settings can be changed through the normal review
and re-voice path.

## What this update changes

- Explicit authority for supplied English and Hindi SRT speaker labels.
- Reference clips based on supplied dialogue spans rather than accidentally
  matching an unrelated audio speaker with the same numeric ID.
- Persistent character names in review, saved speaker records, and checkpoints.
- Manual Hindi edits protected from automatic duration rewrites.
- A failed review callback stops before speech generation and retains the
  checkpoint instead of silently generating unreviewed speech.
- Mixed-speaker subtitle cues highlighted as serious review flags.

Automatic diarization and Hindi pronunciation still need listening checks.
These improvements do not certify a two-hour film's output or add automatic
character recognition from faces.
