"""
Dubbing Pipeline with Translation Support
==========================================
Refactored from pipeline_v2 with:
- Callback-based progress reporting (for SSE)
- Translation step (deep-translator)
- Hindi TTS by default
"""
from __future__ import annotations

import asyncio
import math
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from srt_utils import write_srt


# ── Types ────────────────────────────────────────────────────────────────────
ProgressCallback = Callable[[str, float, str], None]

STEPS = ["download", "extract", "transcribe", "translate", "synthesize", "assemble"]
STEP_WEIGHTS = {
    "download": 0.15,
    "extract": 0.05,
    "transcribe": 0.25,
    "translate": 0.15,
    "synthesize": 0.30,
    "assemble": 0.10,
}


LANGUAGE_NAMES = {
    "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "ur": "Urdu", "en": "English", "es": "Spanish",
    "fr": "French", "de": "German", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic",
    "it": "Italian", "tr": "Turkish",
}

# Average spoken words-per-minute by language for TTS duration estimation.
# Used to compute target word counts so translated segments fit original timing.
LANGUAGE_WPM = {
    "en": 150, "hi": 120, "bn": 120, "ta": 110, "te": 115,
    "mr": 120, "gu": 120, "kn": 110, "ml": 105, "pa": 120,
    "ur": 120, "es": 160, "fr": 155, "de": 130, "ja": 200,
    "ko": 140, "zh": 160, "pt": 155, "ru": 130, "ar": 125,
    "it": 155, "tr": 130,
}

DEFAULT_VOICES = {
    "hi": "hi-IN-SwaraNeural",
    "en": "en-US-JennyNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ar": "ar-SA-ZariyahNeural",
    "it": "it-IT-ElsaNeural",
    "tr": "tr-TR-EmelNeural",
    "bn": "bn-IN-TanishaaNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "pa": "pa-IN-GurpreetNeural",
    "ur": "ur-PK-UzmaNeural",
}

MALE_VOICES = {
    "hi": "hi-IN-MadhurNeural",
    "en": "en-US-GuyNeural",
    "es": "es-ES-AlvaroNeural",
    "fr": "fr-FR-HenriNeural",
    "de": "de-DE-ConradNeural",
    "ja": "ja-JP-KeitaNeural",
    "ko": "ko-KR-InJoonNeural",
    "zh": "zh-CN-YunxiNeural",
    "pt": "pt-BR-AntonioNeural",
    "ru": "ru-RU-DmitryNeural",
    "ar": "ar-SA-HamedNeural",
    "it": "it-IT-DiegoNeural",
    "tr": "tr-TR-AhmetNeural",
    "bn": "bn-IN-BashkarNeural",
    "ta": "ta-IN-ValluvarNeural",
    "te": "te-IN-MohanNeural",
    "mr": "mr-IN-ManoharNeural",
    "gu": "gu-IN-NiranjanNeural",
    "kn": "kn-IN-GaganNeural",
    "ml": "ml-IN-MidhunNeural",
    "pa": "pa-IN-GurpreetNeural",
    "ur": "ur-PK-AsadNeural",
}

# Pool of distinct voices per gender per language for multi-speaker
VOICE_POOL = {
    "en": {
        "female": ["en-US-JennyNeural", "en-US-AriaNeural", "en-US-SaraNeural"],
        "male":   ["en-US-GuyNeural", "en-US-ChristopherNeural", "en-US-EricNeural"],
    },
    "hi": {
        "female": ["hi-IN-SwaraNeural"],
        "male":   ["hi-IN-MadhurNeural"],
    },
    "es": {
        "female": ["es-ES-ElviraNeural", "es-MX-DaliaNeural"],
        "male":   ["es-ES-AlvaroNeural", "es-MX-JorgeNeural"],
    },
    "fr": {
        "female": ["fr-FR-DeniseNeural", "fr-FR-EloiseNeural"],
        "male":   ["fr-FR-HenriNeural"],
    },
    "de": {
        "female": ["de-DE-KatjaNeural", "de-DE-AmalaNeural"],
        "male":   ["de-DE-ConradNeural", "de-DE-KillianNeural"],
    },
    "ja": {
        "female": ["ja-JP-NanamiNeural"],
        "male":   ["ja-JP-KeitaNeural"],
    },
    "ko": {
        "female": ["ko-KR-SunHiNeural"],
        "male":   ["ko-KR-InJoonNeural"],
    },
    "zh": {
        "female": ["zh-CN-XiaoxiaoNeural", "zh-CN-XiaohanNeural"],
        "male":   ["zh-CN-YunxiNeural", "zh-CN-YunjianNeural"],
    },
    "pt": {
        "female": ["pt-BR-FranciscaNeural"],
        "male":   ["pt-BR-AntonioNeural"],
    },
    "ru": {
        "female": ["ru-RU-SvetlanaNeural", "ru-RU-DariyaNeural"],
        "male":   ["ru-RU-DmitryNeural"],
    },
    "ar": {
        "female": ["ar-SA-ZariyahNeural"],
        "male":   ["ar-SA-HamedNeural"],
    },
    "it": {
        "female": ["it-IT-ElsaNeural", "it-IT-IsabellaNeural"],
        "male":   ["it-IT-DiegoNeural"],
    },
    "tr": {
        "female": ["tr-TR-EmelNeural"],
        "male":   ["tr-TR-AhmetNeural"],
    },
    "bn": {
        "female": ["bn-IN-TanishaaNeural"],
        "male":   ["bn-IN-BashkarNeural"],
    },
    "ta": {
        "female": ["ta-IN-PallaviNeural"],
        "male":   ["ta-IN-ValluvarNeural"],
    },
    "te": {
        "female": ["te-IN-ShrutiNeural"],
        "male":   ["te-IN-MohanNeural"],
    },
    "mr": {
        "female": ["mr-IN-AarohiNeural"],
        "male":   ["mr-IN-ManoharNeural"],
    },
    "gu": {
        "female": ["gu-IN-DhwaniNeural"],
        "male":   ["gu-IN-NiranjanNeural"],
    },
    "kn": {
        "female": ["kn-IN-SapnaNeural"],
        "male":   ["kn-IN-GaganNeural"],
    },
    "ml": {
        "female": ["ml-IN-SobhanaNeural"],
        "male":   ["ml-IN-MidhunNeural"],
    },
    "pa": {
        "female": ["pa-IN-GurpreetNeural"],
        "male":   ["pa-IN-GurpreetNeural"],
    },
    "ur": {
        "female": ["ur-PK-UzmaNeural"],
        "male":   ["ur-PK-AsadNeural"],
    },
}


@dataclass
class PipelineConfig:
    source: str
    work_dir: Path
    output_path: Path
    source_language: str = "auto"
    target_language: str = "hi"
    asr_model: str = "large-v3"
    translation_engine: str = "auto"   # auto, gemini, groq, ollama, google, hinglish
    tts_voice: str = "hi-IN-SwaraNeural"
    tts_rate: str = "+0%"
    mix_original: bool = False
    original_volume: float = 0.10
    use_chatterbox: bool = True
    use_elevenlabs: bool = False
    use_google_tts: bool = False
    use_coqui_xtts: bool = False
    use_edge_tts: bool = False
    prefer_youtube_subs: bool = False
    multi_speaker: bool = False
    transcribe_only: bool = False
    # Audio priority: let TTS speak at natural pace, video freezes/extends to match
    audio_priority: bool = False
    # Audio quality: bitrate for final output (128k, 192k, 256k, 320k)
    audio_bitrate: str = "192k"
    # Video encode speed: ultrafast (fastest), veryfast, fast, medium (best quality)
    encode_preset: str = "veryfast"


class Pipeline:
    """Dubbing pipeline with translation and callback-based progress."""

    SAMPLE_RATE = 48000
    N_CHANNELS = 2

    def __init__(self, cfg: PipelineConfig, on_progress: Optional[ProgressCallback] = None):
        self.cfg = cfg
        self._on_progress = on_progress or (lambda *_: None)
        self.segments: List[Dict] = []
        self.video_title: str = ""
        self._voice_map = None
        self._whisper_audio = None  # Lightweight 16kHz mono audio for transcription
        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)

        # Resolve executable paths
        self._ytdlp = self._find_executable("yt-dlp")
        self._ffmpeg = "ffmpeg"  # resolved in _ensure_ffmpeg

    @staticmethod
    def _find_executable(name: str) -> str:
        """Find an executable by checking venv, PATH, WinGet packages, and system PATH."""
        ext = ".exe" if sys.platform == "win32" else ""
        full_name = name + ext

        def _works(path: str) -> bool:
            """Verify executable actually runs (catches broken venv shims)."""
            try:
                r = subprocess.run([path, "--version"], capture_output=True, timeout=5)
                return r.returncode == 0 and (r.stdout or r.stderr)
            except Exception:
                return False

        # 1. Check venv Scripts dir (where python.exe lives)
        venv_path = Path(sys.executable).parent / full_name
        if venv_path.exists() and _works(str(venv_path)):
            return str(venv_path)

        # 2. Check current PATH (verify it works to skip broken shims)
        found = shutil.which(name)
        if found and _works(found):
            return found

        # 2b. Check Python's own Scripts directory (pip-installed tools)
        scripts_path = Path(sys.executable).parent / "Scripts" / full_name
        if scripts_path.exists() and _works(str(scripts_path)):
            os.environ["PATH"] = str(scripts_path.parent) + os.pathsep + os.environ.get("PATH", "")
            return str(scripts_path)

        if sys.platform == "win32":
            # 3. Scan WinGet packages directory
            localappdata = os.environ.get("LOCALAPPDATA", "")
            if not localappdata:
                userprofile = os.environ.get("USERPROFILE", str(Path.home()))
                localappdata = str(Path(userprofile) / "AppData" / "Local")
            if localappdata:
                winget_pkgs = Path(localappdata) / "Microsoft" / "WinGet" / "Packages"
                if winget_pkgs.exists():
                    for exe in winget_pkgs.rglob(full_name):
                        os.environ["PATH"] = str(exe.parent) + os.pathsep + os.environ.get("PATH", "")
                        return str(exe)

            # 4. Check well-known Windows installation directories
            well_known_dirs = [
                Path(os.environ.get("PROGRAMFILES", "C:\\Program Files")),
                Path(os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)")),
                Path(os.environ.get("LOCALAPPDATA", "")),
            ]
            # Map executable names to their typical install subdirectories
            known_subdirs = {
                "node": ["nodejs"],
                "ffmpeg": ["ffmpeg", "ffmpeg\\bin"],
            }
            for base_dir in well_known_dirs:
                if not base_dir or not base_dir.exists():
                    continue
                for subdir in known_subdirs.get(name, []):
                    candidate = base_dir / subdir / full_name
                    if candidate.exists() and _works(str(candidate)):
                        os.environ["PATH"] = str(candidate.parent) + os.pathsep + os.environ.get("PATH", "")
                        return str(candidate)

            # 5. Refresh PATH from system registry and try again
            try:
                result = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-Command",
                     "[System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    os.environ["PATH"] = result.stdout.strip() + os.pathsep + os.environ.get("PATH", "")
                    found = shutil.which(name)
                    if found and _works(found):
                        return found
            except Exception:
                pass

        return name  # fallback to bare name

    def _report(self, step: str, progress: float, message: str):
        """Report progress to callback."""
        self._on_progress(step, min(progress, 1.0), message)

    # ── Speaker Diarization ───────────────────────────────────────────────

    def _diarize(self, wav_path: Path) -> tuple:
        """Run pyannote speaker diarization.
        Returns (speaker_genders, speaker_ranges) or ({}, {}) on failure.
        """
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if not hf_token:
            self._report("transcribe", 0.85, "HF_TOKEN not set — skipping speaker diarization")
            return {}, {}

        try:
            from pyannote.audio import Pipeline as PyannotePipeline
        except ImportError:
            self._report("transcribe", 0.85, "pyannote-audio not installed — skipping diarization")
            return {}, {}

        try:
            self._report("transcribe", 0.82, "Loading speaker diarization model...")
            diarize_pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )

            # Move to GPU if available
            try:
                import torch
                if torch.cuda.is_available():
                    diarize_pipeline.to(torch.device("cuda"))
            except Exception:
                pass

            self._report("transcribe", 0.86, "Running speaker diarization...")
            diarization = diarize_pipeline(str(wav_path))

            # Extract unique speakers and their time ranges
            speaker_ranges: Dict[str, List[tuple]] = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speaker_ranges:
                    speaker_ranges[speaker] = []
                speaker_ranges[speaker].append((turn.start, turn.end))

            if not speaker_ranges:
                return {}, {}

            self._report("transcribe", 0.92,
                         f"Found {len(speaker_ranges)} speakers, detecting genders...")

            # Detect gender via pitch analysis
            speaker_genders = self._detect_speaker_genders(wav_path, speaker_ranges)
            self._report("transcribe", 0.98,
                         f"Speakers: {', '.join(f'{k}={v}' for k, v in speaker_genders.items())}")
            return speaker_genders, speaker_ranges

        except Exception as e:
            self._report("transcribe", 0.85,
                         f"Diarization failed ({e}) — using single voice")
            return {}, {}

    def _detect_speaker_genders(self, wav_path: Path, speakers: Dict[str, List[tuple]]) -> Dict[str, str]:
        """Detect gender per speaker using pitch (F0) analysis. Male < 165Hz, Female >= 165Hz.
        Reads only needed time ranges to avoid OOM on long videos."""
        import struct

        with wave.open(str(wav_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            max_val = float(2 ** (8 * sample_width - 1))
            fmt_char = "h" if sample_width == 2 else "i"

            result = {}
            for speaker, time_ranges in speakers.items():
                speaker_samples: list = []
                for t_start, t_end in time_ranges[:10]:
                    s_start = max(0, min(int(t_start * sample_rate), n_frames - 1))
                    s_end = max(0, min(int(t_end * sample_rate), n_frames))
                    count = s_end - s_start
                    if count < 1:
                        continue
                    wf.setpos(s_start)
                    raw = wf.readframes(count)
                    try:
                        chunk = struct.unpack(f"<{count * n_channels}{fmt_char}", raw)
                    except struct.error:
                        continue
                    if n_channels > 1:
                        chunk = chunk[::n_channels]
                    speaker_samples.extend(s / max_val for s in chunk)

                if len(speaker_samples) < sample_rate * 0.5:
                    result[speaker] = "female"
                    continue

                pitch = self._estimate_pitch_autocorrelation(speaker_samples, sample_rate)
                result[speaker] = "male" if pitch < 165 else "female"

        return result

    def _estimate_pitch_autocorrelation(self, samples: list, sample_rate: int) -> float:
        """Lightweight autocorrelation pitch estimator. Returns average F0 in Hz."""
        window_size = int(0.03 * sample_rate)  # 30ms windows
        hop = window_size // 2
        min_lag = int(sample_rate / 350)  # Max 350Hz
        max_lag = int(sample_rate / 60)   # Min 60Hz

        pitches = []
        for start in range(0, len(samples) - window_size, hop * 4):  # Skip windows for speed
            window = samples[start:start + window_size]
            # Simple energy check — skip silence
            energy = sum(s * s for s in window) / len(window)
            if energy < 0.001:
                continue

            # Autocorrelation for pitch detection
            best_lag = min_lag
            best_corr = -1.0
            for lag in range(min_lag, min(max_lag, len(window))):
                corr = 0.0
                for j in range(len(window) - lag):
                    corr += window[j] * window[j + lag]
                corr /= (len(window) - lag)
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag

            if best_corr > energy * 0.3:  # Confidence threshold
                pitches.append(sample_rate / best_lag)

        if not pitches:
            return 200.0  # Default to ambiguous range

        # Return median pitch
        pitches.sort()
        return pitches[len(pitches) // 2]

    def _assign_speaker_to_segments(self, segments: List[Dict], diarization_speakers: Dict[str, List[tuple]]):
        """Assign speaker labels to transcription segments by max temporal overlap."""
        for seg in segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            best_speaker = None
            best_overlap = 0.0

            for speaker, time_ranges in diarization_speakers.items():
                overlap = 0.0
                for t_start, t_end in time_ranges:
                    ov_start = max(seg_start, t_start)
                    ov_end = min(seg_end, t_end)
                    if ov_end > ov_start:
                        overlap += ov_end - ov_start

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker

            seg["speaker_id"] = best_speaker or "SPEAKER_00"

    def _assign_voices_to_speakers(self, speaker_genders: Dict[str, str]) -> Dict[str, str]:
        """Map each speaker to a distinct Edge-TTS voice from VOICE_POOL."""
        lang = self.cfg.target_language
        pool = VOICE_POOL.get(lang, {})
        female_voices = list(pool.get("female", [DEFAULT_VOICES.get(lang, "en-US-JennyNeural")]))
        male_voices = list(pool.get("male", [MALE_VOICES.get(lang, "en-US-GuyNeural")]))

        voice_map = {}
        female_idx = 0
        male_idx = 0

        for speaker, gender in sorted(speaker_genders.items()):
            if gender == "male":
                voice_map[speaker] = male_voices[male_idx % len(male_voices)]
                male_idx += 1
            else:
                voice_map[speaker] = female_voices[female_idx % len(female_voices)]
                female_idx += 1

        return voice_map

    def _save_speaker_refs(self, audio_raw: Path, segments: List[Dict]):
        """Extract and save per-speaker voice reference clips for Coqui XTTS.

        Combines multiple short segments from each speaker into a single
        10-15 second reference clip for voice cloning.
        """
        refs_dir = self.cfg.work_dir / "speaker_refs"
        refs_dir.mkdir(exist_ok=True)

        # Group segments by speaker
        speaker_segs: Dict[str, List[Dict]] = {}
        for seg in segments:
            spk = seg.get("speaker_id")
            if spk:
                speaker_segs.setdefault(spk, []).append(seg)

        if not speaker_segs:
            return

        self._report("transcribe", 0.92,
                     f"Saving voice references for {len(speaker_segs)} speakers...")

        for spk, segs in speaker_segs.items():
            # Sort by start time and collect up to ~15 seconds of audio
            segs_sorted = sorted(segs, key=lambda s: s["start"])
            total_dur = 0.0
            clip_parts = []
            concat_txt = refs_dir / f"{spk}_concat.txt"

            for seg in segs_sorted:
                if total_dur >= 15.0:
                    break
                dur = seg["end"] - seg["start"]
                if dur < 0.5:
                    continue
                # Extract this segment's audio
                clip_path = refs_dir / f"{spk}_clip_{len(clip_parts):03d}.wav"
                try:
                    subprocess.run(
                        [self._ffmpeg, "-y",
                         "-i", str(audio_raw),
                         "-ss", f"{seg['start']:.3f}",
                         "-t", f"{dur:.3f}",
                         "-ar", "22050", "-ac", "1",
                         str(clip_path)],
                        check=True, capture_output=True,
                    )
                    clip_parts.append(clip_path)
                    total_dur += dur
                except Exception:
                    continue

            if not clip_parts:
                continue

            # Concatenate clips into one reference file
            ref_path = refs_dir / f"{spk}.wav"
            if len(clip_parts) == 1:
                shutil.copy2(clip_parts[0], ref_path)
            else:
                concat_txt.write_text(
                    "\n".join(f"file '{str(c).replace(chr(92), '/')}'" for c in clip_parts),
                    encoding="utf-8",
                )
                try:
                    subprocess.run(
                        [self._ffmpeg, "-y",
                         "-f", "concat", "-safe", "0",
                         "-i", str(concat_txt),
                         "-ar", "22050", "-ac", "1",
                         str(ref_path)],
                        check=True, capture_output=True,
                    )
                except Exception:
                    # Fall back to just the first clip
                    shutil.copy2(clip_parts[0], ref_path)

            # Clean up individual clips
            for c in clip_parts:
                c.unlink(missing_ok=True)
            concat_txt.unlink(missing_ok=True)

        saved = list(refs_dir.glob("SPEAKER_*.wav"))
        self._report("transcribe", 0.95,
                     f"Saved {len(saved)} speaker voice references for XTTS voice cloning")

    # ── Main entry ───────────────────────────────────────────────────────
    def run(self):
        """Execute the full dubbing pipeline."""
        self._ensure_ffmpeg()

        # Step 1: Download / ingest
        self._report("download", 0.0, "Downloading video...")
        video_path = self._ingest_source(self.cfg.source)
        self._report("download", 1.0, f"Downloaded: {video_path.name}")

        # Step 2: Extract audio
        self._report("extract", 0.0, "Extracting audio...")
        audio_raw = self._extract_audio(video_path)
        self._report("extract", 1.0, "Audio extracted")

        # Step 3: Transcribe — try YouTube subs first, fall back to Whisper
        sub_segments = None
        if self.cfg.prefer_youtube_subs:
            self._report("transcribe", 0.0, "Checking for YouTube subtitles...")
            sub_segments = self._fetch_youtube_subtitles(self.cfg.source)

        if sub_segments:
            self.segments = sub_segments
            self._report("transcribe", 1.0,
                         f"Using YouTube subtitles ({len(sub_segments)} segments, skipped Whisper)")
        else:
            if self.cfg.prefer_youtube_subs:
                self._report("transcribe", 0.05, "No subtitles found, using Whisper...")
            self._report("transcribe", 0.1, "Loading ASR model...")
            # Use 16kHz mono audio for transcription (smaller, faster upload for Groq API)
            whisper_audio = getattr(self, '_whisper_audio', audio_raw)
            self.segments = self._transcribe(whisper_audio)
            self._report("transcribe", 1.0, f"Transcribed {len(self.segments)} segments")

        text_segments = [s for s in self.segments if s.get("text", "").strip()]
        if not text_segments:
            raise RuntimeError("No speech detected in the video")

        # Multi-speaker diarization (runs within "transcribe" step progress 82-98%)
        self._voice_map = None
        if self.cfg.multi_speaker:
            speaker_genders, speaker_ranges = self._diarize(audio_raw)
            if speaker_genders and speaker_ranges:
                self._assign_speaker_to_segments(text_segments, speaker_ranges)
                self._voice_map = self._assign_voices_to_speakers(speaker_genders)
                # Save per-speaker voice refs for Coqui XTTS voice cloning
                if self.cfg.use_coqui_xtts:
                    self._save_speaker_refs(audio_raw, text_segments)
                self._report("transcribe", 0.99,
                             f"Assigned {len(self._voice_map)} distinct voices")

        # Transcribe-only mode: save source SRT, extract per-speaker refs, and stop
        if self.cfg.transcribe_only:
            from srt_utils import write_srt as _write_srt
            source_srt = self.cfg.work_dir / "transcript_source.srt"
            has_speakers = any("speaker_id" in s for s in text_segments)
            _write_srt(text_segments, source_srt, text_key="text",
                       include_speaker=has_speakers)

            # Save per-speaker voice reference clips for Coqui XTTS voice cloning
            if has_speakers:
                self._save_speaker_refs(audio_raw, text_segments)

            speaker_note = f" ({len(set(s.get('speaker_id','') for s in text_segments if s.get('speaker_id')))} speakers detected)" if has_speakers else ""
            self._report("transcribe", 1.0,
                         f"Transcription complete — {len(text_segments)} segments{speaker_note}. Download SRT to translate.")
            return  # Pipeline pauses here; user translates externally

        # Step 4: Translate each segment (preserving timestamps for scene sync)
        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        self._report("translate", 0.0, f"Translating segments to {target_name}...")
        self._translate_segments(text_segments)
        self.segments = text_segments
        self._report("translate", 1.0, "Translation complete")

        # Write translated SRT (per-segment subtitles with proper timestamps)
        srt_translated = self.cfg.work_dir / f"transcript_{self.cfg.target_language}.srt"
        write_srt(self.segments, srt_translated, text_key="text_translated")

        # Step 5: Generate TTS (Edge-TTS already includes 1.25x speedup in one pass)
        self._report("synthesize", 0.0,
                     f"Generating speech ({self.cfg.tts_voice})...")
        tts_data = self._generate_tts_natural(text_segments)
        if not tts_data:
            raise RuntimeError("TTS synthesis produced no audio segments — check TTS engine settings")

        # Speed up segments that weren't already sped (Edge-TTS does it in MP3→WAV pass)
        needs_speedup = [(i, t) for i, t in enumerate(tts_data) if not t.get("_already_sped")]
        if needs_speedup:
            self._report("synthesize", 0.9,
                         f"Speeding up {len(needs_speedup)} segments to 1.25x...")
            TTS_SPEED = 1.25
            from concurrent.futures import ThreadPoolExecutor

            def speedup_one(idx_tts):
                idx, tts = idx_tts
                sped_wav = self.cfg.work_dir / f"tts_fast_{idx:04d}.wav"
                self._time_stretch(tts["wav"], TTS_SPEED, sped_wav)
                tts["wav"] = sped_wav
                tts["duration"] = tts["duration"] / TTS_SPEED

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(speedup_one, needs_speedup))

        # Clean up internal flag
        for t in tts_data:
            t.pop("_already_sped", None)

        self._report("synthesize", 1.0,
                     f"All {len(tts_data)} segments ready")

        # Step 6: Assemble — AUDIO IS MASTER, video adapts
        self._report("assemble", 0.0, "Building dubbed output...")
        self.cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        video_duration = self._get_duration(video_path)

        if self.cfg.audio_priority:
            self._report("assemble", 0.05, "Fast assembly: audio priority mode (stream copy)...")
            self._assemble_fast_mux(video_path, audio_raw, tts_data, video_duration)
        else:
            self._report("assemble", 0.1, "Assembling: video will adapt to match natural audio...")
            self._assemble_video_adapts_to_audio(video_path, audio_raw, tts_data, video_duration)

        # Copy SRT to output
        out_srt = self.cfg.output_path.parent / f"subtitles_{self.cfg.target_language}.srt"
        shutil.copy2(srt_translated, out_srt)

        self._report("assemble", 1.0, "Done!")

    def run_from_srt(self, translated_srt_path: Path):
        """Resume pipeline from an uploaded translated SRT — runs TTS + assembly only."""
        from srt_utils import parse_srt

        self._ensure_ffmpeg()

        # Load translated segments
        self._report("translate", 0.0, "Loading uploaded translated SRT...")
        translated = parse_srt(translated_srt_path, text_key="text_translated")
        if not translated:
            raise RuntimeError("Uploaded SRT is empty or invalid")

        self.segments = translated

        # Check for speaker labels from SRT and rebuild voice map
        speakers_found = set(s.get("speaker_id") for s in translated if s.get("speaker_id"))
        if speakers_found and self.cfg.multi_speaker:
            self._report("translate", 0.5,
                         f"Found {len(speakers_found)} speakers in SRT, assigning voices...")
            # Re-run gender detection from original audio if available
            audio_raw = self.cfg.work_dir / "audio_raw.wav"
            if audio_raw.exists():
                speaker_ranges = {}
                for seg in translated:
                    spk = seg.get("speaker_id")
                    if spk:
                        if spk not in speaker_ranges:
                            speaker_ranges[spk] = []
                        speaker_ranges[spk].append((seg["start"], seg["end"]))
                speaker_genders = self._detect_speaker_genders(audio_raw, speaker_ranges)
                self._voice_map = self._assign_voices_to_speakers(speaker_genders)
                self._report("translate", 0.8,
                             f"Assigned voices: {', '.join(f'{k}={v}' for k, v in self._voice_map.items())}")
            else:
                # No audio for gender detection — alternate male/female
                self._voice_map = {}
                lang = self.cfg.target_language
                pool = VOICE_POOL.get(lang, {})
                female_voices = list(pool.get("female", [DEFAULT_VOICES.get(lang, "en-US-JennyNeural")]))
                male_voices = list(pool.get("male", [MALE_VOICES.get(lang, "en-US-GuyNeural")]))
                for i, spk in enumerate(sorted(speakers_found)):
                    if i % 2 == 0:
                        self._voice_map[spk] = female_voices[i // 2 % len(female_voices)]
                    else:
                        self._voice_map[spk] = male_voices[i // 2 % len(male_voices)]
        else:
            self._voice_map = None

        self._report("translate", 1.0, f"Loaded {len(translated)} translated segments")

        # Find existing video and audio from first run
        video_path = None
        for ext in ["mp4", "webm", "mkv"]:
            p = self.cfg.work_dir / f"source.{ext}"
            if p.exists():
                video_path = p
                break
        if not video_path:
            sources = list(self.cfg.work_dir.glob("source.*"))
            video_path = sources[0] if sources else None
        if not video_path:
            raise RuntimeError("Original video not found in work directory")

        audio_raw = self.cfg.work_dir / "audio_raw.wav"
        if not audio_raw.exists():
            self._report("extract", 0.0, "Re-extracting audio...")
            audio_raw = self._extract_audio(video_path)

        # Write the translated SRT to standard location
        srt_translated = self.cfg.work_dir / f"transcript_{self.cfg.target_language}.srt"
        shutil.copy2(translated_srt_path, srt_translated)

        # Step 5: Generate TTS (Edge-TTS includes 1.25x speedup in one pass)
        text_segments = translated
        self._report("synthesize", 0.0,
                     f"Generating speech ({self.cfg.tts_voice})...")
        tts_data = self._generate_tts_natural(text_segments)
        if not tts_data:
            raise RuntimeError("TTS synthesis produced no audio segments — check TTS engine settings")

        needs_speedup = [(i, t) for i, t in enumerate(tts_data) if not t.get("_already_sped")]
        if needs_speedup:
            self._report("synthesize", 0.9, "Speeding up to 1.25x...")
            TTS_SPEED = 1.25
            from concurrent.futures import ThreadPoolExecutor

            def speedup_one(idx_tts):
                idx, tts = idx_tts
                sped_wav = self.cfg.work_dir / f"tts_fast_{idx:04d}.wav"
                self._time_stretch(tts["wav"], TTS_SPEED, sped_wav)
                tts["wav"] = sped_wav
                tts["duration"] = tts["duration"] / TTS_SPEED

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(speedup_one, needs_speedup))

        for t in tts_data:
            t.pop("_already_sped", None)

        self._report("synthesize", 1.0,
                     f"All {len(tts_data)} segments ready")

        # Step 6: Assemble
        self._report("assemble", 0.0, "Building dubbed output...")
        self.cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        video_duration = self._get_duration(video_path)

        if self.cfg.audio_priority:
            self._report("assemble", 0.05, "Fast assembly: audio priority mode (stream copy)...")
            self._assemble_fast_mux(video_path, audio_raw, tts_data, video_duration)
        else:
            self._report("assemble", 0.1, "Assembling: video will adapt to match natural audio...")
            self._assemble_video_adapts_to_audio(video_path, audio_raw, tts_data, video_duration)

        # Copy SRT to output
        out_srt = self.cfg.output_path.parent / f"subtitles_{self.cfg.target_language}.srt"
        shutil.copy2(srt_translated, out_srt)

        self._report("assemble", 1.0, "Done!")

    # ── FFmpeg check ─────────────────────────────────────────────────────
    def _ensure_ffmpeg(self):
        resolved = self._find_executable("ffmpeg")

        # Also scan WinGet install paths as last resort
        if resolved == "ffmpeg" and sys.platform == "win32":
            localappdata = os.environ.get("LOCALAPPDATA", "")
            # Fallback: derive LOCALAPPDATA from USERPROFILE if not set
            if not localappdata:
                userprofile = os.environ.get("USERPROFILE", str(Path.home()))
                localappdata = str(Path(userprofile) / "AppData" / "Local")
            winget_ffmpeg = Path(localappdata) / "Microsoft" / "WinGet" / "Packages"
            if winget_ffmpeg.exists():
                for exe in winget_ffmpeg.rglob("ffmpeg.exe"):
                    resolved = str(exe)
                    os.environ["PATH"] = str(exe.parent) + os.pathsep + os.environ.get("PATH", "")
                    break

        if resolved == "ffmpeg" and shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "FFmpeg not found! Install: winget install Gyan.FFmpeg"
            )
        self._ffmpeg = resolved

    # ── Step 1: Ingest ───────────────────────────────────────────────────
    def _find_cookies_file(self) -> Optional[str]:
        """Find a YouTube cookies file if available."""
        # Check common locations
        for path in [
            Path(__file__).resolve().parent / "cookies.txt",
            Path.home() / "cookies.txt",
            Path("/content/cookies.txt"),  # Colab
        ]:
            if path.exists():
                return str(path)
        return None

    def _ingest_source(self, src: str) -> Path:
        if re.match(r"^https?://", src):
            out_tpl = str(self.cfg.work_dir / "source.%(ext)s")

            # Check for cookies file (needed on Colab/servers)
            cookies_file = self._find_cookies_file()
            cookies_args = ["--cookies", cookies_file] if cookies_file else []

            # Enable Node.js runtime for YouTube extraction (required since yt-dlp 2025+)
            node_path = self._find_executable("node")
            # Only use --js-runtimes if we found a real path (not bare "node" fallback)
            js_args = ["--js-runtimes", f"node:{node_path}"] if (node_path and node_path != "node" and Path(node_path).exists()) else []

            # Fetch video title first (--print is print-only in yt-dlp 2025+)
            try:
                title_cmd = [
                    self._ytdlp, "--print", "%(title)s",
                ] + cookies_args + js_args + [src]
                print(f"[YTDLP] title cmd: {title_cmd}", flush=True)
                title_result = subprocess.run(title_cmd, capture_output=True, text=True, timeout=60)
                print(f"[YTDLP] title rc={title_result.returncode} stdout={repr((title_result.stdout or '')[:200])}", flush=True)
                if title_result.returncode != 0:
                    print(f"[YTDLP] title stderr: {(title_result.stderr or '')[:300]}", flush=True)
                title_line = (title_result.stdout or "").strip().split("\n")[0].strip()
                if title_line:
                    self.video_title = title_line
                    print(f"[YTDLP] Got title: {self.video_title}", flush=True)
                else:
                    print(f"[YTDLP] No title in output, using fallback", flush=True)
            except Exception as e:
                print(f"[YTDLP] Title fetch failed: {e}", flush=True)

            # Download video
            try:
                dl_cmd = [
                    self._ytdlp,
                    "-f", "bv*+ba/b",
                    "--merge-output-format", "mp4",
                    "-o", out_tpl,
                ]
                # Only add --ffmpeg-location if we have a real path (not bare "ffmpeg")
                ffmpeg_path = Path(self._ffmpeg)
                if ffmpeg_path.is_absolute():
                    dl_cmd += ["--ffmpeg-location", str(ffmpeg_path.parent)]
                dl_cmd += cookies_args + js_args + [src]
                print(f"[YTDLP] cmd: {dl_cmd}", flush=True)
                result = subprocess.run(dl_cmd, capture_output=True, text=True)
                print(f"[YTDLP] rc={result.returncode} stdout_len={len(result.stdout or '')} stderr_len={len(result.stderr or '')}", flush=True)
                if result.returncode != 0:
                    error_msg = (result.stderr or result.stdout or "Unknown error").strip()
                    print(f"[YTDLP] error: {error_msg[:500]}", flush=True)
                    raise RuntimeError(f"yt-dlp failed: {error_msg}")
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"yt-dlp failed: {e}") from e

            self._report("download", 0.9, f"Downloaded: {self.video_title}")

            # Find downloaded file
            mp4 = list(self.cfg.work_dir.glob("source.mp4"))
            if mp4:
                return mp4[0]
            all_sources = list(self.cfg.work_dir.glob("source.*"))
            if all_sources:
                return all_sources[0]
            raise RuntimeError("Download completed but no video file found in work directory")

        p = Path(src)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            raise FileNotFoundError(f"Source not found: {src}")
        self.video_title = p.stem
        return p

    # ── Step 2: Extract audio ────────────────────────────────────────────
    def _extract_audio(self, video_path: Path) -> Path:
        """Extract audio from video. Creates both full-quality and lightweight versions.
        Full-quality (48kHz stereo) for final mix, lightweight (16kHz mono) for transcription."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        wav = self.cfg.work_dir / "audio_raw.wav"
        wav_16k = self.cfg.work_dir / "audio_16k.wav"

        def extract_full():
            subprocess.run(
                [self._ffmpeg, "-y", "-i", str(video_path),
                 "-vn", "-ac", str(self.N_CHANNELS), "-ar", str(self.SAMPLE_RATE),
                 "-acodec", "pcm_s16le", str(wav)],
                check=True, capture_output=True,
            )
            return "full"

        def extract_16k():
            subprocess.run(
                [self._ffmpeg, "-y", "-i", str(video_path),
                 "-vn", "-ac", "1", "-ar", "16000",
                 "-acodec", "pcm_s16le", str(wav_16k)],
                check=True, capture_output=True,
            )
            return "16k"

        # Extract both in parallel — saves ~50% extraction time
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_full = pool.submit(extract_full)
            fut_16k = pool.submit(extract_16k)

            # Full-quality extraction is mandatory
            fut_full.result()

            # 16kHz is optional (for Groq API) — fall back to full if it fails
            try:
                fut_16k.result()
                self._whisper_audio = wav_16k
            except Exception:
                self._whisper_audio = wav  # Fall back to full-quality for transcription

        return wav

    # ── Step 3a: Fetch YouTube subtitles (skip Whisper if available) ─────
    @staticmethod
    def _vtt_time_to_seconds(time_str: str) -> float:
        """Convert VTT/SRT timestamp (HH:MM:SS.mmm) to seconds."""
        parts = time_str.split(":")
        h, m = int(parts[0]), int(parts[1])
        s_parts = parts[2].replace(",", ".").split(".")
        s = int(s_parts[0])
        ms = int(s_parts[1]) if len(s_parts) > 1 else 0
        return h * 3600 + m * 60 + s + ms / 1000.0

    _NOISE_RE = re.compile(r"^\[.*\]$|^\(.*\)$|^♪.*♪$")

    def _parse_vtt(self, vtt_path: Path) -> List[Dict]:
        """Parse a WebVTT file into pipeline segment format."""
        content = vtt_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        segments_raw: List[Dict] = []
        i = 0
        while i < len(lines):
            m = re.match(
                r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})",
                lines[i].strip(),
            )
            if m:
                start = self._vtt_time_to_seconds(m.group(1))
                end = self._vtt_time_to_seconds(m.group(2))
                text_lines = []
                i += 1
                while i < len(lines) and lines[i].strip() and not re.match(
                    r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->", lines[i].strip()
                ):
                    text_lines.append(lines[i].strip())
                    i += 1
                raw_text = " ".join(text_lines)
                clean_text = re.sub(r"<[^>]+>", "", raw_text).strip()
                if clean_text and not self._NOISE_RE.match(clean_text):
                    segments_raw.append({"start": start, "end": end, "text": clean_text})
            else:
                i += 1

        if not segments_raw:
            return []

        # Deduplicate YouTube auto-gen rolling two-line format
        deduped = [segments_raw[0]]
        for seg in segments_raw[1:]:
            prev_text = deduped[-1]["text"]
            curr_text = seg["text"]
            if prev_text in curr_text:
                deduped[-1] = seg
            elif curr_text in prev_text:
                continue
            else:
                deduped.append(seg)

        # Merge adjacent segments with identical text
        merged = [deduped[0]]
        for seg in deduped[1:]:
            if seg["text"] == merged[-1]["text"] and seg["start"] - merged[-1]["end"] < 0.5:
                merged[-1]["end"] = seg["end"]
            else:
                merged.append(seg)

        return merged

    def _parse_srt_file(self, srt_path: Path) -> List[Dict]:
        """Parse an SRT file into pipeline segment format."""
        content = srt_path.read_text(encoding="utf-8")
        segments: List[Dict] = []
        pattern = re.compile(
            r"\d+\s*\n"
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n"
            r"((?:(?!\d+\s*\n\d{2}:\d{2}).+\n?)+)",
            re.MULTILINE,
        )
        for m in pattern.finditer(content):
            start_str = m.group(1).replace(",", ".")
            end_str = m.group(2).replace(",", ".")
            text = re.sub(r"<[^>]+>", "", m.group(3)).strip()
            text = " ".join(text.split())
            if text and not self._NOISE_RE.match(text):
                segments.append({
                    "start": self._vtt_time_to_seconds(start_str),
                    "end": self._vtt_time_to_seconds(end_str),
                    "text": text,
                })
        return segments

    def _fetch_youtube_subtitles(self, url: str) -> Optional[List[Dict]]:
        """Try to download and parse YouTube subtitles. Returns segments or None."""
        if not re.match(r"^https?://", url):
            return None  # Local file, no YouTube subs

        lang = self.cfg.source_language if self.cfg.source_language != "auto" else "en"
        cookies_file = self._find_cookies_file()
        cookies_args = ["--cookies", cookies_file] if cookies_file else []

        sub_dir = self.cfg.work_dir / "subs"
        sub_dir.mkdir(exist_ok=True)
        out_tpl = str(sub_dir / "sub.%(ext)s")

        for write_flag in ["--write-sub", "--write-auto-sub"]:
            # Clean previous attempt
            for f in sub_dir.glob("*"):
                try:
                    f.unlink()
                except OSError:
                    pass

            cmd = [
                self._ytdlp,
                write_flag,
                "--sub-lang", lang,
                "--sub-format", "vtt/srt/best",
                "--skip-download",
                "-o", out_tpl,
            ] + cookies_args + [url]

            try:
                subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            except Exception:
                continue

            # Look for downloaded subtitle files
            for vtt_file in sub_dir.glob("*.vtt"):
                segments = self._parse_vtt(vtt_file)
                if segments:
                    return segments
            for srt_file in sub_dir.glob("*.srt"):
                segments = self._parse_srt_file(srt_file)
                if segments:
                    return segments

        return None

    # ── Step 3b: Transcribe speech from audio ─────────────────────────────
    def _transcribe(self, wav_path: Path) -> List[Dict]:
        """Transcribe speech from audio — tries Groq Whisper API first (30x faster),
        falls back to local faster-whisper if no API key or API fails."""
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        if groq_key:
            try:
                return self._transcribe_groq(wav_path, groq_key)
            except Exception as e:
                self._report("transcribe", 0.1, f"Groq Whisper failed ({e}), using local model...")

        return self._transcribe_local(wav_path)

    def _transcribe_groq(self, wav_path: Path, api_key: str) -> List[Dict]:
        """Transcribe using Groq Whisper API — ~25s per hour of audio.
        Handles files >25MB by chunking into smaller pieces."""
        import requests as _requests

        self._report("transcribe", 0.1, "Using Groq Whisper API (cloud, ultra-fast)...")

        # Groq has a 25MB file limit — check if we need to chunk
        file_size_mb = wav_path.stat().st_size / (1024 * 1024)

        if file_size_mb <= 24:
            # Single file upload
            return self._groq_whisper_single(wav_path, api_key)

        # Large file: split into chunks and transcribe each
        self._report("transcribe", 0.1,
                     f"Audio is {file_size_mb:.0f}MB — splitting into chunks for Groq API...")
        return self._groq_whisper_chunked(wav_path, api_key)

    def _groq_whisper_single(self, audio_path: Path, api_key: str) -> List[Dict]:
        """Send a single file to Groq Whisper API."""
        import requests as _requests

        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {api_key}"}

        data = {
            "model": "whisper-large-v3",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        }
        if self.cfg.source_language and self.cfg.source_language != "auto":
            data["language"] = self.cfg.source_language

        with open(audio_path, "rb") as f:
            resp = _requests.post(
                url, headers=headers,
                data=data,
                files={"file": (audio_path.name, f, "audio/wav")},
                timeout=300,
            )
        resp.raise_for_status()
        result = resp.json()

        segments = []
        for seg in result.get("segments", []):
            entry = {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"].strip(),
            }
            segments.append(entry)

        self._report("transcribe", 0.95,
                     f"Groq Whisper: {len(segments)} segments transcribed")
        return segments

    def _groq_whisper_chunked(self, wav_path: Path, api_key: str) -> List[Dict]:
        """Split large audio into ~10min chunks, transcribe each via Groq API in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Split into 10-minute chunks (mono 16kHz = ~19MB per 10min)
        chunk_duration = 600  # 10 minutes in seconds
        total_duration = self._get_duration(wav_path)
        num_chunks = math.ceil(total_duration / chunk_duration)

        self._report("transcribe", 0.15,
                     f"Splitting into {num_chunks} chunks ({chunk_duration//60}min each)...")

        # Create chunks using ffmpeg
        chunk_paths = []
        for i in range(num_chunks):
            start_time = i * chunk_duration
            chunk_path = self.cfg.work_dir / f"whisper_chunk_{i:03d}.wav"
            subprocess.run(
                [self._ffmpeg, "-y", "-i", str(wav_path),
                 "-ss", str(start_time), "-t", str(chunk_duration),
                 "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
                 str(chunk_path)],
                check=True, capture_output=True,
            )
            chunk_paths.append((i, start_time, chunk_path))

        # Transcribe chunks in parallel (up to 4 concurrent API calls)
        all_segments = []
        completed = 0

        def transcribe_chunk(args):
            idx, offset, path = args
            segs = self._groq_whisper_single(path, api_key)
            # Adjust timestamps by chunk offset
            for seg in segs:
                seg["start"] += offset
                seg["end"] += offset
            return idx, segs

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(transcribe_chunk, cp): cp[0] for cp in chunk_paths}
            for fut in as_completed(futures):
                idx, segs = fut.result()
                all_segments.append((idx, segs))
                completed += 1
                self._report("transcribe", 0.15 + 0.8 * (completed / num_chunks),
                             f"Transcribed chunk {completed}/{num_chunks} via Groq API...")

        # Sort by chunk index and flatten
        all_segments.sort(key=lambda x: x[0])
        segments = []
        for _, segs in all_segments:
            segments.extend(segs)

        # Clean up chunk files
        for _, _, path in chunk_paths:
            path.unlink(missing_ok=True)

        return segments

    def _transcribe_local(self, wav_path: Path) -> List[Dict]:
        """Transcribe speech from audio using local faster-whisper (GPU/CPU)."""
        from faster_whisper import WhisperModel

        # Auto-detect GPU: use CUDA if available, else fall back to CPU
        device, compute = "cpu", "int8"
        try:
            import torch
            if torch.cuda.is_available():
                device, compute = "cuda", "float16"
        except ImportError:
            pass

        self._report("transcribe", 0.1, f"Loading model ({self.cfg.asr_model}) on {device.upper()}...")
        model = WhisperModel(self.cfg.asr_model, device=device, compute_type=compute)

        self._report("transcribe", 0.2, "Transcribing audio with word timestamps...")
        transcribe_kwargs = {
            "vad_filter": True,
            "word_timestamps": True,
            "beam_size": 1,
            "best_of": 1,
            "condition_on_previous_text": True,
            "no_speech_threshold": 0.5,
            "compression_ratio_threshold": 2.4,
            "vad_parameters": {"min_silence_duration_ms": 300},
        }
        if self.cfg.source_language and self.cfg.source_language != "auto":
            transcribe_kwargs["language"] = self.cfg.source_language
            self._report("transcribe", 0.2, f"Transcribing ({self.cfg.source_language.upper()}) with word timestamps...")
        seg_iter, info = model.transcribe(str(wav_path), **transcribe_kwargs)

        segments: List[Dict] = []
        for seg in seg_iter:
            entry = {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
            }
            if hasattr(seg, "words") and seg.words:
                entry["words"] = [
                    {"word": w.word.strip(), "start": float(w.start), "end": float(w.end)}
                    for w in seg.words
                ]
            segments.append(entry)
            self._report(
                "transcribe",
                min(0.2 + 0.8 * (len(segments) / max(len(segments) + 5, 1)), 0.95),
                f"Transcribed {len(segments)} segments...",
            )

        return segments

    # ── Step 4: Translate full narrative ─────────────────────────────────
    def _translate_full_narrative(self, text_segments: List[Dict], speech_duration: float = 0) -> tuple:
        """Join all speech into one narrative, translate as a whole."""
        # Combine all transcribed text into one continuous story
        full_text = " ".join(s.get("text", "").strip() for s in text_segments if s.get("text", "").strip())
        self._report("translate", 0.2, f"Translating {len(full_text)} characters...")

        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        engine = self.cfg.translation_engine

        # If a specific engine is selected, use it directly
        if engine == "hinglish" and self._ollama_available():
            self._report("translate", 0.25, "Using Hinglish AI (custom Ollama model)...")
            translated_text = self._translate_with_ollama(full_text, speech_duration, force_model="hinglish-translator")
        elif engine == "gemini" and gemini_key:
            self._report("translate", 0.25, "Using Gemini for translation...")
            translated_text = self._translate_with_gemini(full_text, gemini_key, speech_duration)
        elif engine == "groq" and groq_key:
            self._report("translate", 0.25, "Using Groq (Llama 3.3 70B) for translation...")
            translated_text = self._translate_with_groq(full_text, groq_key, speech_duration)
        elif engine == "ollama" and self._ollama_available():
            self._report("translate", 0.25, "Using Ollama (local LLM) for translation...")
            translated_text = self._translate_with_ollama(full_text, speech_duration)
        elif engine == "google":
            self._report("translate", 0.25, "Using Google Translate...")
            translated_text = self._translate_with_google(full_text)
        # Auto mode: try best available
        elif gemini_key:
            translated_text = self._translate_with_gemini(full_text, gemini_key, speech_duration)
        elif groq_key:
            self._report("translate", 0.25, "Using Groq (Llama 3.3 70B) for translation...")
            translated_text = self._translate_with_groq(full_text, groq_key, speech_duration)
        elif self._ollama_available():
            self._report("translate", 0.25, "Using Ollama (local LLM) for translation...")
            translated_text = self._translate_with_ollama(full_text, speech_duration)
        else:
            self._report("translate", 0.25, "No GEMINI/GROQ/Ollama found, using Google Translate...")
            translated_text = self._translate_with_google(full_text)

        return full_text, translated_text

    def _translate_with_gemini(self, full_text: str, api_key: str, speech_duration: float = 0) -> str:
        """Translate using Gemini LLM for natural, fluent output."""
        from google import genai

        client = genai.Client(api_key=api_key)

        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        source_name = LANGUAGE_NAMES.get(self.cfg.source_language, "the source language") if self.cfg.source_language != "auto" else "the source language"

        # Calculate word count guidance for duration matching
        word_count = len(full_text.split())
        duration_hint = ""
        if speech_duration > 0:
            wpm = LANGUAGE_WPM.get(self.cfg.target_language, 135)
            target_words = int(speech_duration / 60 * wpm)
            duration_hint = (
                f"IMPORTANT TIMING CONSTRAINT: The original narration is {int(speech_duration)} seconds long "
                f"({word_count} words). Your {target_name} translation will be spoken by TTS and "
                f"MUST fit within this duration. Aim for approximately {target_words} {target_name} words. "
                f"Be concise — use shorter phrases where possible without losing meaning. "
                f"Avoid filler words and unnecessary elaboration. "
            )

        # Gemini free tier: 10 RPM, so split large texts into chunks
        chunks = self._split_text_for_translation(full_text, max_chars=8000)
        translated_parts = []
        chunk_duration = speech_duration / len(chunks) if speech_duration > 0 else 0

        for i, chunk in enumerate(chunks):
            chunk_words = len(chunk.split())
            chunk_target = int(chunk_duration / 60 * wpm) if chunk_duration > 0 else 0
            chunk_hint = ""
            if chunk_target > 0:
                chunk_hint = (
                    f"This chunk has {chunk_words} English words and must fit in ~{int(chunk_duration)} seconds. "
                    f"Aim for ~{chunk_target} {target_name} words. "
                )

            prompt = (
                self._get_translation_prompt("system") + "\n\n"
                f"This is a voiceover script for a dubbed video. "
                f"{duration_hint}{chunk_hint}"
                f"Do NOT translate literally word-by-word. Adapt idioms and phrasing naturally. "
                f"Output ONLY the translation, nothing else.\n\n"
                f"{chunk}"
            )

            retries = 3
            for attempt in range(retries):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-pro",
                        contents=prompt,
                    )
                    translated_parts.append((response.text or "").strip())
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        wait = 2 * (attempt + 1)
                        self._report("translate", 0.2 + 0.6 * (i / len(chunks)),
                                     f"Rate limited, retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        # Only translate the failed chunk via fallback, keep already-translated parts
                        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
                        if groq_key:
                            self._report("translate", 0.2, f"Gemini failed on chunk {i+1}: {e}, using Groq for this chunk...")
                            fallback = self._translate_with_groq(chunk, groq_key, chunk_duration)
                        else:
                            self._report("translate", 0.2, f"Gemini failed on chunk {i+1}: {e}, using Google for this chunk...")
                            fallback = self._translate_with_google(chunk)
                        translated_parts.append(fallback)

            self._report("translate", 0.2 + 0.8 * ((i + 1) / len(chunks)),
                         f"Translated chunk {i + 1}/{len(chunks)} (Gemini)")

        return " ".join(translated_parts)

    def _translate_with_groq(self, full_text: str, api_key: str, speech_duration: float = 0) -> str:
        """Translate using Groq (Llama 3.3 70B) for natural, fluent output."""
        import requests as _requests

        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        source_name = LANGUAGE_NAMES.get(self.cfg.source_language, "the source language") if self.cfg.source_language != "auto" else "the source language"

        word_count = len(full_text.split())
        duration_hint = ""
        if speech_duration > 0:
            wpm = LANGUAGE_WPM.get(self.cfg.target_language, 135)
            target_words = int(speech_duration / 60 * wpm)
            duration_hint = (
                f"IMPORTANT TIMING CONSTRAINT: The original narration is {int(speech_duration)} seconds long "
                f"({word_count} words). Aim for approximately {target_words} {target_name} words. "
                f"Be concise — use shorter phrases where possible without losing meaning. "
            )

        chunks = self._split_text_for_translation(full_text, max_chars=8000)
        translated_parts = []

        for i, chunk in enumerate(chunks):
            prompt = (
                self._get_translation_prompt("system") + "\n\n"
                f"This is a voiceover script for a dubbed video. "
                f"{duration_hint}"
                f"Do NOT translate literally word-by-word. Adapt idioms and phrasing naturally. "
                f"Output ONLY the translation, nothing else.\n\n"
                f"{chunk}"
            )

            retries = 3
            for attempt in range(retries):
                try:
                    resp = _requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}]},
                        timeout=60,
                    )
                    data = resp.json()
                    if "choices" in data:
                        translated_parts.append(data["choices"][0]["message"]["content"].strip())
                        break
                    else:
                        raise RuntimeError(data.get("error", {}).get("message", "Groq API error"))
                except Exception as e:
                    if attempt < retries - 1:
                        wait = 2 * (attempt + 1)
                        self._report("translate", 0.2 + 0.6 * (i / len(chunks)),
                                     f"Groq rate limited, retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        self._report("translate", 0.2, f"Groq failed: {e}, falling back to Google Translate...")
                        return self._translate_with_google(full_text)

            self._report("translate", 0.2 + 0.8 * ((i + 1) / len(chunks)),
                         f"Translated chunk {i + 1}/{len(chunks)} (Groq)")

        return " ".join(translated_parts)

    def _ollama_available(self) -> bool:
        """Check if Ollama is running locally."""
        import requests as _requests
        try:
            resp = _requests.get("http://localhost:11434/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def _translate_with_ollama(self, full_text: str, speech_duration: float = 0, force_model: str = "") -> str:
        """Translate using local Ollama LLM."""
        import requests as _requests

        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)

        # Pick the best available model
        try:
            resp = _requests.get("http://localhost:11434/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            models = []

        # If a specific model is requested, use it
        if force_model:
            model = force_model if any(force_model in m for m in models) else None
            if not model:
                self._report("translate", 0.25, f"Model '{force_model}' not found in Ollama, falling back...")
                # Try auto-pick instead
                force_model = ""

        if not force_model:
            # Prefer custom hinglish-translator for Hindi, else pick best available
            preferred = (
                ["hinglish-translator"] if self.cfg.target_language in ("hi", "hi-IN")
                else []
            ) + ["qwen2.5:14b", "qwen2.5:32b", "llama3.1:8b", "llama3:8b", "gemma2:9b", "mistral:7b"]
            model = None
            for pref in preferred:
                for m in models:
                    if pref.split(":")[0] in m:
                        model = m
                        break
                if model:
                    break
            if not model and models:
                model = models[0]
        if not model:
            self._report("translate", 0.25, "No Ollama models found, falling back to Google Translate...")
            return self._translate_with_google(full_text)

        self._report("translate", 0.25, f"Using Ollama model: {model}")

        word_count = len(full_text.split())
        duration_hint = ""
        if speech_duration > 0:
            wpm = LANGUAGE_WPM.get(self.cfg.target_language, 135)
            target_words = int(speech_duration / 60 * wpm)
            duration_hint = (
                f"IMPORTANT TIMING CONSTRAINT: The original narration is {int(speech_duration)} seconds long "
                f"({word_count} words). Aim for approximately {target_words} {target_name} words. "
                f"Be concise — use shorter phrases where possible without losing meaning. "
            )

        chunks = self._split_text_for_translation(full_text, max_chars=4000)
        translated_parts = []

        for i, chunk in enumerate(chunks):
            prompt = (
                self._get_translation_prompt("system") + "\n\n"
                f"This is a voiceover script for a dubbed video. "
                f"{duration_hint}"
                f"Do NOT translate literally word-by-word. Adapt idioms and phrasing naturally. "
                f"Output ONLY the translation, nothing else.\n\n"
                f"{chunk}"
            )

            try:
                resp = _requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=120,
                )
                data = resp.json()
                translated_parts.append(data.get("response", "").strip())
            except Exception as e:
                self._report("translate", 0.2, f"Ollama failed: {e}, falling back to Google Translate...")
                return self._translate_with_google(full_text)

            self._report("translate", 0.2 + 0.8 * ((i + 1) / len(chunks)),
                         f"Translated chunk {i + 1}/{len(chunks)} (Ollama: {model})")

        return " ".join(translated_parts)

    def _translate_with_google(self, full_text: str) -> str:
        """Fallback: translate using free Google Translate via deep-translator."""
        from deep_translator import GoogleTranslator

        src = self.cfg.source_language if self.cfg.source_language != "auto" else "auto"
        translator = GoogleTranslator(source=src, target=self.cfg.target_language)
        chunks = self._split_text_for_translation(full_text, max_chars=4500)
        translated_parts = []

        for i, chunk in enumerate(chunks):
            retries = 3
            for attempt in range(retries):
                try:
                    translated_parts.append(translator.translate(chunk))
                    break
                except Exception:
                    if attempt < retries - 1:
                        time.sleep(1.5 * (attempt + 1))
                    else:
                        translated_parts.append(chunk)

            self._report("translate", 0.2 + 0.8 * ((i + 1) / len(chunks)),
                         f"Translated chunk {i + 1}/{len(chunks)}")

        return " ".join(translated_parts)

    @staticmethod
    def _split_text_for_translation(text: str, max_chars: int = 4500) -> List[str]:
        """Split text into chunks at sentence boundaries for translation API limits."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        current = ""
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?।])\s+', text)

        for sentence in sentences:
            if len(current) + len(sentence) + 1 > max_chars and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current = (current + " " + sentence).strip()

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [text]

    @staticmethod
    def _compute_target_word_count(duration_seconds: float, target_language: str) -> int:
        """Compute target word count for translation based on language speaking rate."""
        wpm = LANGUAGE_WPM.get(target_language, 135)
        return max(1, round((duration_seconds / 60.0) * wpm))

    def _get_translation_prompt(self, mode: str = "system") -> str:
        """Build the translation system/user prompt based on target language.

        For Hindi: uses a detailed natural-spoken-Hindi prompt.
        For other languages: uses a generic colloquial prompt.
        mode: 'system' for system message, 'user_prefix' for user message intro.
        """
        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        source_name = LANGUAGE_NAMES.get(self.cfg.source_language, "the source language") if self.cfg.source_language != "auto" else "the detected language"
        is_hindi = self.cfg.target_language in ("hi", "hi-IN")

        if mode == "system":
            if is_hindi:
                return (
                    "You are a professional Hindi dubbing writer for YouTube story recap channels.\n"
                    "Your job: rewrite English lines as ORIGINAL Hindi narration — as if the script was "
                    "WRITTEN IN HINDI from the start. It must NEVER feel translated.\n\n"

                    "═══ CORE IDENTITY ═══\n"
                    "You are a Hindi narrator telling a gripping story to an audience. Think of channels "
                    "like 'Anime Jeetu', 'Recap Jeetu', 'Movie Jeetu' — dramatic, engaging, emotional Hindi "
                    "narration that hooks viewers.\n\n"

                    "═══ BANNED WORDS (NEVER USE) ═══\n"
                    "अतः, किन्तु, तथापि, उक्त, यद्यपि, अपितु, एवं, तदनुसार, अभिप्राय, कृपया (in narration), "
                    "आवश्यकता (use ज़रूरत), वास्तव में (use सच में/असल में), विक्षिप्त, घटित, "
                    "उत्साहित (use excited), सामान्य (use normal), तुरंत (use फौरन/झट से), "
                    "अचानक से (just अचानक). "
                    "If you catch yourself writing ANY of these, rewrite the line immediately.\n\n"

                    "═══ NATURAL HINDI RULES ═══\n"
                    "1. REWRITE, don't translate. Think: 'How would I SAY this in Hindi to a friend?'\n"
                    "2. Use तू/तुम for conversations, not आप (unless showing respect to elders intentionally)\n"
                    "3. Use contractions: नहीं→नी, कुछ नहीं→कुछ नी, मत करो→मत कर\n"
                    "4. Dramatic moments need PUNCH: 'और बस... यहीं से खेल शुरू हुआ' not 'और इसी स्थान से...'\n"
                    "5. Dialogue should sound REAL: 'अबे रुक! कहाँ जा रहा है?' not 'कृपया रुकिए'\n"
                    "6. Internal thoughts: 'मैंने सोचा, बस अब इसकी खैर नहीं' not 'मैंने विचार किया'\n"
                    "7. Suspense/tension: 'और तभी... जो हुआ वो किसी ने सोचा भी नहीं था' \n"
                    "8. Keep English words Indians actually use: truck, plan, phone, message, delete, "
                    "hospital, police, driver, accident, cargo, store, impact, vest, glass, bulletproof\n"
                    "9. Hindi words for impact: धड़ाम (crash sound), धम्म (thud), झन्नाटेदार (thrilling), "
                    "सनसनी (sensation), कंपकंपी (shiver), तबाही (destruction)\n"
                    "10. Emotional connectors: और भाई सुनो, अब यहाँ twist आता है, पर wait करो, "
                    "तो basically हुआ ये कि, और जो हुआ अगला वो तो...\n\n"

                    "═══ TRANSLATION FIXES (LEARN THESE) ═══\n"
                    "BAD: 'उसने ठंडी सांस ली' → GOOD: 'उसने फुंकार मारी'\n"
                    "BAD: 'कृपया हमें बचाओ' → GOOD: 'बचाओ! किसी को बुलाओ!'\n"
                    "BAD: 'बुजुर्ग आदमी उत्साहित होकर आया' → GOOD: 'बुड्ढा excited होके आया'\n"
                    "BAD: 'उसने अपना पेट पकड़ लिया और कराह उठा' → GOOD: 'उसने पेट पकड़ा और चिल्लाया'\n"
                    "BAD: 'मैंने उसे पहचान लिया' → GOOD: 'मैंने उसे फौरन पहचान लिया — ये वही था'\n"
                    "BAD: 'एक और शिकार' → GOOD: 'एक और शिकार अपने आप जाल में आ रहा था'\n"
                    "BAD: 'सबको तुरंत डर लग गया' → GOOD: 'सबकी हालत खराब हो गई'\n"
                    "BAD: 'वह चोटों से ढका हुआ था' → GOOD: 'पूरा लहूलुहान था'\n\n"

                    "═══ COMPLETENESS ═══\n"
                    "Translate EVERY word and idea. NOTHING may be skipped or summarized.\n"
                    "Each line must be COMPLETE — no truncation.\n"
                    "Output ONLY numbered translations, one per line, matching input numbering.\n"
                    "Do NOT add notes, comments, Chinese/Japanese characters, or metadata.\n"
                    "Do NOT output anything in any script other than Devanagari and English."
                )
            else:
                return (
                    f"You are a world-class dubbing translator. Make {target_name} sound like it was ORIGINALLY "
                    f"written by a native speaker — a real person talking, not a textbook.\n"
                    f"Sound like a YouTuber, podcast host, or friend explaining something. "
                    f"Use the EXACT way native speakers ACTUALLY talk in everyday life. "
                    f"Keep English words people naturally mix in: 'actually', 'basically', 'problem', 'phone', 'video', 'so', 'but'. "
                    f"NEVER use formal/literary/textbook language. "
                    f"Match the speaker's energy and emotion exactly. "
                    f"Keep proper nouns, brands, and technical terms unchanged.\n\n"
                    f"COMPLETENESS: Translate EVERY word and idea. NOTHING may be skipped. "
                    f"Each translation must be COMPLETE. Video adapts to audio — don't worry about length. "
                    f"Output ONLY numbered translations. No notes or metadata."
                )

        elif mode == "user_prefix":
            if is_hindi:
                return (
                    f"Rewrite each numbered line as ORIGINAL Hindi narration/dialogue. "
                    f"This is a story recap — make it dramatic, gripping, and natural. "
                    f"Every line must sound like it was WRITTEN in Hindi, not translated. "
                    f"Use natural spoken Hindi. Translate COMPLETELY — skip nothing.\n\n"
                )
            else:
                return (
                    f"Translate each line from {source_name} to {target_name}. "
                    f"Natural, conversational, the way real people actually talk — like a friend explaining something. "
                    f"Translate COMPLETELY — every idea, every nuance.\n\n"
                )

        return ""

    # ── Step 4b: Segment-level translation ────────────────────────────────
    def _get_turbo_engines(self):
        """Get available engines for turbo parallel translation."""
        engines = []
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        sambanova_key = os.environ.get("SAMBANOVA_API_KEY", "").strip()
        if groq_key:
            engines.append(("Groq", groq_key))
        if sambanova_key:
            engines.append(("SambaNova", sambanova_key))
        return engines

    def _translate_segments(self, segments):
        """Translate each segment individually, preserving timestamps for sync.

        Respects self.cfg.translation_engine setting.
        Turbo mode: Groq + SambaNova in parallel (both Llama 3.3 70B).
        """
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        sambanova_key = os.environ.get("SAMBANOVA_API_KEY", "").strip()
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        engine = self.cfg.translation_engine

        turbo_engines = self._get_turbo_engines()

        # Specific engine selected
        if engine == "turbo" and len(turbo_engines) >= 2:
            names = " + ".join(e[0] for e in turbo_engines)
            self._report("translate", 0.05, f"TURBO: {names} parallel translation ({len(turbo_engines)} engines)...")
            self._translate_segments_turbo(segments, turbo_engines)
            return
        elif engine == "turbo":
            self._report("translate", 0.05, "Turbo needs 2+ engine keys, using best available...")
        elif engine == "hinglish" and self._ollama_available():
            self._report("translate", 0.05, "Using Hinglish AI (custom Ollama model)...")
            self._translate_segments_ollama(segments, force_model="hinglish-translator")
            return
        elif engine == "sambanova" and sambanova_key:
            self._report("translate", 0.05, "Using SambaNova (Llama 3.3 70B) for translation...")
            self._translate_segments_sambanova(segments, sambanova_key)
            return
        elif engine == "groq" and groq_key:
            self._report("translate", 0.05, "Using Groq (Llama 3.3 70B) for translation...")
            self._translate_segments_groq(segments, groq_key)
            return
        elif engine == "gemini" and gemini_key:
            self._report("translate", 0.05, "Using Gemini for translation...")
            self._translate_segments_gemini(segments, gemini_key)
            return
        elif engine == "ollama" and self._ollama_available():
            self._report("translate", 0.05, "Using Ollama (local LLM) for translation...")
            self._translate_segments_ollama(segments)
            return
        elif engine == "google":
            self._report("translate", 0.1, "Using Google Translate...")
            self._translate_segments_google(segments)
            return

        # Auto mode: turbo if 2+ engines, else best single
        if len(turbo_engines) >= 2:
            names = " + ".join(e[0] for e in turbo_engines)
            self._report("translate", 0.05, f"TURBO: {names} parallel translation ({len(turbo_engines)} engines)...")
            self._translate_segments_turbo(segments, turbo_engines)
        elif openai_key:
            self._report("translate", 0.05, "Using GPT-4o for premium translation...")
            self._translate_segments_openai(segments, openai_key)
        elif groq_key:
            self._report("translate", 0.05, "Using Groq (Llama 3.3 70B) for translation...")
            self._translate_segments_groq(segments, groq_key)
        elif sambanova_key:
            self._report("translate", 0.05, "Using SambaNova (Llama 3.3 70B) for translation...")
            self._translate_segments_sambanova(segments, sambanova_key)
        elif gemini_key:
            self._report("translate", 0.05, "Using Gemini for colloquial translation...")
            self._translate_segments_gemini(segments, gemini_key)
        elif self._ollama_available():
            self._report("translate", 0.05, "Using Ollama (local LLM) for translation...")
            self._translate_segments_ollama(segments)
        else:
            self._report("translate", 0.1, "No API keys found, using Google Translate...")
            self._translate_segments_google(segments)

    def _translate_segments_gemini(self, segments, api_key):
        """Translate segments in numbered batches using Gemini for context-aware output."""
        from google import genai
        client = genai.Client(api_key=api_key)

        batch_size = 30
        total_batches = (len(segments) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batch = segments[start:end]

            lines = [f"{i+1}. {seg['text']}" for i, seg in enumerate(batch)]
            prompt = (
                self._get_translation_prompt("system") + "\n\n"
                + self._get_translation_prompt("user_prefix")
                + "\n".join(lines)
            )

            retries = 3
            success = False
            for attempt in range(retries):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-pro", contents=prompt)
                    translations = self._parse_numbered_translations(response.text, len(batch))
                    for i, seg in enumerate(batch):
                        seg["text_translated"] = translations[i] if translations[i] else seg["text"]
                    success = True
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        wait = 2 * (attempt + 1)
                        self._report("translate", 0.1 + 0.8 * (batch_idx / total_batches),
                                     f"Rate limited, retrying in {wait}s...")
                        time.sleep(wait)

            if not success:
                self._report("translate", 0.1, "Gemini failed, using Google Translate for batch...")
                for seg in batch:
                    seg["text_translated"] = self._translate_single_google(seg["text"])

            self._report("translate", 0.1 + 0.9 * ((batch_idx + 1) / total_batches),
                         f"Translated batch {batch_idx + 1}/{total_batches}")

    def _translate_segments_groq(self, segments, api_key):
        """Translate segments using Groq (Llama 3.3 70B) — fast, free, context-aware."""
        from groq import Groq
        client = Groq(api_key=api_key)

        batch_size = 50  # Larger batches = fewer API calls = faster
        total_batches = (len(segments) + batch_size - 1) // batch_size
        system_msg = self._get_translation_prompt("system")

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batch = segments[start:end]

            lines = [f"{i+1}. {seg['text']}" for i, seg in enumerate(batch)]
            user_msg = self._get_translation_prompt("user_prefix") + "\n".join(lines)

            retries = 3
            success = False
            for attempt in range(retries):
                try:
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.7,
                        max_tokens=8192,
                    )
                    result_text = response.choices[0].message.content
                    translations = self._parse_numbered_translations(result_text, len(batch))
                    for i, seg in enumerate(batch):
                        seg["text_translated"] = translations[i] if translations[i] else seg["text"]
                    success = True
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        wait = 2 * (attempt + 1)
                        self._report("translate", 0.1 + 0.8 * (batch_idx / total_batches),
                                     f"Groq rate limited, retrying in {wait}s...")
                        time.sleep(wait)

            if not success:
                self._report("translate", 0.1, "Groq failed, using Google Translate for batch...")
                for seg in batch:
                    seg["text_translated"] = self._translate_single_google(seg["text"])

            self._report("translate", 0.1 + 0.9 * ((batch_idx + 1) / total_batches),
                         f"Translated batch {batch_idx + 1}/{total_batches} (Groq)")

    def _translate_batch_openai_compat(self, batch, api_url, api_key, model, engine_name):
        """Translate a batch using any OpenAI-compatible API. Returns translations list or None."""
        import requests as _requests
        lines = [f"{i+1}. {seg['text']}" for i, seg in enumerate(batch)]
        system_msg = self._get_translation_prompt("system")
        user_msg = self._get_translation_prompt("user_prefix") + "\n".join(lines)

        for attempt in range(3):
            try:
                resp = _requests.post(
                    api_url,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        "temperature": 0.7,
                        "max_tokens": 8192,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                return self._parse_numbered_translations(
                    resp.json()["choices"][0]["message"]["content"], len(batch))
            except Exception:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
        return None

    # Engine configs for OpenAI-compatible APIs (all using Llama 3.3 70B)
    TURBO_ENGINE_CONFIG = {
        "Groq": ("https://api.groq.com/openai/v1/chat/completions", "llama-3.3-70b-versatile"),
        "SambaNova": ("https://api.sambanova.ai/v1/chat/completions", "Meta-Llama-3.3-70B-Instruct"),
    }

    def _translate_segments_sambanova(self, segments, api_key):
        """Translate segments using SambaNova (Llama 3.3 70B) — fast, free."""
        url, model = self.TURBO_ENGINE_CONFIG["SambaNova"]
        batch_size = 50
        total_batches = (len(segments) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batch = segments[start:end]

            translations = self._translate_batch_openai_compat(batch, url, api_key, model, "SambaNova")
            if translations:
                for i, seg in enumerate(batch):
                    seg["text_translated"] = translations[i] if translations[i] else seg["text"]
            else:
                self._report("translate", 0.1, "SambaNova failed, using Google Translate for batch...")
                for seg in batch:
                    seg["text_translated"] = self._translate_single_google(seg["text"])

            self._report("translate", 0.1 + 0.9 * ((batch_idx + 1) / total_batches),
                         f"Translated batch {batch_idx + 1}/{total_batches} (SambaNova)")

    def _translate_segments_turbo(self, segments, engines):
        """TURBO: Distribute batches across multiple engines in parallel.

        Round-robin batches across all available engines (Groq, SambaNova).
        All run simultaneously via threads = up to 2x faster translation.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        batch_size = 50
        total_batches = (len(segments) + batch_size - 1) // batch_size
        num_engines = len(engines)

        # Build batch list
        batches = []
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batches.append((batch_idx, segments[start:end]))

        # Round-robin batches across engines, run all in parallel
        completed = 0
        with ThreadPoolExecutor(max_workers=num_engines * 2) as pool:
            futures = {}
            for batch_idx, batch in batches:
                engine_name, api_key = engines[batch_idx % num_engines]
                url, model = self.TURBO_ENGINE_CONFIG[engine_name]
                fut = pool.submit(
                    self._translate_batch_openai_compat,
                    batch, url, api_key, model, engine_name)
                futures[fut] = (batch_idx, batch, engine_name)

            for fut in as_completed(futures):
                batch_idx, batch, engine_name = futures[fut]
                translations = fut.result()
                if translations:
                    for i, seg in enumerate(batch):
                        seg["text_translated"] = translations[i] if translations[i] else seg["text"]
                else:
                    self._report("translate", 0.1,
                                 f"{engine_name} failed batch {batch_idx+1}, falling back to Google...")
                    for seg in batch:
                        seg["text_translated"] = self._translate_single_google(seg["text"])

                completed += 1
                self._report("translate", 0.1 + 0.9 * (completed / total_batches),
                             f"TURBO: {completed}/{total_batches} batches ({engine_name})")

    def _translate_segments_ollama(self, segments, force_model: str = ""):
        """Translate segments using local Ollama LLM in batches."""
        import requests as _requests

        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        source_name = LANGUAGE_NAMES.get(self.cfg.source_language, "the source language") if self.cfg.source_language != "auto" else "the detected language"

        # Pick model
        try:
            resp = _requests.get("http://localhost:11434/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            models = []

        model = None
        if force_model:
            model = force_model if any(force_model in m for m in models) else None
        if not model:
            preferred = (
                ["hinglish-translator"] if self.cfg.target_language in ("hi", "hi-IN") else []
            ) + ["qwen2.5:14b", "qwen2.5:32b", "llama3.1:8b", "llama3:8b", "gemma2:9b", "mistral:7b"]
            for pref in preferred:
                for m in models:
                    if pref.split(":")[0] in m:
                        model = m
                        break
                if model:
                    break
            if not model and models:
                model = models[0]
        if not model:
            self._report("translate", 0.1, "No Ollama models found, using Google Translate...")
            self._translate_segments_google(segments)
            return

        self._report("translate", 0.05, f"Using Ollama model: {model}")

        batch_size = 15  # Smaller batches for local LLM
        total_batches = (len(segments) + batch_size - 1) // batch_size

        system_msg = self._get_translation_prompt("system")

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batch = segments[start:end]

            lines = [f"{i+1}. {seg['text']}" for i, seg in enumerate(batch)]
            user_msg = self._get_translation_prompt("user_prefix") + "\n".join(lines)

            try:
                resp = _requests.post("http://localhost:11434/api/chat", json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.7},
                }, timeout=120)
                result_text = resp.json().get("message", {}).get("content", "")
                translations = self._parse_numbered_translations(result_text, len(batch))
                for i, seg in enumerate(batch):
                    seg["text_translated"] = translations[i] if translations[i] else seg["text"]
            except Exception as e:
                self._report("translate", 0.1, f"Ollama failed for batch: {e}, using Google Translate...")
                for seg in batch:
                    seg["text_translated"] = self._translate_single_google(seg["text"])

            self._report("translate", 0.1 + 0.9 * ((batch_idx + 1) / total_batches),
                         f"Translated batch {batch_idx + 1}/{total_batches} (Ollama: {model})")

    def _translate_segments_openai(self, segments, api_key):
        """Translate segments using OpenAI GPT-4o for highest quality."""
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        batch_size = 30
        total_batches = (len(segments) + batch_size - 1) // batch_size
        system_msg = self._get_translation_prompt("system")

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batch = segments[start:end]

            lines = [f"{i+1}. {seg['text']}" for i, seg in enumerate(batch)]
            user_msg = self._get_translation_prompt("user_prefix") + "\n".join(lines)

            retries = 3
            success = False
            for attempt in range(retries):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.3,
                    )
                    text = response.choices[0].message.content or ""
                    translations = self._parse_numbered_translations(text, len(batch))
                    for i, seg in enumerate(batch):
                        seg["text_translated"] = translations[i] if translations[i] else seg["text"]
                    success = True
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        wait = 2 * (attempt + 1)
                        self._report("translate", 0.1 + 0.8 * (batch_idx / total_batches),
                                     f"GPT-4o rate limited, retrying in {wait}s...")
                        time.sleep(wait)

            if not success:
                # Fall back to Groq > Gemini > Google Translate
                groq_key = os.environ.get("GROQ_API_KEY", "").strip()
                gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
                if groq_key:
                    self._report("translate", 0.1, "GPT-4o failed, falling back to Groq...")
                    for seg in batch:
                        seg["text_translated"] = seg.get("text", "")
                    self._translate_segments_groq(batch, groq_key)
                elif gemini_key:
                    self._report("translate", 0.1, "GPT-4o failed, falling back to Gemini...")
                    for seg in batch:
                        seg["text_translated"] = seg.get("text", "")
                    self._translate_segments_gemini(batch, gemini_key)
                else:
                    for seg in batch:
                        seg["text_translated"] = self._translate_single_google(seg["text"])

            self._report("translate", 0.1 + 0.9 * ((batch_idx + 1) / total_batches),
                         f"Translated batch {batch_idx + 1}/{total_batches} (GPT-4o)")

    def _translate_segments_google(self, segments):
        """Fallback: translate each segment using Google Translate."""
        for i, seg in enumerate(segments):
            seg["text_translated"] = self._translate_single_google(seg["text"])
            self._report("translate", 0.1 + 0.9 * ((i + 1) / len(segments)),
                         f"Translated {i + 1}/{len(segments)} segments")

    def _translate_single_google(self, text: str) -> str:
        """Translate a single text with Google Translate."""
        from deep_translator import GoogleTranslator
        try:
            src = self.cfg.source_language if self.cfg.source_language != "auto" else "auto"
            translator = GoogleTranslator(source=src, target=self.cfg.target_language)
            return translator.translate(text) or text
        except Exception:
            return text

    @staticmethod
    def _parse_numbered_translations(text: str, expected_count: int) -> List[str]:
        """Parse numbered translation output from Gemini."""
        lines = text.strip().split("\n")
        translations = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Match: "1. translation" or "1) translation" or "1. [3.2s | 7w -> aim ~4w] translation"
            match = re.match(r'\s*\d+[\.\)]\s*(?:\[[^\]]*\]\s*)?(.*)', line)
            if match:
                trans = match.group(1).strip()
                if trans:
                    translations.append(trans)
        # Pad with empty strings if Gemini returned fewer lines
        while len(translations) < expected_count:
            translations.append("")
        return translations[:expected_count]

    # ── Step 5: Continuous TTS ────────────────────────────────────────────
    def _tts_continuous(self, translated_text: str) -> Path:
        """Synthesize the entire translated narrative as ONE single TTS call."""
        import edge_tts

        out_mp3 = self.cfg.work_dir / "tts_full.mp3"
        out_wav = self.cfg.work_dir / "tts_full.wav"
        voice = self.cfg.tts_voice
        rate = self.cfg.tts_rate

        self._report("synthesize", 0.1, "Generating speech (single voice)...")

        async def synthesize():
            communicate = edge_tts.Communicate(translated_text, voice, rate=rate)
            await communicate.save(str(out_mp3))

        asyncio.run(synthesize())

        if not out_mp3.exists() or out_mp3.stat().st_size == 0:
            raise RuntimeError("TTS synthesis produced no audio")

        self._report("synthesize", 0.8, "Converting to WAV...")

        # Convert to WAV
        subprocess.run(
            [
                self._ffmpeg, "-y", "-i", str(out_mp3),
                "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                str(out_wav),
            ],
            check=True, capture_output=True,
        )
        out_mp3.unlink(missing_ok=True)

        return out_wav

    # ── Step 5b: Time-aligned TTS ─────────────────────────────────────────
    def _tts_time_aligned(self, segments, total_duration, prefix="", progress_base=0.0, progress_span=1.0):
        """Natural-flow TTS: generate at natural speed, place at original timestamps.

        No speed manipulation — speech sounds completely natural.
        If a segment runs longer than its slot, it simply overlaps into the next gap.
        """
        import edge_tts
        voice = self.cfg.tts_voice
        rate = self.cfg.tts_rate

        # Generate all TTS at natural rate
        async def tts_generate():
            for i, seg in enumerate(segments):
                text = seg.get("text_translated", seg.get("text", "")).strip()
                if not text:
                    continue
                mp3 = self.cfg.work_dir / f"{prefix}seg_{i:04d}.mp3"
                try:
                    comm = edge_tts.Communicate(text, voice, rate=rate)
                    await comm.save(str(mp3))
                    seg["_tts_mp3"] = mp3
                except Exception:
                    pass

        asyncio.run(tts_generate())

        # Convert to WAV (no speed changes)
        tts_data = []
        for i, seg in enumerate(segments):
            mp3 = seg.pop("_tts_mp3", None)
            if not mp3 or not mp3.exists():
                continue

            wav = self.cfg.work_dir / f"{prefix}seg_{i:04d}.wav"
            subprocess.run(
                [self._ffmpeg, "-y", "-i", str(mp3),
                 "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                 str(wav)],
                check=True, capture_output=True,
            )
            mp3.unlink(missing_ok=True)

            tts_dur = self._get_duration(wav)
            tts_data.append({
                "start": seg["start"],
                "wav": wav,
                "duration": tts_dur,
            })

        return self._build_timeline(tts_data, total_duration, prefix)

    def _build_timeline(self, tts_data, total_duration, prefix=""):
        """Place TTS segments at their original timestamps on a silent audio track."""
        total_samples = int((total_duration + 0.5) * self.SAMPLE_RATE)
        bytes_per_frame = 2 * self.N_CHANNELS  # 16-bit stereo = 4 bytes
        timeline = bytearray(total_samples * bytes_per_frame)

        for seg in tts_data:
            start_byte = int(seg["start"] * self.SAMPLE_RATE) * bytes_per_frame

            with wave.open(str(seg["wav"]), 'rb') as w:
                # Validate format matches timeline expectations
                if w.getnchannels() != self.N_CHANNELS or w.getsampwidth() != 2:
                    continue  # Skip mismatched format to avoid corrupted audio
                raw = w.readframes(w.getnframes())

            end_byte = min(start_byte + len(raw), len(timeline))
            copy_len = end_byte - start_byte
            if copy_len > 0:
                timeline[start_byte:end_byte] = raw[:copy_len]

        output = self.cfg.work_dir / f"{prefix}tts_aligned.wav"
        with wave.open(str(output), 'wb') as w:
            w.setnchannels(self.N_CHANNELS)
            w.setsampwidth(2)
            w.setframerate(self.SAMPLE_RATE)
            w.writeframes(bytes(timeline))

        return output

    # ── Speed-fit & no-cut assembly ─────────────────────────────────────

    # Speed limits: slow TTS → speed up to max 1.1x, long TTS → speed up to max 1.25x
    SPEED_MIN = 1.0 / 1.1    # 0.909 — slowest we allow (1.1x slower than natural)
    SPEED_MAX = 1.25          # fastest we allow (1.25x faster than natural)

    def _speed_fit_segments(self, tts_data):
        """Speed-adjust each TTS segment to fit its time slot.

        Rules:
        - If TTS is shorter than slot → slow down up to 1.1x (SPEED_MIN atempo)
        - If TTS is longer than slot → speed up up to 1.25x (SPEED_MAX atempo)
        - NEVER cut or truncate audio — all speech must be heard completely
        - If TTS still doesn't fit after max speed-up, let it overflow into the gap

        Audio Priority mode: NO speed adjustment — TTS plays at natural pace always.
        Video will be adjusted to match audio instead.
        """
        # Audio Priority: skip all speed adjustments, let audio play naturally
        if self.cfg.audio_priority:
            return [tts.copy() for tts in tts_data]

        from concurrent.futures import ThreadPoolExecutor

        def fit_one(idx_tts):
            idx, tts = idx_tts
            seg_start = tts["start"]
            seg_end = tts["end"]
            slot_dur = seg_end - seg_start
            tts_dur = tts["duration"]
            tts_wav = tts["wav"]

            if slot_dur < 0.1 or tts_dur < 0.1:
                return (idx, tts.copy())

            ratio = tts_dur / slot_dur  # >1 = TTS longer than slot

            # Close enough — no adjustment needed
            if abs(ratio - 1.0) < 0.05:
                return (idx, tts.copy())

            # Clamp ratio to our speed limits
            clamped_ratio = max(self.SPEED_MIN, min(ratio, self.SPEED_MAX))

            stretched_wav = self.cfg.work_dir / f"speedfit_{idx:04d}.wav"
            self._time_stretch(tts_wav, clamped_ratio, stretched_wav)

            new_dur = tts_dur / clamped_ratio  # duration after speed change
            return (idx, {
                "start": seg_start,
                "end": seg_end,
                "wav": stretched_wav,
                "duration": new_dur,
            })

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(fit_one, enumerate(tts_data)))

        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def _build_timeline_no_cut(self, tts_data, total_duration, prefix=""):
        """Place TTS segments on a timeline WITHOUT cutting any audio.

        If a segment overflows its slot, the full audio is still placed —
        it may overlap into the next gap, but no speech is ever truncated.
        The timeline is extended if necessary to fit all audio.
        """
        # Calculate the required timeline length — may exceed video duration
        max_end = total_duration
        for seg in tts_data:
            seg_audio_end = seg["start"] + seg["duration"]
            if seg_audio_end > max_end:
                max_end = seg_audio_end

        total_samples = int((max_end + 0.5) * self.SAMPLE_RATE)
        bytes_per_frame = 2 * self.N_CHANNELS
        timeline = bytearray(total_samples * bytes_per_frame)

        for seg in tts_data:
            start_byte = int(seg["start"] * self.SAMPLE_RATE) * bytes_per_frame

            with wave.open(str(seg["wav"]), 'rb') as w:
                raw = w.readframes(w.getnframes())

            # Place ALL audio — never truncate
            end_byte = start_byte + len(raw)
            if end_byte > len(timeline):
                # Extend the timeline to fit
                timeline.extend(b'\x00' * (end_byte - len(timeline)))
            timeline[start_byte:end_byte] = raw

        # Trim back to video duration (but only silence, not speech)
        # The muxer will use video duration as the master
        final_samples = int((total_duration + 0.5) * self.SAMPLE_RATE)
        final_bytes = final_samples * bytes_per_frame
        if len(timeline) > final_bytes:
            # Keep all audio up to video end — FFmpeg will naturally end at video length
            timeline = timeline[:final_bytes]

        output = self.cfg.work_dir / f"{prefix}tts_no_cut.wav"
        with wave.open(str(output), 'wb') as w:
            w.setnchannels(self.N_CHANNELS)
            w.setsampwidth(2)
            w.setframerate(self.SAMPLE_RATE)
            w.writeframes(bytes(timeline))

        return output

    # ── Video-adapts-to-audio assembly ──────────────────────────────────
    # Maximum video slowdown: 1.1x (setpts=1.1*PTS makes it 10% slower)
    VIDEO_SLOW_MAX = 1.1

    def _assemble_video_adapts_to_audio(self, video_path, audio_raw, tts_data, total_video_duration):
        """Assemble dubbed video where AUDIO IS MASTER and video adapts.

        For each segment:
        1. TTS audio plays at NATURAL speed — never sped up, slowed, or cut.
        2. If TTS fits within the video slot: use the scene as-is (or slow up to 1.1x).
        3. If TTS is longer than the slot even at 1.1x slowdown: LOOP the scene
           to fill the remaining time so the full audio is heard.
        4. Gaps between speech segments play at normal video speed.

        Result: complete, uniform audio with video stretched to match.
        """
        sections = []
        current_pos = 0.0
        audio_pos = 0.0  # running position in the output audio timeline

        for tts in tts_data:
            seg_start = tts["start"]
            seg_end = tts["end"]
            tts_dur = tts["duration"]
            slot_dur = seg_end - seg_start

            # Gap before this segment — play at normal speed
            if seg_start > current_pos + 0.05:
                gap_dur = seg_start - current_pos
                sections.append({
                    "type": "gap",
                    "video_start": current_pos,
                    "video_end": seg_start,
                    "output_dur": gap_dur,
                })
                audio_pos += gap_dur

            # Speech segment — video adapts to TTS duration
            if tts_dur <= slot_dur:
                # TTS fits or is shorter — slow video slightly to fill
                slow_factor = max(1.0, tts_dur / slot_dur) if slot_dur > 0.1 else 1.0
                sections.append({
                    "type": "speech",
                    "video_start": seg_start,
                    "video_end": seg_end,
                    "pts_factor": slow_factor,
                    "freeze": False,
                    "tts_wav": tts["wav"],
                    "tts_dur": tts_dur,
                    "output_dur": max(tts_dur, slot_dur * slow_factor),
                })
            else:
                # TTS is longer than slot — slow video to 1.1x first
                slowed_slot = slot_dur * self.VIDEO_SLOW_MAX
                if tts_dur <= slowed_slot:
                    # Slowing to 1.1x is enough
                    pts_factor = tts_dur / slot_dur  # between 1.0 and 1.1
                    sections.append({
                        "type": "speech",
                        "video_start": seg_start,
                        "video_end": seg_end,
                        "pts_factor": pts_factor,
                        "freeze": False,
                        "tts_wav": tts["wav"],
                        "tts_dur": tts_dur,
                        "output_dur": tts_dur,
                    })
                else:
                    # Even at 1.1x the TTS doesn't fit — slow to 1.1x AND freeze last frame
                    sections.append({
                        "type": "speech",
                        "video_start": seg_start,
                        "video_end": seg_end,
                        "pts_factor": self.VIDEO_SLOW_MAX,
                        "freeze": True,
                        "freeze_target_dur": tts_dur,
                        "tts_wav": tts["wav"],
                        "tts_dur": tts_dur,
                        "output_dur": tts_dur,
                    })

            audio_pos += sections[-1]["output_dur"]
            current_pos = max(current_pos, seg_end)  # Never move backward for overlapping segments

        # Trailing gap
        if current_pos < total_video_duration - 0.05:
            gap_dur = total_video_duration - current_pos
            sections.append({
                "type": "gap",
                "video_start": current_pos,
                "video_end": total_video_duration,
                "output_dur": gap_dur,
            })
            audio_pos += gap_dur

        # Build video clips for each section
        num_sections = len(sections)
        clip_paths = []
        clip_section_indices = []  # track which section each clip belongs to

        for idx, sec in enumerate(sections):
            self._report("assemble",
                         0.15 + 0.50 * (idx / max(num_sections, 1)),
                         f"Building section {idx + 1}/{num_sections}...")

            clip = self.cfg.work_dir / f"adapt_{idx:04d}.mp4"
            vs = sec["video_start"]
            ve = sec["video_end"]
            dur = ve - vs

            if dur < 0.05:
                continue

            if sec["type"] == "gap":
                # Normal speed gap
                subprocess.run(
                    [self._ffmpeg, "-y",
                     "-ss", f"{vs:.3f}", "-i", str(video_path),
                     "-t", f"{dur:.3f}",
                     "-an",
                     "-c:v", "libx264", "-preset", self.cfg.encode_preset, "-crf", "20",
                     str(clip)],
                    check=True, capture_output=True,
                )
                clip_paths.append(clip)
                clip_section_indices.append(idx)

            elif sec["type"] == "speech":
                pts_factor = sec["pts_factor"]

                if not sec.get("freeze", False):
                    # Slow down video (or keep normal) — no freeze needed
                    if abs(pts_factor - 1.0) < 0.03:
                        # Normal speed
                        subprocess.run(
                            [self._ffmpeg, "-y",
                             "-ss", f"{vs:.3f}", "-i", str(video_path),
                             "-t", f"{dur:.3f}",
                             "-an",
                             "-c:v", "libx264", "-preset", self.cfg.encode_preset, "-crf", "20",
                             str(clip)],
                            check=True, capture_output=True,
                        )
                    else:
                        subprocess.run(
                            [self._ffmpeg, "-y",
                             "-ss", f"{vs:.3f}", "-i", str(video_path),
                             "-t", f"{dur:.3f}",
                             "-filter:v", f"setpts={pts_factor:.6f}*PTS",
                             "-an",
                             "-c:v", "libx264", "-preset", self.cfg.encode_preset, "-crf", "20",
                             str(clip)],
                            check=True, capture_output=True,
                        )
                    clip_paths.append(clip)
                    clip_section_indices.append(idx)

                else:
                    # Freeze: slow the scene to 1.1x, then freeze the last frame
                    target_dur = sec["freeze_target_dur"]

                    # First, create the slowed scene clip
                    slowed_clip = self.cfg.work_dir / f"adapt_{idx:04d}_slow.mp4"
                    subprocess.run(
                        [self._ffmpeg, "-y",
                         "-ss", f"{vs:.3f}", "-i", str(video_path),
                         "-t", f"{dur:.3f}",
                         "-filter:v", f"setpts={self.VIDEO_SLOW_MAX:.6f}*PTS",
                         "-an",
                         "-c:v", "libx264", "-preset", self.cfg.encode_preset, "-crf", "20",
                         str(slowed_clip)],
                        check=True, capture_output=True,
                    )
                    slowed_dur = dur * self.VIDEO_SLOW_MAX

                    if slowed_dur >= target_dur:
                        # Slowed clip is long enough (shouldn't normally reach here)
                        clip_paths.append(slowed_clip)
                        clip_section_indices.append(idx)
                    else:
                        # Freeze the last frame to fill remaining time
                        freeze_dur = target_dur - slowed_dur
                        # Extract the last frame from the slowed clip
                        last_frame = self.cfg.work_dir / f"adapt_{idx:04d}_lastframe.png"
                        # Try -sseof first; if clip is too short, fall back to extracting first frame
                        frame_ok = subprocess.run(
                            [self._ffmpeg, "-y",
                             "-sseof", "-0.1", "-i", str(slowed_clip),
                             "-frames:v", "1",
                             "-update", "1",
                             str(last_frame)],
                            capture_output=True,
                        )
                        if frame_ok.returncode != 0 or not last_frame.exists():
                            # Fallback: extract first frame instead
                            subprocess.run(
                                [self._ffmpeg, "-y",
                                 "-i", str(slowed_clip),
                                 "-frames:v", "1",
                                 "-update", "1",
                                 str(last_frame)],
                                capture_output=True,
                            )
                        if not last_frame.exists():
                            # If we still can't get a frame, just use the slowed clip as-is
                            clip_paths.append(slowed_clip)
                            clip_section_indices.append(idx)
                            continue
                        # Create a still video from the last frame
                        freeze_clip = self.cfg.work_dir / f"adapt_{idx:04d}_freeze.mp4"
                        subprocess.run(
                            [self._ffmpeg, "-y",
                             "-loop", "1", "-i", str(last_frame),
                             "-t", f"{freeze_dur:.3f}",
                             "-vf", "fps=30",
                             "-c:v", "libx264", "-preset", self.cfg.encode_preset, "-crf", "20",
                             "-pix_fmt", "yuv420p",
                             str(freeze_clip)],
                            check=True, capture_output=True,
                        )
                        # Concatenate slowed clip + frozen frame clip
                        concat_list = self.cfg.work_dir / f"adapt_{idx:04d}_concat.txt"
                        concat_list.write_text(
                            f"file '{str(slowed_clip).replace(chr(92), '/')}'\n"
                            f"file '{str(freeze_clip).replace(chr(92), '/')}'\n"
                        )
                        subprocess.run(
                            [self._ffmpeg, "-y",
                             "-f", "concat", "-safe", "0",
                             "-i", str(concat_list),
                             "-c:v", "libx264", "-preset", self.cfg.encode_preset, "-crf", "20",
                             str(clip)],
                            check=True, capture_output=True,
                        )
                        clip_paths.append(clip)
                        clip_section_indices.append(idx)

        if not clip_paths:
            raise RuntimeError("No video sections produced")

        # Concatenate all video clips
        self._report("assemble", 0.70, "Joining video sections...")
        adapted_video = self.cfg.work_dir / "video_adapted.mp4"
        if len(clip_paths) == 1:
            shutil.copy2(clip_paths[0], adapted_video)
        else:
            self._concatenate_videos(clip_paths, adapted_video)

        # Probe actual durations of each video clip to avoid cumulative drift
        self._report("assemble", 0.72, "Measuring actual clip durations...")
        actual_durations = {}  # section_index -> actual duration
        for clip_path, sec_idx in zip(clip_paths, clip_section_indices):
            actual_dur = self._get_duration(clip_path)
            if actual_dur > 0:
                actual_durations[sec_idx] = actual_dur

        # Build the TTS audio timeline using ACTUAL video clip durations (not theoretical)
        self._report("assemble", 0.80, "Building audio timeline...")
        audio_timeline_pos = 0.0
        audio_segments = []
        produced_sections = set(clip_section_indices)
        for idx, sec in enumerate(sections):
            if idx not in produced_sections:
                continue  # skip sections that produced no video clip (e.g. dur < 0.05)
            # Use actual probed duration if available, otherwise fallback to theoretical
            real_dur = actual_durations.get(idx, sec["output_dur"])
            if sec["type"] == "speech":
                audio_segments.append({
                    "start": audio_timeline_pos,
                    "wav": sec["tts_wav"],
                    "duration": sec["tts_dur"],
                })
            audio_timeline_pos += real_dur

        total_output_dur = audio_timeline_pos
        tts_audio = self._build_timeline(audio_segments, total_output_dur, prefix="adapted_")

        # Mix original audio at low volume if requested
        if self.cfg.mix_original:
            self._report("assemble", 0.85, "Mixing original audio...")
            tts_audio = self._mix_audio(audio_raw, tts_audio, self.cfg.original_volume)

        # Mux adapted video + natural TTS audio
        self._report("assemble", 0.90, "Muxing final video...")
        self._mux_replace_audio(adapted_video, tts_audio, self.cfg.output_path)

    # ── Natural TTS + Video sync ────────────────────────────────────────
    # Languages that Chatterbox TTS can pronounce well (English only for now)
    CHATTERBOX_SUPPORTED_LANGS = {"en"}

    def _generate_tts_natural(self, segments):
        """Generate TTS at natural speed. Uses first enabled engine in priority order.

        Hybrid mode: If both Coqui XTTS + Edge-TTS are enabled, splits segments
        between GPU (Coqui) and cloud (Edge-TTS) in parallel for ~2x speed.

        Chatterbox is English-only — for other languages it auto-falls back to
        ElevenLabs (multilingual) or Edge-TTS (which has native voices for 70+ languages).
        """
        target = self.cfg.target_language

        if self.cfg.use_chatterbox:
            if target in self.CHATTERBOX_SUPPORTED_LANGS:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        raise RuntimeError("No CUDA GPU available")
                    self._report("synthesize", 0.05, "Using Chatterbox TTS (GPU, human-like voice)...")
                    return self._tts_chatterbox(segments)
                except Exception as e:
                    self._report("synthesize", 0.05,
                                 f"Chatterbox failed ({e}) — falling back to Edge-TTS...")
            else:
                target_name = LANGUAGE_NAMES.get(target, target)
                self._report("synthesize", 0.05,
                             f"Chatterbox is English-only — trying next engine for {target_name}...")

        if self.cfg.use_elevenlabs:
            elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
            if elevenlabs_key:
                self._report("synthesize", 0.05, "Using ElevenLabs for human-like voice...")
                return self._tts_elevenlabs(segments, elevenlabs_key)
            if not self.cfg.use_chatterbox:
                raise RuntimeError("ElevenLabs enabled but ELEVENLABS_API_KEY not set in .env")

        # HYBRID MODE: Coqui XTTS (GPU) + Edge-TTS (cloud) in parallel
        if self.cfg.use_coqui_xtts and self.cfg.use_edge_tts:
            try:
                import torch
                if torch.cuda.is_available():
                    self._report("synthesize", 0.05,
                                 "HYBRID: Coqui XTTS (GPU) + Edge-TTS (cloud) in parallel...")
                    return self._tts_hybrid_coqui_edge(segments)
            except ImportError:
                pass
            except Exception as e:
                self._report("synthesize", 0.05,
                             f"Hybrid TTS failed ({e}) — trying single engine...")

        if self.cfg.use_coqui_xtts:
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("No CUDA GPU available for XTTS")
                self._report("synthesize", 0.05, "Using Coqui XTTS v2 (GPU, voice cloning)...")
                return self._tts_coqui_xtts(segments)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._report("synthesize", 0.05,
                             f"Coqui XTTS failed ({e}) — falling back...")

        if self.cfg.use_google_tts:
            google_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
            if google_creds or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip():
                self._report("synthesize", 0.05, "Using Google Cloud TTS (WaveNet/Neural2)...")
                try:
                    return self._tts_google_cloud(segments)
                except Exception as e:
                    self._report("synthesize", 0.05,
                                 f"Google Cloud TTS failed ({e}) — falling back to Edge-TTS...")
            else:
                self._report("synthesize", 0.05,
                             "Google Cloud TTS enabled but no credentials found — falling back to Edge-TTS...")

        # Fall through to Edge-TTS (supports 70+ languages with native voices)
        voice = self.cfg.tts_voice
        target_name = LANGUAGE_NAMES.get(target, target)
        if self._voice_map:
            voices_used = len(set(self._voice_map.values()))
            self._report("synthesize", 0.05,
                         f"Using Edge-TTS with {voices_used} distinct voices for {target_name}...")
        else:
            self._report("synthesize", 0.05, f"Using Edge-TTS ({voice}) for {target_name}...")
        return self._tts_edge(segments, voice_map=self._voice_map)

    def _tts_hybrid_coqui_edge(self, segments):
        """HYBRID: Split segments between Coqui XTTS (GPU) and Edge-TTS (cloud).

        Coqui runs on GPU (sequential but high quality voice cloning).
        Edge-TTS runs on cloud (highly parallel, native voices).
        Both run simultaneously in separate threads = ~2x faster.
        """
        import threading

        # Split: even-indexed segments → Coqui XTTS, odd-indexed → Edge-TTS
        coqui_segments = [(i, seg) for i, seg in enumerate(segments) if i % 2 == 0]
        edge_segments = [(i, seg) for i, seg in enumerate(segments) if i % 2 == 1]

        coqui_results = {}
        edge_results = {}
        errors = []

        # Separate work dirs to prevent filename collisions
        coqui_dir = self.cfg.work_dir / "hybrid_coqui"
        edge_dir = self.cfg.work_dir / "hybrid_edge"
        coqui_dir.mkdir(exist_ok=True)
        edge_dir.mkdir(exist_ok=True)

        self._report("synthesize", 0.05,
                     f"Hybrid: {len(coqui_segments)} segs → Coqui GPU, {len(edge_segments)} segs → Edge-TTS cloud")

        def _match_results_to_segments(results, indexed_segments, out_dict):
            """Match TTS results back to original segment indices by start time.
            TTS functions may skip segments (empty text, errors), so positional
            zip would misalign results. Match by start time instead."""
            for result in results:
                for orig_idx, seg in indexed_segments:
                    if orig_idx not in out_dict and abs(result["start"] - seg["start"]) < 0.01:
                        out_dict[orig_idx] = result
                        break

        def run_coqui():
            try:
                coqui_segs_only = [seg for _, seg in coqui_segments]
                results = self._tts_coqui_xtts(coqui_segs_only, work_dir=coqui_dir)
                _match_results_to_segments(results, coqui_segments, coqui_results)
            except Exception as e:
                errors.append(f"Coqui: {e}")

        def run_edge():
            try:
                edge_segs_only = [seg for _, seg in edge_segments]
                results = self._tts_edge(edge_segs_only, voice_map=self._voice_map, work_dir=edge_dir)
                _match_results_to_segments(results, edge_segments, edge_results)
            except Exception as e:
                errors.append(f"Edge: {e}")

        # Launch both in parallel
        t_coqui = threading.Thread(target=run_coqui, name="tts-coqui")
        t_edge = threading.Thread(target=run_edge, name="tts-edge")
        t_coqui.start()
        t_edge.start()
        t_coqui.join()
        t_edge.join()

        # If Coqui failed, Edge may have succeeded for its half
        if errors:
            self._report("synthesize", 0.85,
                         f"Hybrid partial errors: {'; '.join(errors)}")

        # Merge results in original order
        all_results = {**coqui_results, **edge_results}

        # If Coqui failed entirely, re-generate its segments via Edge-TTS
        if not coqui_results and coqui_segments:
            self._report("synthesize", 0.85,
                         "Coqui failed — re-generating its segments via Edge-TTS...")
            coqui_segs_only = [seg for _, seg in coqui_segments]
            fallback_results = self._tts_edge(coqui_segs_only, voice_map=self._voice_map, work_dir=coqui_dir)
            _match_results_to_segments(fallback_results, coqui_segments, all_results)

        # If Edge-TTS failed entirely, re-generate its segments via Edge-TTS retry
        if not edge_results and edge_segments:
            self._report("synthesize", 0.87,
                         "Edge-TTS failed — retrying its segments...")
            edge_segs_only = [seg for _, seg in edge_segments]
            fallback_results = self._tts_edge(edge_segs_only, voice_map=self._voice_map, work_dir=edge_dir)
            _match_results_to_segments(fallback_results, edge_segments, all_results)

        # Sort by original segment index
        tts_data = [all_results[i] for i in sorted(all_results.keys())]
        self._report("synthesize", 0.9,
                     f"Hybrid TTS complete: {len(tts_data)} segments")
        return tts_data

    def _tts_chatterbox(self, segments):
        """Generate TTS using Chatterbox — free, local, human-like AI voice."""
        import torch
        import torchaudio
        from chatterbox.tts import ChatterboxTTS

        self._report("synthesize", 0.05, "Loading Chatterbox model on GPU...")
        model = ChatterboxTTS.from_pretrained(device="cuda")

        tts_data = []
        try:
            for i, seg in enumerate(segments):
                text = seg.get("text_translated", seg.get("text", "")).strip()
                if not text:
                    continue

                wav_path = self.cfg.work_dir / f"tts_{i:04d}.wav"

                try:
                    wav_tensor = model.generate(text)
                    # Save as WAV — Chatterbox outputs at 24kHz
                    torchaudio.save(str(wav_path), wav_tensor.cpu(), model.sr)

                    # Resample to our pipeline's sample rate
                    if model.sr != self.SAMPLE_RATE:
                        resampled = self.cfg.work_dir / f"tts_{i:04d}_rs.wav"
                        subprocess.run(
                            [self._ffmpeg, "-y", "-i", str(wav_path),
                             "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                             str(resampled)],
                            check=True, capture_output=True,
                        )
                        wav_path.unlink(missing_ok=True)
                        resampled.rename(wav_path)

                except Exception as e:
                    self._report("synthesize", 0.1 + 0.8 * ((i + 1) / len(segments)),
                                 f"Chatterbox error on seg {i+1}: {e}, skipping...")
                    continue

                if not wav_path.exists() or wav_path.stat().st_size == 0:
                    continue

                tts_dur = self._get_duration(wav_path)
                tts_data.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "wav": wav_path,
                    "duration": tts_dur,
                })
                self._report(
                    "synthesize",
                    0.1 + 0.8 * ((i + 1) / len(segments)),
                    f"Synthesized {i + 1}/{len(segments)} segments (Chatterbox)...",
                )
        finally:
            del model
            torch.cuda.empty_cache()

        return tts_data

    def _tts_elevenlabs(self, segments, api_key):
        """Generate TTS using ElevenLabs — paid API, most human-like."""
        from elevenlabs import ElevenLabs

        client = ElevenLabs(api_key=api_key)

        voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
        if not voice_id:
            voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel — clear, natural

        model_id = "eleven_multilingual_v2"

        tts_data = []
        for i, seg in enumerate(segments):
            text = seg.get("text_translated", seg.get("text", "")).strip()
            if not text:
                continue

            mp3 = self.cfg.work_dir / f"tts_{i:04d}.mp3"

            try:
                audio_gen = client.text_to_speech.convert(
                    text=text,
                    voice_id=voice_id,
                    model_id=model_id,
                    output_format="mp3_44100_128",
                )
                with open(mp3, "wb") as f:
                    for chunk in audio_gen:
                        f.write(chunk)
            except Exception:
                try:
                    asyncio.run(self._edge_tts_single(text, mp3))
                except Exception:
                    continue

            if not mp3.exists() or mp3.stat().st_size == 0:
                continue

            wav = self.cfg.work_dir / f"tts_{i:04d}.wav"
            subprocess.run(
                [self._ffmpeg, "-y", "-i", str(mp3),
                 "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                 str(wav)],
                check=True, capture_output=True,
            )
            mp3.unlink(missing_ok=True)
            tts_dur = self._get_duration(wav)
            tts_data.append({
                "start": seg["start"],
                "end": seg["end"],
                "wav": wav,
                "duration": tts_dur,
            })
            self._report(
                "synthesize",
                0.1 + 0.8 * ((i + 1) / len(segments)),
                f"Synthesized {i + 1}/{len(segments)} segments (ElevenLabs)...",
            )

        return tts_data

    def _tts_google_cloud(self, segments):
        """Generate TTS using Google Cloud Text-to-Speech (WaveNet/Neural2).

        Free tier: 1M characters/month (WaveNet) or 1M chars (Neural2).
        Requires GOOGLE_APPLICATION_CREDENTIALS env var pointing to service account JSON,
        or running on a Google Cloud instance with default credentials.
        """
        from google.cloud import texttospeech

        client = texttospeech.TextToSpeechClient()
        target = self.cfg.target_language

        # Map target language to Google Cloud voice config
        GOOGLE_VOICE_MAP = {
            "hi": ("hi-IN", "hi-IN-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "bn": ("bn-IN", "bn-IN-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "ta": ("ta-IN", "ta-IN-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "te": ("te-IN", "te-IN-Standard-A", texttospeech.SsmlVoiceGender.FEMALE),
            "mr": ("mr-IN", "mr-IN-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "gu": ("gu-IN", "gu-IN-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "kn": ("kn-IN", "kn-IN-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "ml": ("ml-IN", "ml-IN-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "pa": ("pa-IN", "pa-IN-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "en": ("en-US", "en-US-Neural2-F", texttospeech.SsmlVoiceGender.FEMALE),
            "es": ("es-ES", "es-ES-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "fr": ("fr-FR", "fr-FR-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "de": ("de-DE", "de-DE-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "ja": ("ja-JP", "ja-JP-Neural2-B", texttospeech.SsmlVoiceGender.FEMALE),
            "ko": ("ko-KR", "ko-KR-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "zh": ("cmn-CN", "cmn-CN-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "ar": ("ar-XA", "ar-XA-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "pt": ("pt-BR", "pt-BR-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "ru": ("ru-RU", "ru-RU-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
            "it": ("it-IT", "it-IT-Neural2-A", texttospeech.SsmlVoiceGender.FEMALE),
        }

        lang_code, voice_name, gender = GOOGLE_VOICE_MAP.get(
            target, (f"{target}-IN", None, texttospeech.SsmlVoiceGender.FEMALE)
        )

        voice_params = texttospeech.VoiceSelectionParams(
            language_code=lang_code,
            name=voice_name,
            ssml_gender=gender,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.SAMPLE_RATE,
        )

        tts_data = []
        for i, seg in enumerate(segments):
            text = seg.get("text_translated", seg.get("text", "")).strip()
            if not text:
                continue

            wav_path = self.cfg.work_dir / f"tts_{i:04d}.wav"

            try:
                synthesis_input = texttospeech.SynthesisInput(text=text)
                response = client.synthesize_speech(
                    input=synthesis_input, voice=voice_params, audio_config=audio_config
                )
                with open(wav_path, "wb") as f:
                    f.write(response.audio_content)

                # Ensure correct format (stereo, target sample rate)
                wav_fixed = self.cfg.work_dir / f"tts_{i:04d}_fixed.wav"
                subprocess.run(
                    [self._ffmpeg, "-y", "-i", str(wav_path),
                     "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                     str(wav_fixed)],
                    check=True, capture_output=True,
                )
                shutil.move(str(wav_fixed), str(wav_path))

            except Exception as e:
                self._report("synthesize", 0.1 + 0.8 * ((i + 1) / len(segments)),
                             f"Google TTS error on seg {i+1}: {e}, falling back to Edge-TTS...")
                mp3 = self.cfg.work_dir / f"tts_{i:04d}.mp3"
                try:
                    asyncio.run(self._edge_tts_single(text, mp3))
                    subprocess.run(
                        [self._ffmpeg, "-y", "-i", str(mp3),
                         "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                         str(wav_path)],
                        check=True, capture_output=True,
                    )
                    mp3.unlink(missing_ok=True)
                except Exception:
                    continue

            if not wav_path.exists() or wav_path.stat().st_size == 0:
                continue

            tts_dur = self._get_duration(wav_path)
            tts_data.append({
                "start": seg["start"],
                "end": seg["end"],
                "wav": wav_path,
                "duration": tts_dur,
            })
            self._report(
                "synthesize",
                0.1 + 0.8 * ((i + 1) / len(segments)),
                f"Synthesized {i + 1}/{len(segments)} segments (Google Cloud TTS)...",
            )

        return tts_data

    def _tts_coqui_xtts(self, segments, work_dir=None):
        """Generate TTS using Coqui XTTS v2 — open-source voice cloning.

        Can clone the original speaker's voice and speak in Hindi/Bengali/etc.
        Requires GPU. Uses the extracted audio as voice reference.
        """
        if work_dir is None:
            work_dir = self.cfg.work_dir
        import torch
        import torchaudio
        # Accept Coqui TOS automatically (non-interactive server environment)
        os.environ["COQUI_TOS_AGREED"] = "1"
        # Fix Windows encoding: Coqui TTS prints non-ASCII text to stdout/stderr
        os.environ["PYTHONIOENCODING"] = "utf-8"
        import sys, io
        _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
        try:
            if hasattr(sys.stdout, 'buffer'):
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
            if hasattr(sys.stderr, 'buffer'):
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except Exception:
            pass  # If wrapping fails, continue with originals
        # PyTorch 2.6+ defaults weights_only=True which breaks Coqui's model loading
        # Use a lock to prevent concurrent pipelines from racing on the global monkey-patch
        import threading
        if not hasattr(Pipeline, '_torch_load_lock'):
            Pipeline._torch_load_lock = threading.Lock()

        from TTS.api import TTS

        self._report("synthesize", 0.02, "Loading XTTS v2 model on GPU (this may take a minute)...")
        tts_data = []
        tts_model = None
        try:
            with Pipeline._torch_load_lock:
                _original_torch_load = torch.load
                torch.load = lambda *args, **kwargs: _original_torch_load(
                    *args, **{**kwargs, "weights_only": False}
                )
                try:
                    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
                finally:
                    torch.load = _original_torch_load

            # Build per-speaker reference map if available
            refs_dir = self.cfg.work_dir / "speaker_refs"
            speaker_ref_map: Dict[str, Path] = {}
            if refs_dir.exists():
                for ref_file in refs_dir.glob("SPEAKER_*.wav"):
                    spk_id = ref_file.stem  # e.g. "SPEAKER_00"
                    speaker_ref_map[spk_id] = ref_file
                if speaker_ref_map:
                    self._report("synthesize", 0.04,
                                 f"Found {len(speaker_ref_map)} per-speaker voice references for cloning")

            # Default reference: single voice from original audio
            ref_wav = self.cfg.work_dir / "voice_ref.wav"
            audio_raw = self.cfg.work_dir / "audio_raw.wav"
            if not ref_wav.exists() and audio_raw.exists():
                subprocess.run(
                    [self._ffmpeg, "-y", "-i", str(audio_raw),
                     "-t", "15", "-ar", "22050", "-ac", "1",
                     str(ref_wav)],
                    check=True, capture_output=True,
                )

            # Check if we also have a user-provided reference in voices/
            user_ref = Path(__file__).resolve().parent / "voices" / "my_voice_refs"
            if user_ref.exists():
                ref_files = list(user_ref.glob("*.wav")) + list(user_ref.glob("*.mp3"))
                if ref_files:
                    ref_wav = ref_files[0]
                    self._report("synthesize", 0.05, f"Using custom voice reference: {ref_wav.name}")

            if not ref_wav.exists() and not speaker_ref_map:
                raise RuntimeError("No voice reference available for XTTS voice cloning")

            # Map language codes for XTTS
            XTTS_LANG_MAP = {
                "hi": "hi", "bn": "bn", "ta": "ta", "te": "te",
                "en": "en", "es": "es", "fr": "fr", "de": "de",
                "it": "it", "pt": "pt", "pl": "pl", "tr": "tr",
                "ru": "ru", "nl": "nl", "cs": "cs", "ar": "ar",
                "zh": "zh-cn", "ja": "ja", "ko": "ko", "hu": "hu",
            }
            xtts_lang = XTTS_LANG_MAP.get(self.cfg.target_language, self.cfg.target_language)
            for i, seg in enumerate(segments):
                text = seg.get("text_translated", seg.get("text", "")).strip()
                if not text:
                    continue

                wav_path = work_dir / f"tts_{i:04d}.wav"

                # Pick per-speaker reference if available, else default
                spk_id = seg.get("speaker_id", "")
                seg_ref = speaker_ref_map.get(spk_id, ref_wav) if speaker_ref_map else ref_wav

                try:
                    tts_model.tts_to_file(
                        text=text,
                        speaker_wav=str(seg_ref),
                        language=xtts_lang,
                        file_path=str(wav_path),
                    )

                    # Convert to target sample rate and stereo
                    wav_fixed = work_dir / f"tts_{i:04d}_fixed.wav"
                    subprocess.run(
                        [self._ffmpeg, "-y", "-i", str(wav_path),
                         "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                         str(wav_fixed)],
                        check=True, capture_output=True,
                    )
                    shutil.move(str(wav_fixed), str(wav_path))

                except Exception as e:
                    self._report("synthesize", 0.1 + 0.8 * ((i + 1) / len(segments)),
                                 f"XTTS error on seg {i+1}: {e}, skipping...")
                    continue

                if not wav_path.exists() or wav_path.stat().st_size == 0:
                    continue

                tts_dur = self._get_duration(wav_path)
                tts_data.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "wav": wav_path,
                    "duration": tts_dur,
                })
                self._report(
                    "synthesize",
                    0.1 + 0.8 * ((i + 1) / len(segments)),
                    f"Synthesized {i + 1}/{len(segments)} segments (XTTS v2 voice clone)...",
                )
        finally:
            if tts_model is not None:
                del tts_model
                torch.cuda.empty_cache()
            # Detach wrappers before restoring to prevent GC from closing shared buffer
            if sys.stdout is not _orig_stdout:
                try:
                    sys.stdout.detach()
                except Exception:
                    pass
            if sys.stderr is not _orig_stderr:
                try:
                    sys.stderr.detach()
                except Exception:
                    pass
            sys.stdout, sys.stderr = _orig_stdout, _orig_stderr

        return tts_data

    async def _edge_tts_single(self, text, mp3_path):
        """Generate a single segment with edge-tts."""
        import edge_tts
        comm = edge_tts.Communicate(text, self.cfg.tts_voice, rate=self.cfg.tts_rate)
        await comm.save(str(mp3_path))

    def _tts_edge(self, segments, voice_map=None, work_dir=None):
        """Generate TTS using edge-tts (free Microsoft voices).
        If voice_map is provided, each segment uses its speaker's assigned voice.
        Uses parallel processing for speed — 20 segments at once.
        """
        if work_dir is None:
            work_dir = self.cfg.work_dir
        import edge_tts
        default_voice = self.cfg.tts_voice
        rate = self.cfg.tts_rate
        concurrency = 80  # Process 80 segments simultaneously

        async def generate_one(i, seg, semaphore):
            text = seg.get("text_translated", seg.get("text", "")).strip()
            if not text:
                return
            seg_voice = default_voice
            if voice_map and "speaker_id" in seg:
                seg_voice = voice_map.get(seg["speaker_id"], default_voice)
            mp3 = work_dir / f"tts_{i:04d}.mp3"
            async with semaphore:
                try:
                    comm = edge_tts.Communicate(text, seg_voice, rate=rate)
                    await comm.save(str(mp3))
                    seg["_tts_mp3"] = mp3
                except Exception:
                    pass

        async def generate_all():
            semaphore = asyncio.Semaphore(concurrency)
            total = len(segments)
            tasks = []
            for i, seg in enumerate(segments):
                tasks.append(generate_one(i, seg, semaphore))
            # Process in batches and report progress
            batch_size = concurrency
            for batch_start in range(0, len(tasks), batch_size):
                batch = tasks[batch_start:batch_start + batch_size]
                await asyncio.gather(*batch)
                done = min(batch_start + batch_size, total)
                self._report(
                    "synthesize",
                    0.1 + 0.8 * (done / total),
                    f"Synthesized {done}/{total} segments (parallel x{concurrency})...",
                )

        asyncio.run(generate_all())

        # Parallel MP3→WAV + 1.25x speedup in ONE ffmpeg call (eliminates separate speedup step)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        TTS_SPEED = 1.25

        def convert_and_speed(i, seg):
            mp3 = seg.pop("_tts_mp3", None)
            if not mp3 or not mp3.exists():
                return None
            wav = work_dir / f"tts_{i:04d}.wav"
            # Single ffmpeg: MP3→WAV + resample + 1.25x speed in one pass
            subprocess.run(
                [self._ffmpeg, "-y", "-i", str(mp3),
                 "-af", f"atempo={TTS_SPEED}",
                 "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                 str(wav)],
                check=True, capture_output=True,
            )
            mp3.unlink(missing_ok=True)
            tts_dur = self._get_duration(wav)
            return {
                "start": seg["start"],
                "end": seg["end"],
                "wav": wav,
                "duration": tts_dur,
                "_order": i,
                "_already_sped": True,
            }

        tts_data = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(convert_and_speed, i, seg): i for i, seg in enumerate(segments)}
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    tts_data.append(result)

        # Sort back to original order and strip internal key
        tts_data.sort(key=lambda x: x.get("_order", 0))
        for t in tts_data:
            t.pop("_order", None)
        return tts_data

    def _build_fitted_audio(self, video_path, audio_raw, tts_data, total_video_duration):
        """Keep video at original speed, speed-adjust each TTS clip (1.1x–1.25x).

        Rules:
        - Slow TTS (shorter than slot) → slow down up to 1.1x
        - Fast TTS (longer than slot) → speed up up to 1.25x
        - NEVER cut or truncate any audio — all speech must be heard
        """
        num_segs = len(tts_data)
        fitted_segments = []

        for idx, tts in enumerate(tts_data):
            self._report("assemble",
                         0.05 + 0.55 * (idx / max(num_segs, 1)),
                         f"Fitting segment {idx + 1}/{num_segs} to scene...")

            seg_start = tts["start"]
            seg_end = tts["end"]
            original_dur = seg_end - seg_start
            tts_dur = tts["duration"]
            tts_wav = tts["wav"]

            if original_dur < 0.1 or tts_dur < 0.1:
                fitted_segments.append({
                    "start": seg_start,
                    "wav": tts_wav,
                    "duration": tts_dur,
                })
                continue

            ratio = tts_dur / original_dur  # > 1 = TTS is longer, need to speed up

            if abs(ratio - 1.0) < 0.05:
                fitted_segments.append({
                    "start": seg_start,
                    "wav": tts_wav,
                    "duration": tts_dur,
                })
                continue

            # Clamp ratio: slow down max 1.1x, speed up max 1.25x
            clamped_ratio = max(self.SPEED_MIN, min(ratio, self.SPEED_MAX))

            stretched_wav = self.cfg.work_dir / f"fitted_{idx:04d}.wav"
            self._time_stretch(tts_wav, clamped_ratio, stretched_wav)

            new_dur = tts_dur / clamped_ratio
            fitted_segments.append({
                "start": seg_start,
                "wav": stretched_wav,
                "duration": new_dur,
            })

        # Build audio timeline — no cutting, all speech preserved
        self._report("assemble", 0.65, "Building audio timeline (no cuts)...")
        fitted_audio = self._build_timeline_no_cut(fitted_segments, total_video_duration, prefix="fitted_")

        # Mix original audio at low volume if requested
        if self.cfg.mix_original:
            fitted_audio = self._mix_audio(audio_raw, fitted_audio, self.cfg.original_volume)

        # Mux: original video (untouched) + fitted TTS audio
        self._report("assemble", 0.85, "Muxing final video...")
        self._mux_replace_audio(video_path, fitted_audio, self.cfg.output_path)

    def _build_video_synced(self, video_path, audio_raw, tts_data, total_video_duration):
        """Adjust video speed per-segment to match natural TTS duration.

        Instead of changing audio speed, adjust video speed:
        - TTS longer than original → slow video down (setpts > 1)
        - TTS shorter than original → speed video up (setpts < 1)
        - Gaps between speech play at normal speed
        This keeps TTS voices sounding natural.
        """
        # Build sections: alternating gaps and speech segments
        sections = []
        current_pos = 0.0

        for tts in tts_data:
            seg_start = tts["start"]
            seg_end = tts["end"]
            tts_dur = tts["duration"]
            original_dur = seg_end - seg_start

            # Gap before this segment
            if seg_start > current_pos + 0.05:
                sections.append({
                    "type": "gap",
                    "video_start": current_pos,
                    "video_end": seg_start,
                })

            # Speech segment — compute video speed factor
            # setpts factor: tts_dur / original_dur
            # > 1 = slow video down (TTS is longer), < 1 = speed video up (TTS is shorter)
            pts_factor = (tts_dur / original_dur) if original_dur > 0.1 else 1.0
            # Clamp to 1.1x slow / 1.25x fast limits
            pts_factor = max(self.SPEED_MIN, min(pts_factor, self.SPEED_MAX))

            sections.append({
                "type": "speech",
                "video_start": seg_start,
                "video_end": seg_end,
                "pts_factor": pts_factor,
                "tts_wav": tts["wav"],
                "tts_dur": tts_dur,
            })

            current_pos = seg_end

        # Trailing gap after last speech
        if current_pos < total_video_duration - 0.05:
            sections.append({
                "type": "gap",
                "video_start": current_pos,
                "video_end": total_video_duration,
            })

        # Create video clips for each section
        num_sections = len(sections)
        clip_paths = []

        for idx, sec in enumerate(sections):
            self._report("assemble",
                         0.1 + 0.6 * (idx / max(num_sections, 1)),
                         f"Syncing section {idx + 1}/{num_sections}...")

            clip = self.cfg.work_dir / f"vsync_{idx:04d}.mp4"
            vs = sec["video_start"]
            ve = sec["video_end"]
            dur = ve - vs

            if dur < 0.05:
                continue

            pts_factor = sec.get("pts_factor", 1.0)

            if sec["type"] == "gap" or abs(pts_factor - 1.0) < 0.08:
                # No speed change needed — extract at normal speed
                subprocess.run(
                    [self._ffmpeg, "-y",
                     "-ss", f"{vs:.3f}", "-i", str(video_path),
                     "-t", f"{dur:.3f}",
                     "-an",
                     "-c:v", "libx264", "-preset", self.cfg.encode_preset, "-crf", "20",
                     str(clip)],
                    check=True, capture_output=True,
                )
            else:
                # Adjust video speed: setpts=factor*PTS
                # factor > 1 = slow down, factor < 1 = speed up
                subprocess.run(
                    [self._ffmpeg, "-y",
                     "-ss", f"{vs:.3f}", "-i", str(video_path),
                     "-t", f"{dur:.3f}",
                     "-filter:v", f"setpts={pts_factor:.6f}*PTS",
                     "-an",
                     "-c:v", "libx264", "-preset", self.cfg.encode_preset, "-crf", "20",
                     str(clip)],
                    check=True, capture_output=True,
                )

            clip_paths.append(clip)

        if not clip_paths:
            raise RuntimeError("No video sections produced")

        # Concatenate all video clips
        self._report("assemble", 0.75, "Joining video sections...")
        synced_video = self.cfg.work_dir / "video_synced.mp4"
        if len(clip_paths) == 1:
            shutil.copy2(clip_paths[0], synced_video)
        else:
            self._concatenate_videos(clip_paths, synced_video)

        # Build TTS audio timeline matching the adjusted video timing
        new_pos = 0.0
        audio_segments = []
        for sec in sections:
            dur = sec["video_end"] - sec["video_start"]
            if dur < 0.05:
                continue
            if sec["type"] == "speech":
                audio_segments.append({
                    "start": new_pos,
                    "wav": sec["tts_wav"],
                    "duration": sec["tts_dur"],
                })
                new_pos += sec["tts_dur"]
            else:
                new_pos += dur  # gaps keep original duration

        self._report("assemble", 0.85, "Building audio timeline...")
        synced_audio = self._build_timeline(audio_segments, new_pos, prefix="synced_")

        # Mix original audio at low volume if requested
        if self.cfg.mix_original:
            synced_audio = self._mix_audio(audio_raw, synced_audio, self.cfg.original_volume)

        # Mux final video + audio
        self._report("assemble", 0.90, "Muxing final video...")
        self._mux_replace_audio(synced_video, synced_audio, self.cfg.output_path)

    @staticmethod
    def _parse_tts_rate(rate_str: str) -> int:
        """Parse edge-tts rate string like '-5%' or '+20%' to integer."""
        return int(rate_str.replace("%", "").replace("+", ""))

    def _time_stretch(self, wav_path: Path, ratio: float, output_path: Path) -> Path:
        """Time-stretch audio (ratio > 1 = speed up). Tries rubberband then atempo."""
        # Try ffmpeg rubberband filter first (better pitch preservation)
        try:
            subprocess.run(
                [self._ffmpeg, "-y", "-i", str(wav_path),
                 "-filter:a", f"rubberband=tempo={ratio:.4f}",
                 "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                 str(output_path)],
                check=True, capture_output=True,
            )
            return output_path
        except subprocess.CalledProcessError:
            pass  # rubberband not available, fall back to atempo

        # Fallback: atempo filter (chain for ratios outside 0.5-2.0)
        tempo = ratio
        filters = []
        while tempo > 2.0:
            filters.append("atempo=2.0")
            tempo /= 2.0
        while tempo < 0.5:
            filters.append("atempo=0.5")
            tempo /= 0.5
        filters.append(f"atempo={tempo:.4f}")
        subprocess.run(
            [self._ffmpeg, "-y", "-i", str(wav_path),
             "-filter:a", ",".join(filters),
             "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
             str(output_path)],
            check=True, capture_output=True,
        )
        return output_path

    # ── Duration & tempo adjustment ───────────────────────────────────────
    def _get_duration(self, media_path: Path) -> float:
        """Get duration of a media file in seconds using ffprobe."""
        ffmpeg_path = Path(self._ffmpeg)
        if ffmpeg_path.is_absolute():
            ffprobe = str(ffmpeg_path.parent / "ffprobe")
            if sys.platform == "win32" and not ffprobe.endswith(".exe"):
                ffprobe += ".exe"
        else:
            ffprobe = shutil.which("ffprobe") or "ffprobe"
        try:
            result = subprocess.run(
                [ffprobe, "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(media_path)],
                capture_output=True, text=True, timeout=15,
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _adjust_tempo(self, wav_path: Path, ratio: float) -> Path:
        """Speed up or slow down audio to match video duration.

        ratio = tts_duration / video_duration
        ratio > 1 means TTS is longer → speed up (atempo > 1)
        ratio < 1 means TTS is shorter → slow down (atempo < 1)
        """
        adjusted = self.cfg.work_dir / "tts_adjusted.wav"

        # ffmpeg atempo filter accepts 0.5 to 100.0
        # For values outside 0.5-2.0, chain multiple filters
        tempo = ratio
        filters = []
        while tempo > 2.0:
            filters.append("atempo=2.0")
            tempo /= 2.0
        while tempo < 0.5:
            filters.append("atempo=0.5")
            tempo /= 0.5
        filters.append(f"atempo={tempo:.4f}")

        filter_str = ",".join(filters)
        subprocess.run(
            [
                self._ffmpeg, "-y", "-i", str(wav_path),
                "-filter:a", filter_str,
                "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                str(adjusted),
            ],
            check=True, capture_output=True,
        )
        return adjusted

    def _adjust_video_duration(self, video_path: Path, target_duration: float) -> Path:
        """Adjust video duration to match the dubbed audio using setpts filter.

        If dubbed audio is longer than video → slow down video (scenes last longer).
        If dubbed audio is shorter than video → speed up video (scenes go faster).
        """
        video_duration = self._get_duration(video_path)
        if video_duration <= 0 or target_duration <= 0:
            return video_path

        # PTS factor: >1 slows video down, <1 speeds it up
        pts_factor = target_duration / video_duration
        if abs(pts_factor - 1.0) < 0.02:  # Less than 2% difference, skip
            return video_path

        adjusted = self.cfg.work_dir / "video_adjusted.mp4"
        self._report("assemble", 0.1,
                     f"Adjusting video speed ({1/pts_factor:.2f}x) to match audio...")

        # setpts=PTS*factor changes video timing
        # factor > 1 → slower (stretches video), factor < 1 → faster (compresses video)
        # fps filter re-establishes constant frame rate after pts change
        subprocess.run(
            [
                self._ffmpeg, "-y", "-i", str(video_path),
                "-filter:v", f"setpts={pts_factor:.6f}*PTS",
                "-an",  # Drop original audio (we'll add dubbed audio)
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                str(adjusted),
            ],
            check=True, capture_output=True,
        )
        return adjusted

    # ── Audio mixing ─────────────────────────────────────────────────────
    def _mix_audio(self, original: Path, tts: Path, original_vol: float) -> Path:
        mixed = self.cfg.work_dir / "audio_mixed.wav"
        subprocess.run(
            [
                self._ffmpeg, "-y",
                "-i", str(tts),
                "-i", str(original),
                "-filter_complex",
                f"[1:a]volume={original_vol}[orig];[0:a][orig]amix=inputs=2:duration=first:dropout_transition=2[out]",
                "-map", "[out]",
                "-ar", str(self.SAMPLE_RATE),
                "-ac", str(self.N_CHANNELS),
                str(mixed),
            ],
            check=True,
            capture_output=True,
        )
        return mixed

    # ── Video split / concat ─────────────────────────────────────────────
    def _split_video(self, video_path: Path, start: float, duration: float, output_path: Path):
        """Extract a clip from the video using stream copy (fast, no re-encode)."""
        subprocess.run(
            [self._ffmpeg, "-y",
             "-ss", f"{start:.3f}", "-i", str(video_path),
             "-t", f"{duration:.3f}",
             "-c", "copy", "-an",  # copy video only, drop audio
             str(output_path)],
            check=True, capture_output=True,
        )

    def _concatenate_videos(self, video_paths: List[Path], output_path: Path):
        """Concatenate multiple video files using ffmpeg concat demuxer."""
        concat_list = self.cfg.work_dir / "concat_list.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for vp in video_paths:
                # ffmpeg concat needs forward slashes even on Windows
                safe_path = str(vp).replace("\\", "/")
                f.write(f"file '{safe_path}'\n")
        subprocess.run(
            [self._ffmpeg, "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_list), "-c", "copy",
             str(output_path)],
            check=True, capture_output=True,
        )

    # ── Fast assembly: build audio timeline + stream-copy video ────────
    def _assemble_fast_mux(self, video_path, audio_raw, tts_data, total_video_duration):
        """Ultra-fast assembly for long videos: build full audio, mux with original video.

        No per-segment video re-encoding. Just:
        1. Build the complete dubbed audio timeline (TTS segments placed at correct times)
        2. Optionally mix with original background audio
        3. Stream-copy the video and replace audio track → instant mux
        """
        num_segs = len(tts_data)
        self._report("assemble", 0.05, f"Building audio timeline for {num_segs} segments...")

        # Speed-fit TTS to their time slots
        fitted = self._speed_fit_segments(tts_data)

        # Build the dubbed audio timeline
        self._report("assemble", 0.2, "Placing audio segments on timeline...")
        dubbed_audio = self._build_timeline_no_cut(fitted, total_video_duration, prefix="fast_")

        # Mix with original audio if requested
        if self.cfg.mix_original and audio_raw and audio_raw.exists():
            self._report("assemble", 0.6, "Mixing with original audio...")
            mixed_audio = self.cfg.work_dir / "fast_mixed.wav"
            vol = self.cfg.original_volume
            subprocess.run(
                [self._ffmpeg, "-y",
                 "-i", str(dubbed_audio),
                 "-i", str(audio_raw),
                 "-filter_complex",
                 f"[1:a]volume={vol}[bg];[0:a][bg]amix=inputs=2:duration=longest",
                 "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                 str(mixed_audio)],
                check=True, capture_output=True,
            )
            dubbed_audio = mixed_audio

        # Stream-copy video + replace audio (NO re-encode = instant)
        self._report("assemble", 0.8, "Muxing audio with video (stream copy — fast)...")
        self._mux_replace_audio(video_path, dubbed_audio, self.cfg.output_path)
        self._report("assemble", 0.95, "Fast assembly complete!")

    # ── Video muxing ─────────────────────────────────────────────────────
    def _mux_replace_audio(self, video_path: Path, audio_path: Path, output_path: Path):
        subprocess.run(
            [
                self._ffmpeg, "-y",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", self.cfg.audio_bitrate,
                "-map", "0:v:0",
                "-map", "1:a:0",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )


async def list_voices(language_filter: str = "hi"):
    """List available edge-tts voices filtered by language."""
    import edge_tts

    voices = await edge_tts.list_voices()
    if language_filter:
        voices = [v for v in voices if v.get("Locale", "").startswith(language_filter)]
    return voices
