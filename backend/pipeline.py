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
    asr_model: str = "small"
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


class Pipeline:
    """Dubbing pipeline with translation and callback-based progress."""

    SAMPLE_RATE = 48000
    N_CHANNELS = 2

    def __init__(self, cfg: PipelineConfig, on_progress: Optional[ProgressCallback] = None):
        self.cfg = cfg
        self._on_progress = on_progress or (lambda *_: None)
        self.segments: List[Dict] = []
        self.video_title: str = ""
        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)

        # Resolve executable paths
        self._ytdlp = self._find_executable("yt-dlp")
        self._ffmpeg = "ffmpeg"  # resolved in _ensure_ffmpeg

    @staticmethod
    def _find_executable(name: str) -> str:
        """Find an executable by checking venv, PATH, WinGet packages, and system PATH."""
        ext = ".exe" if sys.platform == "win32" else ""
        full_name = name + ext

        # 1. Check venv Scripts dir (where python.exe lives)
        venv_path = Path(sys.executable).parent / full_name
        if venv_path.exists():
            return str(venv_path)

        # 2. Check current PATH
        found = shutil.which(name)
        if found:
            return found

        if sys.platform == "win32":
            # 3. Scan WinGet packages directory
            localappdata = os.environ.get("LOCALAPPDATA", "")
            if localappdata:
                winget_pkgs = Path(localappdata) / "Microsoft" / "WinGet" / "Packages"
                if winget_pkgs.exists():
                    for exe in winget_pkgs.rglob(full_name):
                        os.environ["PATH"] = str(exe.parent) + os.pathsep + os.environ.get("PATH", "")
                        return str(exe)

            # 4. Refresh PATH from system registry and try again
            try:
                result = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-Command",
                     "[System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    os.environ["PATH"] = result.stdout.strip() + os.pathsep + os.environ.get("PATH", "")
                    found = shutil.which(name)
                    if found:
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
        """Detect gender per speaker using pitch (F0) analysis. Male < 165Hz, Female >= 165Hz."""
        import struct

        with wave.open(str(wav_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)

        # Convert to float samples (mono)
        fmt = f"<{n_frames * n_channels}h" if sample_width == 2 else f"<{n_frames * n_channels}i"
        try:
            samples = list(struct.unpack(fmt, raw_data))
        except struct.error:
            # Fallback: treat all as unknown
            return {spk: "female" for spk in speakers}

        # Take every nth channel for mono
        if n_channels > 1:
            samples = samples[::n_channels]

        max_val = float(2 ** (8 * sample_width - 1))
        samples = [s / max_val for s in samples]

        result = {}
        for speaker, time_ranges in speakers.items():
            # Collect audio samples for this speaker
            speaker_samples = []
            for t_start, t_end in time_ranges[:10]:  # Limit to first 10 segments
                s_start = int(t_start * sample_rate)
                s_end = int(t_end * sample_rate)
                s_start = max(0, min(s_start, len(samples) - 1))
                s_end = max(0, min(s_end, len(samples)))
                speaker_samples.extend(samples[s_start:s_end])

            if len(speaker_samples) < sample_rate * 0.5:
                # Too little audio, default to female
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
            self.segments = self._transcribe(audio_raw)
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
                self._report("transcribe", 0.99,
                             f"Assigned {len(self._voice_map)} distinct voices")

        # Step 4: Translate each segment (preserving timestamps for scene sync)
        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        self._report("translate", 0.0, f"Translating segments to {target_name}...")
        self._translate_segments(text_segments)
        self.segments = text_segments
        self._report("translate", 1.0, "Translation complete")

        # Write translated SRT (per-segment subtitles with proper timestamps)
        srt_translated = self.cfg.work_dir / f"transcript_{self.cfg.target_language}.srt"
        write_srt(self.segments, srt_translated, text_key="text_translated")

        # Step 5: Generate TTS at natural speed, then speed up to 1.5x
        self._report("synthesize", 0.0,
                     f"Generating natural speech ({self.cfg.tts_voice})...")
        tts_data = self._generate_tts_natural(text_segments)
        self._report("synthesize", 0.9,
                     f"Generated {len(tts_data)} segments, speeding up to 1.25x...")

        # Speed up all TTS clips to 1.25x for snappier delivery
        TTS_SPEED = 1.25
        for idx, tts in enumerate(tts_data):
            sped_wav = self.cfg.work_dir / f"tts_fast_{idx:04d}.wav"
            self._time_stretch(tts["wav"], TTS_SPEED, sped_wav)
            tts["wav"] = sped_wav
            tts["duration"] = tts["duration"] / TTS_SPEED

        self._report("synthesize", 1.0,
                     f"All {len(tts_data)} segments at 1.25x speed")

        # Step 6: Assemble — AUDIO IS MASTER, video adapts
        # TTS plays at natural speed (uniform, no speedup/slowdown).
        # Video is slowed (max 1.1x) and scenes are looped to match audio duration.
        self._report("assemble", 0.0, "Building dubbed output (video adapts to audio)...")
        self.cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        video_duration = self._get_duration(video_path)

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

            # Get video title first
            try:
                title_cmd = [self._ytdlp, "--print", "%(title)s", "--no-download"] + cookies_args + [src]
                title_result = subprocess.run(
                    title_cmd, capture_output=True, text=True, timeout=30,
                )
                if title_result.returncode == 0 and title_result.stdout.strip():
                    self.video_title = title_result.stdout.strip().split("\n")[0]
            except Exception:
                self.video_title = "Untitled"

            self._report("download", 0.2, f"Downloading: {self.video_title}")

            # Download video
            try:
                dl_cmd = [
                    self._ytdlp,
                    "--ffmpeg-location", str(Path(self._ffmpeg).parent),
                    "-f", "bv*+ba/b",
                    "--merge-output-format", "mp4",
                    "-o", out_tpl,
                ] + cookies_args + [src]
                subprocess.run(dl_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "Unknown error")
                raise RuntimeError(f"yt-dlp failed: {stderr}") from e

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
        wav = self.cfg.work_dir / "audio_raw.wav"
        subprocess.run(
            [
                self._ffmpeg, "-y", "-i", str(video_path),
                "-vn", "-ac", str(self.N_CHANNELS), "-ar", str(self.SAMPLE_RATE),
                "-acodec", "pcm_s16le", str(wav),
            ],
            check=True,
            capture_output=True,
        )
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
                f.unlink()

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
        """Transcribe speech from audio using Whisper (picks up only spoken words)."""
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
        # Pass language hint if source language is specified (not auto)
        transcribe_kwargs = {"vad_filter": True, "word_timestamps": True}
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
            # Word-level timestamps for fine-grained alignment
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
        if gemini_key:
            translated_text = self._translate_with_gemini(full_text, gemini_key, speech_duration)
        else:
            self._report("translate", 0.25, "No GEMINI_API_KEY found, using Google Translate...")
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
                f"Translate the following narration from {source_name} into natural, fluent {target_name}. "
                f"This is a voiceover script for a dubbed video, so it must sound like a native "
                f"{target_name} speaker is narrating — conversational, smooth, and natural. "
                f"{duration_hint}{chunk_hint}"
                f"Do NOT translate literally word-by-word. Adapt idioms and phrasing to sound "
                f"natural in {target_name}. Keep proper nouns (names, places, brands) as-is. "
                f"Output ONLY the {target_name} translation, nothing else.\n\n"
                f"{chunk}"
            )

            retries = 3
            for attempt in range(retries):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-pro",
                        contents=prompt,
                    )
                    translated_parts.append(response.text.strip())
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        wait = 2 * (attempt + 1)
                        self._report("translate", 0.2 + 0.6 * (i / len(chunks)),
                                     f"Rate limited, retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        self._report("translate", 0.2, f"Gemini failed: {e}, falling back to Google Translate...")
                        return self._translate_with_google(full_text)

            self._report("translate", 0.2 + 0.8 * ((i + 1) / len(chunks)),
                         f"Translated chunk {i + 1}/{len(chunks)} (Gemini)")

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

    # ── Step 4b: Segment-level translation ────────────────────────────────
    def _translate_segments(self, segments):
        """Translate each segment individually, preserving timestamps for sync.

        Priority: OpenAI GPT-4o > Groq Llama 3.3 > Gemini > Google Translate fallback
        """
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

        if openai_key:
            self._report("translate", 0.05, "Using GPT-4o for premium translation...")
            self._translate_segments_openai(segments, openai_key)
        elif gemini_key:
            self._report("translate", 0.05, "Using Gemini for colloquial translation...")
            self._translate_segments_gemini(segments, gemini_key)
        elif groq_key:
            self._report("translate", 0.05, "Using Groq (Llama 3.3 70B) for translation...")
            self._translate_segments_groq(segments, groq_key)
        else:
            self._report("translate", 0.1, "No API keys found, using Google Translate...")
            self._translate_segments_google(segments)

    def _translate_segments_gemini(self, segments, api_key):
        """Translate segments in numbered batches using Gemini for context-aware output."""
        from google import genai
        client = genai.Client(api_key=api_key)

        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        source_name = LANGUAGE_NAMES.get(self.cfg.source_language, "the source language") if self.cfg.source_language != "auto" else "the detected language"

        batch_size = 30
        total_batches = (len(segments) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batch = segments[start:end]

            # Build numbered input
            lines = []
            for i, seg in enumerate(batch):
                lines.append(f"{i+1}. {seg['text']}")

            prompt = (
                f"You are a world-class dubbing translator. Your ONE job: make {target_name} translations "
                f"sound like they were ORIGINALLY written in {target_name} by a native speaker.\n\n"
                f"Translate each numbered line from {source_name} to {target_name}.\n\n"
                f"VOICE & STYLE (THE #1 PRIORITY):\n"
                f"- Sound like a REAL PERSON talking — a friend explaining something, a YouTuber narrating, "
                f"a podcast host telling a story. NOT a textbook, NOT a news anchor, NOT a robot.\n"
                f"- Use the EXACT way native speakers ACTUALLY talk in everyday life.\n"
                f"  - Hindi example: 'toh basically yeh cheez hai ki...' NOT 'yah vastu yah hai ki...'\n"
                f"  - Bengali example: 'mane hocche ta holo...' NOT 'arthat etar tatparya holo...'\n"
                f"- Keep English words that people naturally use: 'actually', 'problem', 'basically', "
                f"'phone', 'video', 'use', 'point', 'so', 'but', 'right?', 'like' etc.\n"
                f"- Use contractions, filler words, and sentence connectors that real people use.\n"
                f"- Match the speaker's ENERGY and EMOTION exactly — excited = excited, sarcastic = sarcastic.\n"
                f"- NEVER use formal/literary/textbook language. NEVER use shudh Hindi or sadhu Bengali.\n"
                f"- Keep proper nouns, brand names, and technical terms unchanged.\n\n"
                f"TRANSLATION COMPLETENESS (ABSOLUTELY CRITICAL):\n"
                f"- Translate EVERY word and EVERY idea. NOTHING may be skipped or summarized.\n"
                f"- Each line must be COMPLETE — no truncation, no cutting, no paraphrasing that loses detail.\n"
                f"- The video will adapt to match audio length — so DO NOT worry about translation length.\n"
                f"- Longer translations are fine. Incomplete translations are UNACCEPTABLE.\n\n"
                f"Output ONLY the numbered {target_name} translations, one per line, matching input numbering.\n"
                f"Do NOT add notes, comments, explanations, or metadata.\n\n"
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

        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        source_name = LANGUAGE_NAMES.get(self.cfg.source_language, "the source language") if self.cfg.source_language != "auto" else "the detected language"

        batch_size = 30
        total_batches = (len(segments) + batch_size - 1) // batch_size

        system_msg = (
            f"You are a world-class dubbing translator. Make {target_name} sound like it was ORIGINALLY "
            f"written by a native speaker — a real person talking, not a textbook.\n"
            f"Sound like a YouTuber, podcast host, or friend explaining something. "
            f"Use the EXACT way native speakers talk in everyday life. "
            f"Keep English words people naturally mix in: 'actually', 'basically', 'problem', 'phone', 'video', 'so', 'but'. "
            f"NEVER use formal/literary/textbook language. NO shudh Hindi, NO sadhu Bengali. "
            f"Match the speaker's energy and emotion exactly. "
            f"Keep proper nouns, brands, and technical terms unchanged.\n\n"
            f"COMPLETENESS: Translate EVERY word and idea. NOTHING may be skipped. "
            f"Each translation must be COMPLETE. Video adapts to audio — don't worry about length. "
            f"Output ONLY numbered translations. No notes or metadata."
        )

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batch = segments[start:end]

            lines = []
            for i, seg in enumerate(batch):
                lines.append(f"{i+1}. {seg['text']}")

            user_msg = (
                f"Translate each line from {source_name} to {target_name}. "
                f"Natural, conversational, the way real people actually talk — like a friend explaining something. "
                f"Translate COMPLETELY — every idea, every nuance. Video adapts to audio length.\n\n"
                + "\n".join(lines)
            )

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
                        max_tokens=4096,
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

    def _translate_segments_openai(self, segments, api_key):
        """Translate segments using OpenAI GPT-4o for highest quality."""
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        target_name = LANGUAGE_NAMES.get(self.cfg.target_language, self.cfg.target_language)
        source_name = LANGUAGE_NAMES.get(self.cfg.source_language, "the source language") if self.cfg.source_language != "auto" else "the detected language"

        batch_size = 30
        total_batches = (len(segments) + batch_size - 1) // batch_size

        system_msg = (
            f"You are a world-class dubbing translator. Make {target_name} sound like it was ORIGINALLY "
            f"written by a native speaker — a real person talking, not a textbook.\n"
            f"Sound like a YouTuber, podcast host, or friend explaining something. "
            f"Use the EXACT way native speakers talk in everyday life. "
            f"Keep English words people naturally mix in: 'actually', 'basically', 'problem', 'phone', 'video', 'so', 'but'. "
            f"NEVER use formal/literary/textbook language. NO shudh Hindi, NO sadhu Bengali. "
            f"Match the speaker's energy and emotion exactly. "
            f"Keep proper nouns, brands, and technical terms unchanged.\n\n"
            f"COMPLETENESS: Translate EVERY word and idea. NOTHING may be skipped. "
            f"Each translation must be COMPLETE. Video adapts to audio — don't worry about length. "
            f"Output ONLY numbered translations. No notes or metadata."
        )

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(segments))
            batch = segments[start:end]

            lines = []
            for i, seg in enumerate(batch):
                lines.append(f"{i+1}. {seg['text']}")

            user_msg = (
                f"Translate from {source_name} to {target_name}. "
                f"Natural, conversational, the way real people actually talk — like a friend explaining something. "
                f"Translate COMPLETELY — every idea, every nuance. Video adapts to audio length.\n\n"
                + "\n".join(lines)
            )

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
                text = seg.get("text_translated", seg["text"]).strip()
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
        """
        fitted = []
        for idx, tts in enumerate(tts_data):
            seg_start = tts["start"]
            seg_end = tts["end"]
            slot_dur = seg_end - seg_start
            tts_dur = tts["duration"]
            tts_wav = tts["wav"]

            if slot_dur < 0.1 or tts_dur < 0.1:
                fitted.append(tts.copy())
                continue

            ratio = tts_dur / slot_dur  # >1 = TTS longer than slot

            # Close enough — no adjustment needed
            if abs(ratio - 1.0) < 0.05:
                fitted.append(tts.copy())
                continue

            # Clamp ratio to our speed limits
            clamped_ratio = max(self.SPEED_MIN, min(ratio, self.SPEED_MAX))

            stretched_wav = self.cfg.work_dir / f"speedfit_{idx:04d}.wav"
            self._time_stretch(tts_wav, clamped_ratio, stretched_wav)

            new_dur = tts_dur / clamped_ratio  # duration after speed change
            fitted.append({
                "start": seg_start,
                "end": seg_end,
                "wav": stretched_wav,
                "duration": new_dur,
            })

        return fitted

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
            current_pos = seg_end

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
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
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
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
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
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
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
                         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
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
                        subprocess.run(
                            [self._ffmpeg, "-y",
                             "-sseof", "-0.1", "-i", str(slowed_clip),
                             "-frames:v", "1",
                             "-update", "1",
                             str(last_frame)],
                            check=True, capture_output=True,
                        )
                        # Create a still video from the last frame
                        freeze_clip = self.cfg.work_dir / f"adapt_{idx:04d}_freeze.mp4"
                        subprocess.run(
                            [self._ffmpeg, "-y",
                             "-loop", "1", "-i", str(last_frame),
                             "-t", f"{freeze_dur:.3f}",
                             "-vf", "fps=30",
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                             "-pix_fmt", "yuv420p",
                             str(freeze_clip)],
                            check=True, capture_output=True,
                        )
                        # Concatenate slowed clip + frozen frame clip
                        concat_list = self.cfg.work_dir / f"adapt_{idx:04d}_concat.txt"
                        concat_list.write_text(
                            f"file '{slowed_clip}'\nfile '{freeze_clip}'\n"
                        )
                        subprocess.run(
                            [self._ffmpeg, "-y",
                             "-f", "concat", "-safe", "0",
                             "-i", str(concat_list),
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
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
        for idx, sec in enumerate(sections):
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
                             f"Chatterbox is English-only — switching to Edge-TTS for {target_name}...")

        if self.cfg.use_elevenlabs:
            elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
            if elevenlabs_key:
                self._report("synthesize", 0.05, "Using ElevenLabs for human-like voice...")
                return self._tts_elevenlabs(segments, elevenlabs_key)
            if not self.cfg.use_chatterbox:
                raise RuntimeError("ElevenLabs enabled but ELEVENLABS_API_KEY not set in .env")

        if self.cfg.use_coqui_xtts:
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("No CUDA GPU available for XTTS")
                self._report("synthesize", 0.05, "Using Coqui XTTS v2 (GPU, voice cloning)...")
                return self._tts_coqui_xtts(segments)
            except Exception as e:
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

    def _tts_chatterbox(self, segments):
        """Generate TTS using Chatterbox — free, local, human-like AI voice."""
        import torch
        import torchaudio
        from chatterbox.tts import ChatterboxTTS

        self._report("synthesize", 0.05, "Loading Chatterbox model on GPU...")
        model = ChatterboxTTS.from_pretrained(device="cuda")

        tts_data = []
        for i, seg in enumerate(segments):
            text = seg.get("text_translated", seg["text"]).strip()
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

        # Free GPU memory
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
            text = seg.get("text_translated", seg["text"]).strip()
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
            text = seg.get("text_translated", seg["text"]).strip()
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
                import shutil
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

    def _tts_coqui_xtts(self, segments):
        """Generate TTS using Coqui XTTS v2 — open-source voice cloning.

        Can clone the original speaker's voice and speak in Hindi/Bengali/etc.
        Requires GPU. Uses the extracted audio as voice reference.
        """
        import torch
        import torchaudio
        from TTS.api import TTS

        self._report("synthesize", 0.02, "Loading XTTS v2 model on GPU (this may take a minute)...")
        tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

        # Use original audio as speaker reference for voice cloning
        ref_wav = self.cfg.work_dir / "voice_ref.wav"
        audio_raw = self.cfg.work_dir / "audio_raw.wav"
        if not ref_wav.exists() and audio_raw.exists():
            # Extract a 15-second clip from the original audio for voice reference
            subprocess.run(
                [self._ffmpeg, "-y", "-i", str(audio_raw),
                 "-t", "15", "-ar", "22050", "-ac", "1",
                 str(ref_wav)],
                check=True, capture_output=True,
            )

        # Check if we also have a user-provided reference in voices/
        user_ref = Path("voices/my_voice_refs")
        if user_ref.exists():
            ref_files = list(user_ref.glob("*.wav")) + list(user_ref.glob("*.mp3"))
            if ref_files:
                ref_wav = ref_files[0]
                self._report("synthesize", 0.05, f"Using custom voice reference: {ref_wav.name}")

        if not ref_wav.exists():
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

        tts_data = []
        for i, seg in enumerate(segments):
            text = seg.get("text_translated", seg["text"]).strip()
            if not text:
                continue

            wav_path = self.cfg.work_dir / f"tts_{i:04d}.wav"

            try:
                tts_model.tts_to_file(
                    text=text,
                    speaker_wav=str(ref_wav),
                    language=xtts_lang,
                    file_path=str(wav_path),
                )

                # Convert to target sample rate and stereo
                wav_fixed = self.cfg.work_dir / f"tts_{i:04d}_fixed.wav"
                subprocess.run(
                    [self._ffmpeg, "-y", "-i", str(wav_path),
                     "-ar", str(self.SAMPLE_RATE), "-ac", str(self.N_CHANNELS),
                     str(wav_fixed)],
                    check=True, capture_output=True,
                )
                import shutil
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

        # Free GPU memory
        del tts_model
        torch.cuda.empty_cache()

        return tts_data

    async def _edge_tts_single(self, text, mp3_path):
        """Generate a single segment with edge-tts."""
        import edge_tts
        comm = edge_tts.Communicate(text, self.cfg.tts_voice, rate=self.cfg.tts_rate)
        await comm.save(str(mp3_path))

    def _tts_edge(self, segments, voice_map=None):
        """Generate TTS using edge-tts (free Microsoft voices).
        If voice_map is provided, each segment uses its speaker's assigned voice.
        """
        import edge_tts
        default_voice = self.cfg.tts_voice
        rate = self.cfg.tts_rate

        async def generate():
            for i, seg in enumerate(segments):
                text = seg.get("text_translated", seg["text"]).strip()
                if not text:
                    continue
                # Pick voice: per-speaker if multi-speaker, else default
                seg_voice = default_voice
                if voice_map and "speaker_id" in seg:
                    seg_voice = voice_map.get(seg["speaker_id"], default_voice)
                mp3 = self.cfg.work_dir / f"tts_{i:04d}.mp3"
                try:
                    comm = edge_tts.Communicate(text, seg_voice, rate=rate)
                    await comm.save(str(mp3))
                    seg["_tts_mp3"] = mp3
                except Exception:
                    pass
                self._report(
                    "synthesize",
                    0.1 + 0.8 * ((i + 1) / len(segments)),
                    f"Synthesized {i + 1}/{len(segments)} segments ({seg_voice})...",
                )

        asyncio.run(generate())

        tts_data = []
        for i, seg in enumerate(segments):
            mp3 = seg.pop("_tts_mp3", None)
            if not mp3 or not mp3.exists():
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
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
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
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
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

        # Fallback: atempo filter (chain for ratios > 2.0)
        tempo = ratio
        filters = []
        while tempo > 2.0:
            filters.append("atempo=2.0")
            tempo /= 2.0
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
        ffprobe = str(Path(self._ffmpeg).parent / "ffprobe")
        if sys.platform == "win32" and not ffprobe.endswith(".exe"):
            ffprobe += ".exe"
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

    # ── Video muxing ─────────────────────────────────────────────────────
    def _mux_replace_audio(self, video_path: Path, audio_path: Path, output_path: Path):
        subprocess.run(
            [
                self._ffmpeg, "-y",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
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
