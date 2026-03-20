from io import BytesIO
import os
import subprocess
from typing import List, Tuple
from .aligner import align_audio_segments
from .mixer import Mixer
from ..io.youtube import download_video, fetch_captions
from ..stt.transcriber import Transcriber
from ..mt.translator import translate_text
from ..tts.synthesizer import Synthesizer
from ..subtitles.srt import create_srt_subtitle
from ..utils.logging import setup_logging
from ..utils.watermark import detect_static_overlay, remove_watermark

logger = setup_logging()

class DubbingPipeline:
    def __init__(self, video_url: str, target_language: str, remove_watermark: bool = False):
        self.video_url = video_url
        self.target_language = target_language
        self.remove_watermark_flag = remove_watermark
        self.video_path = None
        self.audio_path = None
        self.transcription = None
        self.translated_text = None
        self.dubbed_audio_path = None

    def run(self):
        logger.info("Starting the dubbing pipeline.")
        self.download_video()
        if self.remove_watermark_flag:
            self.clean_watermark()
        self.extract_audio()
        self.transcribe_audio()
        self.translate_text()
        self.synthesize_speech()
        self.create_subtitles()
        self.mix_audio()

    def download_video(self):
        logger.info("Downloading video from URL.")
        self.video_path = download_video(self.video_url)

    def clean_watermark(self):
        logger.info("Detecting and removing watermark from video.")
        cleaned_path = self.video_path.replace('.mp4', '_clean.mp4')
        remove_watermark(self.video_path, cleaned_path)
        self.video_path = cleaned_path

    def extract_audio(self):
        logger.info("Extracting audio from video.")
        self.audio_path = self.video_path.replace('.mp4', '.wav')
        subprocess.run(['ffmpeg', '-i', self.video_path, self.audio_path])

    def transcribe_audio(self):
        logger.info("Transcribing audio.")
        transcriber = Transcriber()
        self.transcription = transcriber.transcribe_audio(self.audio_path)

    def translate_text(self):
        logger.info("Translating text.")
        self.translated_text = translate_text(self.transcription, self.target_language)

    def synthesize_speech(self):
        logger.info("Synthesizing speech.")
        self.dubbed_audio_path = self.audio_path.replace('.wav', f'_{self.target_language}.wav')
        synthesizer = Synthesizer(language=self.target_language)
        synthesizer.synthesize(self.translated_text, self.dubbed_audio_path)

    def create_subtitles(self):
        logger.info("Creating subtitles.")
        srt_path = self.audio_path.replace('.wav', f'_{self.target_language}.srt')
        create_srt_subtitle(self.transcription, srt_path)

    def mix_audio(self):
        logger.info("Mixing dubbed audio with original audio.")
        mixed_audio_path = self.audio_path.replace('.wav', f'_mixed_{self.target_language}.wav')
        mixer = Mixer(self.audio_path, self.dubbed_audio_path)
        mixer.mix_audios(mixed_audio_path)