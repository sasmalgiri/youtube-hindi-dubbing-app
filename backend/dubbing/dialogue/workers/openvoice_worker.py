"""OpenVoice v2 tone colour converter worker (runs in its own process, usually its own venv).

Standalone on purpose: imports nothing from the app, so it can run under a
separate interpreter with the versions OpenVoice pins (numpy 1.22.0,
librosa 0.9.1).

Makes a finished Hindi TTS clip sound more like the original speaker: only
the voice colour (timbre) changes; words, pronunciation and timing stay the
TTS engine's. Checkpoint: the "converter" folder of OpenVoice's
checkpoints_v2 (Hugging Face myshell-ai/OpenVoiceV2, MIT), downloaded once
into backend/models/openvoice (copy converter/config.json and
converter/checkpoint.pth there by hand to skip the download). The calls
follow OpenVoice's own demo and SoniTranslate's toneconverter_openvoice /
se_process_audio_segments: ToneColorConverter(config, device) +
load_ckpt(checkpoint), extract_se(wavs) for a speaker embedding (the mean
over the files), convert(src, src_se, tgt_se). OpenVoice adds its
inaudible watermark to what it converts.

The output keeps the input's sample rate, channel count and exact length
(the converter runs at 22.05 kHz and may end up to one STFT hop off; that
is resampled back and padded/trimmed at the end), so the dub's timing never
changes.

Requests (one JSON per line):
  {"op":"init","models_dir":"C:/.../backend/models/openvoice","tau":0.3}
  {"op":"prepare","speaker":"S1","refs":["C:/.../ref_1.wav"],"se_path":"C:/.../se_S1.pth"}
  {"op":"convert","in":"C:/.../t0001.wav","out":"C:/.../t0001.wav","speaker":"S1",
   "source_key":null}
  {"op":"quit"}
"""
import json
import os
import sys
import wave
from pathlib import Path

_PROTO = sys.stdout
sys.stdout = sys.stderr          # library prints must not corrupt the protocol


def reply(**kw):
    _PROTO.write(json.dumps(kw, ensure_ascii=False) + "\n")
    _PROTO.flush()


REPO = "myshell-ai/OpenVoiceV2"
FILES = ("converter/config.json", "converter/checkpoint.pth")
# backend/models/openvoice (this file is backend/dubbing/dialogue/workers/...)
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[3] / "models" / "openvoice"
# The converter's output can be up to one STFT hop (256 samples at 22.05 kHz,
# 11.6 ms) plus resampling rounding off the input; more means something went
# wrong, and the clip is refused rather than re-timed.
MAX_LENGTH_FIX_S = 0.03
MIN_CLIP_S = 0.3                 # shorter clips give no usable spectrogram/embedding
SOURCE_POOL = 3                  # clips averaged into a cached per-voice source embedding
SOURCE_MIN_S = 1.0               # shorter clips do not join that average
WATERMARK = "@MyShell"           # OpenVoice's own demo message

STATE = {}


def init(req):
    import torch
    from openvoice.api import ToneColorConverter
    models = Path(req.get("models_dir") or DEFAULT_MODELS_DIR)
    paths = [models / f for f in FILES]
    if not all(p.is_file() and p.stat().st_size > 0 for p in paths):
        from huggingface_hub import hf_hub_download
        models.mkdir(parents=True, exist_ok=True)
        paths = [Path(hf_hub_download(REPO, f, local_dir=str(models))) for f in FILES]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    conv = ToneColorConverter(str(paths[0]), device=device)
    conv.load_ckpt(str(paths[1]))
    STATE.update(conv=conv, torch=torch, device=device, tau=float(req.get("tau") or 0.3),
                 targets={}, sources={})
    return {"device": device, "sampling_rate": int(conv.hps.data.sampling_rate),
            "version": str(getattr(conv, "version", "")), "converter": str(paths[0].parent)}


def prepare(req):
    """Target speaker embedding from the reference wavs, kept for the job
    (and saved to se_path, so a resumed run reuses it)."""
    torch, conv = STATE["torch"], STATE["conv"]
    se_path = req.get("se_path") or None
    se = None
    if se_path and os.path.isfile(se_path):
        try:
            se = torch.load(se_path, map_location=STATE["device"], weights_only=True)
        except Exception as e:      # a damaged cache file: extract again
            print(f"[openvoice] ignoring {se_path}: {e}", file=sys.stderr)
    cached = se is not None
    if se is None:
        refs = [r for r in req.get("refs") or [] if os.path.isfile(r)]
        if not refs:
            raise ValueError("no reference audio found")
        if se_path:
            os.makedirs(os.path.dirname(se_path) or ".", exist_ok=True)
        se = conv.extract_se(refs, se_save_path=se_path)
    STATE["targets"][req["speaker"]] = se.to(STATE["device"])
    return {"speaker": req["speaker"], "cached": cached}


def _mean(torch, ses):
    return ses[0] if len(ses) == 1 else torch.stack(ses).mean(0)


def source_se(path, key, seconds):
    """The embedding of the voice being converted: from the clip itself, or,
    with a source_key (one TTS voice), the mean over that voice's first clips."""
    torch, conv = STATE["torch"], STATE["conv"]
    pool = STATE["sources"].setdefault(key, []) if key else None
    if pool is not None and len(pool) >= SOURCE_POOL:
        return _mean(torch, pool)
    se = conv.extract_se([path])
    if pool is None:
        return se
    if seconds >= SOURCE_MIN_S:
        pool.append(se)
    return _mean(torch, pool) if pool else se


def read_wav(path):
    """(float32 samples shaped (frames, channels), sample rate) of a 16-bit PCM wav."""
    import numpy as np
    with wave.open(path, "rb") as wf:
        sr, ch, sw, n = wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes()
        raw = wf.readframes(n)
    if sw != 2:
        raise ValueError(f"expected a 16-bit PCM wav: {path}")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, ch) / 32768.0, sr


def write_wav(path, data, sr):
    """Written next to the target and moved into place: `path` may be the input."""
    import numpy as np
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    tmp = f"{path}.vm.tmp"
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(pcm.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    os.replace(tmp, path)


def _rms(x):
    import numpy as np
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def convert(req):
    import numpy as np
    conv = STATE["conv"]
    tgt = STATE["targets"].get(req["speaker"])
    if tgt is None:      # the app re-sends "prepare" on this message (e.g. after a restart)
        raise RuntimeError(f"speaker {req['speaker']} was not prepared")
    src = req["in"]
    x, sr = read_wav(src)
    n, ch = x.shape
    if n < MIN_CLIP_S * sr:
        raise ValueError(f"clip too short to convert ({n / sr:.2f} s)")
    y = conv.convert(audio_src_path=src, src_se=source_se(src, req.get("source_key"), n / sr),
                     tgt_se=tgt, output_path=None, tau=STATE["tau"], message=WATERMARK)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    model_sr = int(conv.hps.data.sampling_rate)
    if model_sr != sr:
        import librosa
        y = librosa.resample(y, orig_sr=model_sr, target_sr=sr).astype(np.float32)
    fix = len(y) - n
    if abs(fix) > MAX_LENGTH_FIX_S * sr:
        raise ValueError(f"converted audio is {fix / sr * 1000:+.0f} ms off the clip's length")
    y = y[:n] if fix > 0 else np.pad(y, (0, -fix))
    # Same speech level as the TTS clip (bounded), never clipping.
    mono = x.mean(axis=1)
    level_in, level_out = _rms(mono), _rms(y)
    if level_in > 1e-5 and level_out > 1e-5:
        y = y * min(4.0, max(0.25, level_in / level_out))
    peak = float(np.max(np.abs(y)))
    if peak > 0.98:
        y = y * (0.98 / peak)
    write_wav(req["out"], np.repeat(y[:, None], ch, axis=1), sr)
    # length_off_ms: how much longer (+, trimmed) or shorter (-, padded) the
    # converter's own output was
    return {"out": req["out"], "sampling_rate": sr, "frames": n,
            "length_off_ms": round(fix / sr * 1000.0, 1)}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "quit":
                reply(ok=True)
                break
            if op == "init":
                reply(ok=True, info=init(req))
            elif op == "prepare":
                reply(ok=True, **prepare(req))
            elif op == "convert":
                reply(ok=True, **convert(req))
            else:
                reply(ok=False, error=f"unknown op {op}")
        except Exception as e:  # report and keep serving
            reply(ok=False, error=f"{type(e).__name__}: {str(e)[:400]}")


if __name__ == "__main__":
    main()
