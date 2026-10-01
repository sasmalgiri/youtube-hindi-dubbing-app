"""Indic Parler-TTS worker (runs in its own process, possibly its own venv).

Standalone on purpose: imports nothing from the app, so it can run under a
separate interpreter with transformers==4.46.1 (the version parler-tts pins).

Model card: https://huggingface.co/ai4bharat/indic-parler-tts (Apache-2.0,
gated: accept the conditions and set HF_TOKEN). Uses two tokenizers: the
model tokenizer for the Hindi prompt and the text-encoder tokenizer
(flan-t5) for the voice description. The speaker is chosen by naming it in
the description. The card advises short inputs (~10-12 s of speech), so the
caller sends sentence-sized chunks.

Requests (one JSON per line):
  {"op":"init","model":"ai4bharat/indic-parler-tts"}
  {"op":"tts","text":"...","description":"...","out":"C:/.../x.wav","seed":1234}
  {"op":"quit"}
"""
import json
import os
import sys

_PROTO = sys.stdout
sys.stdout = sys.stderr          # library prints must not corrupt the protocol


def reply(**kw):
    _PROTO.write(json.dumps(kw, ensure_ascii=False) + "\n")
    _PROTO.flush()


STATE = {}


def init(req):
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    name = req.get("model") or "ai4bharat/indic-parler-tts"
    token = os.environ.get("HF_TOKEN") or None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float32)
    # Load, then cast the WHOLE model. from_pretrained(torch_dtype=bf16) only
    # converts the top-level layers: the text encoder, decoder and DAC are
    # built from sub-configs that say float32 and stay fp32, so generate()
    # fails on every line with a dtype mismatch (BFloat16 != float) and
    # every clip falls back to another voice. The parler-tts README loads
    # the same way (.to(device, dtype=...)).
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        name, token=token, attn_implementation="eager").to(device, dtype=dtype)
    tok = AutoTokenizer.from_pretrained(name, token=token)
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path, token=token)
    STATE.update(model=model, tok=tok, desc_tok=desc_tok, device=device, torch=torch)
    return {"model": name, "device": device, "dtype": str(dtype).replace("torch.", ""),
            "sampling_rate": int(model.config.sampling_rate)}


def tts(req):
    torch = STATE["torch"]
    model, tok, desc_tok, device = STATE["model"], STATE["tok"], STATE["desc_tok"], STATE["device"]
    if req.get("seed") is not None:
        torch.manual_seed(int(req["seed"]))
    d = desc_tok(req["description"], return_tensors="pt").to(device)
    p = tok(req["text"], return_tensors="pt").to(device)
    with torch.inference_mode():
        gen = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                             prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
    audio = gen.to(torch.float32).cpu().numpy().squeeze()
    sr = int(model.config.sampling_rate)
    import wave
    import numpy as np
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(req["out"], "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return {"out": req["out"], "sampling_rate": sr, "seconds": round(len(pcm) / sr, 3)}


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
            elif op == "tts":
                reply(ok=True, **tts(req))
            else:
                reply(ok=False, error=f"unknown op {op}")
        except Exception as e:  # report and keep serving
            reply(ok=False, error=f"{type(e).__name__}: {str(e)[:400]}")


if __name__ == "__main__":
    main()
