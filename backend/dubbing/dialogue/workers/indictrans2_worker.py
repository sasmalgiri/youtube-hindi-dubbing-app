"""IndicTrans2 English->Hindi worker (own process, possibly its own venv).

Standalone on purpose (no app imports). IndicTrans2's remote code and
IndicTransToolkit break on transformers 5.x; use transformers 4.x (>=4.51 per
the toolkit README; an exact working version must be confirmed on your PC).

Model: ai4bharat/indictrans2-en-indic-dist-200M by default (MIT, gated:
accept conditions + HF_TOKEN); set INDICTRANS2_MODEL for the 1B model.
The maintainers advise sentence-level input; callers send one dialogue turn
per item.

Requests: {"op":"init","model":...} | {"op":"translate","texts":[...]} | {"op":"quit"}
"""
import json
import os
import sys

_PROTO = sys.stdout
sys.stdout = sys.stderr


def reply(**kw):
    _PROTO.write(json.dumps(kw, ensure_ascii=False) + "\n")
    _PROTO.flush()


STATE = {}
SRC, TGT = "eng_Latn", "hin_Deva"


def _processor():
    try:
        from IndicTransToolkit.processor import IndicProcessor
    except ImportError:
        try:
            from IndicTransToolkit import IndicProcessor
        except ImportError:
            from IndicTransToolkit.IndicTransToolkit import IndicProcessor
    return IndicProcessor(inference=True)


def init(req):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    name = req.get("model") or os.environ.get("INDICTRANS2_MODEL",
                                              "ai4bharat/indictrans2-en-indic-dist-200M")
    token = os.environ.get("HF_TOKEN") or None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, token=token)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        name, trust_remote_code=True, token=token,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    STATE.update(tok=tok, model=model, ip=_processor(), device=device, torch=torch)
    return {"model": name, "device": device}


def translate(req, batch_size=16):
    torch = STATE["torch"]
    tok, model, ip, device = STATE["tok"], STATE["model"], STATE["ip"], STATE["device"]
    texts = [t if t.strip() else "." for t in req["texts"]]
    out = []
    for i in range(0, len(texts), batch_size):
        batch = ip.preprocess_batch(texts[i:i + batch_size], src_lang=SRC, tgt_lang=TGT)
        enc = tok(batch, padding="longest", truncation=True, max_length=256,
                  return_tensors="pt").to(device)
        with torch.inference_mode():
            gen = model.generate(**enc, num_beams=5, num_return_sequences=1, max_length=256)
        dec = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        out.extend(ip.postprocess_batch(dec, lang=TGT))
    return {"texts": out}


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
            elif op == "translate":
                reply(ok=True, **translate(req))
            else:
                reply(ok=False, error=f"unknown op {op}")
        except Exception as e:
            reply(ok=False, error=f"{type(e).__name__}: {str(e)[:400]}")


if __name__ == "__main__":
    main()
