"""Child process: Wan 2.1 text-to-video, one JSON request per stdin line.

Runs in its own process so the backend never holds the model, and a crash
(CUDA OOM) kills only this child. Protocol (stdout, one JSON per line):
  {"event":"ready"}                       after the model is loaded
  {"event":"step","id":..,"step":n,"of":N}  during a clip
  {"event":"done","id":..,"seconds":..}   a clip was written
  {"event":"error","id":..,"error":..}
Requests: {"id":"S01_0","prompt":..,"negative":..,"frames":81,"steps":25,"seed":7,"out":"path.mp4"}
"""
import json
import os
import sys
import time

MODEL = os.environ.get("STORY_VIDEO_MODEL", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
NEG = "blurry, low quality, distorted face, deformed hands, extra fingers, text, watermark, subtitles, static"


def emit(**kw):
    sys.stdout.write(json.dumps(kw) + "\n")
    sys.stdout.flush()


def main():
    sys.stdout, log = open(os.devnull, "w"), sys.stdout      # library prints go nowhere; protocol on `log`
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    sys.stdout = log
    emit(event="ready", model=MODEL)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("quit"):
            break
        cid = req["id"]
        steps = int(req.get("steps", 25))
        t0 = time.time()
        try:
            def cb(p, i, t, kw):
                emit(event="step", id=cid, step=i + 1, of=steps)
                return kw
            sys.stdout = open(os.devnull, "w")
            try:
                frames = pipe(prompt=req["prompt"], negative_prompt=req.get("negative") or NEG,
                              height=int(req.get("height", 480)), width=int(req.get("width", 832)),
                              num_frames=int(req.get("frames", 81)), num_inference_steps=steps,
                              guidance_scale=float(req.get("guidance", 5.0)),
                              generator=torch.Generator("cuda").manual_seed(int(req.get("seed", 0))),
                              callback_on_step_end=cb).frames[0]
                export_to_video(frames, req["out"], fps=int(req.get("fps", 16)))
            finally:
                sys.stdout = log
            emit(event="done", id=cid, seconds=round(time.time() - t0, 1), out=req["out"])
        except Exception as e:                    # noqa: BLE001 - reported to the parent
            sys.stdout = log
            emit(event="error", id=cid, error=f"{type(e).__name__}: {str(e)[:300]}")
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
