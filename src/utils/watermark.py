import cv2
import numpy as np
import os
import subprocess
import logging

logger = logging.getLogger(__name__)


def detect_static_overlay(video_path, sample_count=30, threshold=25):
    """Detect static watermark/logo regions by finding pixels that barely change
    across many sampled frames while the rest of the video moves."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < sample_count * 2:
        cap.release()
        return None

    # Sample frames evenly spread across the video (skip first/last 10%)
    start = int(total_frames * 0.1)
    end = int(total_frames * 0.9)
    indices = np.linspace(start, end, sample_count, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray.astype(np.float32))
    cap.release()

    if len(frames) < 10:
        return None

    # Compute per-pixel standard deviation across sampled frames.
    # Static overlays will have very low std (they don't change).
    stack = np.stack(frames, axis=0)
    std_map = np.std(stack, axis=0)

    # Pixels with very low variation are candidates for static overlay.
    # But we also need to exclude truly static background areas (like black bars).
    mean_map = np.mean(stack, axis=0)

    # Static overlay mask: low variation AND not near-black (>30) AND not near-white (<240)
    # This filters out letterbox bars and plain backgrounds.
    static_mask = (std_map < threshold) & (mean_map > 30) & (mean_map < 240)
    static_mask = static_mask.astype(np.uint8) * 255

    # Clean up noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Only keep regions in the corners/edges (watermarks are typically placed there)
    h, w = static_mask.shape
    corner_mask = np.zeros_like(static_mask)
    margin_h, margin_w = int(h * 0.25), int(w * 0.25)
    # Top-left
    corner_mask[:margin_h, :margin_w] = 255
    # Top-right
    corner_mask[:margin_h, w - margin_w:] = 255
    # Bottom-left
    corner_mask[h - margin_h:, :margin_w] = 255
    # Bottom-right
    corner_mask[h - margin_h:, w - margin_w:] = 255

    static_mask = cv2.bitwise_and(static_mask, corner_mask)

    # Filter out tiny noise regions — keep only blobs with decent area
    contours, _ = cv2.findContours(static_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (h * w) * 0.0005  # at least 0.05% of frame
    filtered_mask = np.zeros_like(static_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

    if cv2.countNonZero(filtered_mask) == 0:
        logger.info("No static watermark detected.")
        return None

    # Dilate the mask a bit to cover edges of the watermark
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    filtered_mask = cv2.dilate(filtered_mask, dilate_kernel, iterations=2)

    logger.info(f"Detected static watermark region ({cv2.countNonZero(filtered_mask)} pixels).")
    return filtered_mask


def remove_watermark(video_path, output_path, mask=None, sample_count=30):
    """Remove static watermark from video using inpainting.

    If mask is None, auto-detects the watermark first.
    """
    if mask is None:
        mask = detect_static_overlay(video_path, sample_count=sample_count)

    if mask is None:
        logger.info("No watermark to remove — copying video as-is.")
        if video_path != output_path:
            import shutil
            shutil.copy2(video_path, output_path)
        return output_path

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write to a temp file, then mux audio back in
    temp_video = output_path + ".temp_noaudio.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

    frame_num = 0
    log_interval = max(1, total_frames // 20)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inpaint the watermark region
        cleaned = cv2.inpaint(frame, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        out.write(cleaned)

        frame_num += 1
        if frame_num % log_interval == 0:
            pct = int(frame_num / total_frames * 100)
            logger.info(f"Watermark removal: {pct}% ({frame_num}/{total_frames} frames)")

    cap.release()
    out.release()

    # Mux original audio back onto the cleaned video
    try:
        subprocess.run([
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', video_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0?',
            output_path
        ], check=True, capture_output=True)
        os.remove(temp_video)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If ffmpeg fails, just rename temp as output (no audio)
        logger.warning("ffmpeg not available — output will have no audio track.")
        os.rename(temp_video, output_path)

    logger.info(f"Watermark removed. Output: {output_path}")
    return output_path
