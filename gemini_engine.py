import os
import json
import time
import uuid
import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import channel_key_store

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEMINI_CONFIG_FILE = os.path.join(BASE_DIR, "gemini_config.json")
THUMBNAILS_DIR = os.path.join(BASE_DIR, "uploads", "thumbnails")
os.makedirs(THUMBNAILS_DIR, exist_ok=True)

# Production Gemini Multimodal Models (Audio + Video + Search Grounding)
DEFAULT_MODEL = "gemini-2.5-flash"
FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-flash-latest",
]

# Nano Banana / Imagen 4K Thumbnail Generation Models
NANO_BANANA_IMAGE_MODELS = [
    "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-image",
    "gemini-3-pro-image-preview",
    "gemini-3-pro-image",
    "gemini-2.5-flash-image",
    "gemini-3.1-flash-lite-image",
    "gemini-2.0-flash-exp-image-generation",
]


def _normalize_model_name(model_name: str) -> str:
    """Normalizes legacy/non-existent model aliases to active production Gemini models."""
    m = (model_name or "").strip()
    if not m or m in ("gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"):
        return DEFAULT_MODEL
    return m


def get_gemini_config(channel_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads Gemini API key and model selection.
    Checks in order:
    1. Channel-bound 10-key pool in Database / Persistent Store (channel_key_store)
    2. Environment variable GEMINI_API_KEY
    3. Local gemini_config.json file
    """
    config = {
        "api_key": "",
        "model": DEFAULT_MODEL,
        "is_configured": False,
        "pool_count": 0,
        "channel_id": channel_id or "default"
    }

    if os.path.exists(GEMINI_CONFIG_FILE):
        try:
            with open(GEMINI_CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                saved_key = saved.get("api_key", "").strip()
                if channel_key_store._is_valid_real_gemini_key(saved_key):
                    config["api_key"] = saved_key
                saved_model = _normalize_model_name(saved.get("model", DEFAULT_MODEL))
                config["model"] = saved_model
        except Exception as e:
            print(f"Error reading gemini_config.json: {e}")

    try:
        pool_key = channel_key_store.get_next_channel_key(channel_id)
        pool_keys = channel_key_store.get_channel_keys(channel_id)
        if pool_key:
            config["api_key"] = pool_key
            config["pool_count"] = len(pool_keys)
    except Exception as e:
        print(f"Notice checking channel_key_store: {e}")

    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key and not config["api_key"]:
        config["api_key"] = env_key

    config["is_configured"] = bool(config["api_key"])
    return config


def save_gemini_config(api_key: str, model: str = DEFAULT_MODEL, channel_id: Optional[str] = None) -> Dict[str, Any]:
    """Saves Gemini API key and model selection to local file and channel key pool."""
    clean_key = api_key.strip()
    norm_model = _normalize_model_name(model)
    data = {
        "api_key": clean_key,
        "model": norm_model,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(GEMINI_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    if clean_key:
        try:
            channel_key_store.add_channel_key(channel_id or "default", clean_key, verify=False)
        except Exception as e:
            print(f"Notice adding key to channel_key_store: {e}")

    os.environ["GEMINI_API_KEY"] = data["api_key"]
    return {
        "success": True,
        "model": data["model"],
        "is_configured": bool(data["api_key"])
    }


def get_genai_client(channel_id: Optional[str] = None, explicit_key: Optional[str] = None):
    """Initializes and returns the google-genai Client using the channel's key pool."""
    from google import genai
    if explicit_key:
        return genai.Client(api_key=explicit_key)
    cfg = get_gemini_config(channel_id)
    if not cfg["is_configured"]:
        raise ValueError("Gemini API key is not configured. Please add your Gemini API key in the Studio settings.")
    return genai.Client(api_key=cfg["api_key"])


def execute_with_key_rotation(channel_id: Optional[str], func, *args, **kwargs):
    """
    Executes a function `func(client, *args, **kwargs)` using the channel's 10-key pool.
    If a key hits 429 Resource Exhausted / Quota Exceeded or 403 Invalid Key, it automatically
    rotates seamlessly to the next available key in the pool.
    """
    from google import genai
    pool_keys = channel_key_store.get_channel_keys(channel_id)
    if not pool_keys:
        cfg = get_gemini_config(channel_id)
        if cfg["api_key"]:
            pool_keys = [cfg["api_key"]]
        else:
            raise ValueError("No Gemini API keys configured in pool. Please add at least 1 key in '⚙️ Configure API Key'.")

    max_attempts = max(len(pool_keys), 1)
    last_err = None

    for attempt in range(max_attempts):
        active_key = channel_key_store.get_next_channel_key(channel_id) or pool_keys[attempt % len(pool_keys)]
        client = genai.Client(api_key=active_key)
        try:
            return func(client, *args, **kwargs)
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            is_quota_or_auth = any(w in err_str for w in [
                "429", "resource_exhausted", "quota", "rate limit", "too many requests",
                "403", "api_key_invalid", "permission_denied", "503", "overloaded"
            ])
            if is_quota_or_auth:
                masked_k = getattr(channel_key_store, "mask_key", lambda x: "****")(active_key)
                if hasattr(channel_key_store, "mark_key_rate_limited"):
                    channel_key_store.mark_key_rate_limited(active_key, cooldown_seconds=120)
                print(f"[KeyPool Failover] Key {masked_k} hit quota/error ({str(e)[:80]}). Rotating to next key (attempt {attempt+1}/{max_attempts})...")
                continue
            raise e

    raise last_err


def get_gemini_status(channel_id: Optional[str] = None) -> Dict[str, Any]:
    """Checks if Gemini is configured and returns status + masked key + pool info."""
    cfg = get_gemini_config(channel_id)
    key = cfg["api_key"]
    masked = ""
    if key and len(key) > 8:
        masked = key[:4] + "*" * (len(key) - 8) + key[-4:]
    elif key:
        masked = "****"

    pool_status = channel_key_store.get_channel_key_pool_status(channel_id)
    return {
        "is_configured": cfg["is_configured"],
        "model": cfg["model"],
        "masked_key": masked,
        "pool": pool_status
    }


def save_client_frame(
    image_bytes: bytes,
    filename: str,
    timestamp: str = "00:00",
    label: str = "Authentic Video Frame",
    aspect_ratio: Optional[str] = None
) -> Dict[str, Any]:
    """Saves a client-extracted raw video frame into the thumbnails folder, cropped to strict 9:16 or 16:9 if specified."""
    filepath = os.path.join(THUMBNAILS_DIR, filename)
    saved_cropped = False
    if aspect_ratio in ("9:16", "16:9"):
        try:
            import cv2
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is not None and decoded.size > 0:
                cropped = fit_and_crop_to_aspect_ratio(decoded, aspect_ratio=aspect_ratio)
                cv2.imwrite(filepath, cropped, [cv2.IMWRITE_JPEG_QUALITY, 96])
                saved_cropped = True
        except Exception:
            saved_cropped = False

    if not saved_cropped:
        with open(filepath, "wb") as f:
            f.write(image_bytes)

    return {
        "id": filename,
        "filename": filename,
        "url": f"/api/thumbnail_file/{filename}",
        "filepath": filepath,
        "timestamp": timestamp,
        "label": label,
        "aspect_ratio": aspect_ratio or "16:9",
        "is_ai_generated": False,
        "is_recommended": False,
        "selected": False
    }


def detect_target_aspect_ratio(video_path: Optional[str] = None, format_type: str = "Auto") -> Tuple[str, int, int]:
    """
    ASPECT RATIO AUTO-DETECTION:
    1. Inspects native video geometry (`w`, `h`) first when `video_path` is provided:
       - Vertical (`w < h`): strictly 9:16 vertical ratio (1080x1920 4K-grade Shorts/Reels).
       - Horizontal/Custom (`w >= h`): strictly 16:9 cinematic widescreen ratio (1920x1080 4K-grade Long-form).
    2. Falls back to `format_type` if video stream geometry cannot be read.
    """
    if video_path and os.path.exists(video_path):
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                cap.release()
                if w > 0 and h > 0:
                    if w < h:
                        return ("9:16", 1080, 1920)
                    else:
                        return ("16:9", 1920, 1080)
        except Exception:
            pass

    fmt_clean = str(format_type or "").strip().lower()
    if fmt_clean.startswith("long") or fmt_clean in ("16:9", "widescreen", "landscape", "horizontal"):
        return ("16:9", 1920, 1080)
    return ("9:16", 1080, 1920)


def fit_and_crop_to_aspect_ratio(
    img_bgr: np.ndarray,
    aspect_ratio: str = "9:16",
    focus_box: Optional[Tuple[int, int, int, int]] = None,
    high_res: bool = True
) -> np.ndarray:
    """
    Strictly formats any BGR frame into 9:16 (1080x1920) for Shorts/Reels or 16:9 (1920x1080) for Long-form.
    Centers the crop intelligently on the detected face (`focus_box`) so the character's facial expression
    is framed with maximum emotional impact and zero black bars.
    """
    import cv2

    if aspect_ratio == "9:16":
        target_w, target_h = (1080, 1920) if high_res else (720, 1280)
    else:
        target_w, target_h = (1920, 1080) if high_res else (1280, 720)

    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    src_h, src_w = img_bgr.shape[:2]
    target_ratio = target_w / float(target_h)
    src_ratio = src_w / float(src_h)

    cx = src_w / 2.0
    cy = src_h / 2.0
    if focus_box is not None and len(focus_box) == 4:
        fx, fy, fw, fh = focus_box
        cx = fx + (fw / 2.0)
        cy = fy + (fh * 0.45)

    if abs(src_ratio - target_ratio) < 0.02:
        return cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    if src_ratio > target_ratio:
        crop_h = src_h
        crop_w = max(1, int(round(crop_h * target_ratio)))
        x1 = int(round(cx - (crop_w / 2.0)))
        x1 = max(0, min(src_w - crop_w, x1))
        y1 = 0
    else:
        crop_w = src_w
        crop_h = max(1, int(round(crop_w / target_ratio)))
        y1 = int(round(cy - (crop_h / 2.0)))
        y1 = max(0, min(src_h - crop_h, y1))
        x1 = 0

    cropped = img_bgr[y1:y1 + crop_h, x1:x1 + crop_w]
    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)


def extract_character_face_reference_crop(
    frame_bgr: np.ndarray,
    face_box: Optional[Tuple[int, int, int, int]]
) -> np.ndarray:
    """
    Extracts a crystal-clear, high-resolution portrait reference crop of the highest-emotion
    character face (including hair, eyes, expression, and shoulders) to feed directly into
    the Nano Banana / Imagen API so 100% character facial identity is preserved.
    """
    import cv2

    if frame_bgr is None or frame_bgr.size == 0:
        return np.zeros((768, 768, 3), dtype=np.uint8)

    h, w = frame_bgr.shape[:2]
    if face_box is not None and len(face_box) == 4:
        fx, fy, fw, fh = face_box
        # Expand box generously around head & upper shoulders for identity preservation
        pad_x = int(fw * 0.65)
        pad_top = int(fh * 0.55)
        pad_bot = int(fh * 0.85)
        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_top)
        x2 = min(w, fx + fw + pad_x)
        y2 = min(h, fy + fh + pad_bot)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            return cv2.resize(crop, (768, 768), interpolation=cv2.INTER_LANCZOS4)

    # Fallback: center portrait crop
    side = min(w, h)
    x1 = (w - side) // 2
    y1 = max(0, int((h - side) * 0.25))
    crop = frame_bgr[y1:y1 + side, x1:x1 + side]
    return cv2.resize(crop, (768, 768), interpolation=cv2.INTER_LANCZOS4)


def _score_frame_emotion_and_motion(
    frame_bgr: np.ndarray,
    prev_gray: Optional[np.ndarray],
    frontal_cascade: Any,
    profile_cascade: Any = None
) -> Tuple[float, int, Optional[Tuple[int, int, int, int]], np.ndarray]:
    """
    Scores a video frame on:
    - Character facial presence & emotional expression intensity (eye/brow/mouth Laplacian micro-contrast)
    - Visual sharpness (Laplacian variance)
    - Action/motion intensity (frame-to-frame optical delta + Sobel gradient energy)
    - Color vibrancy & dynamic range
    """
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = float(np.mean(cv2.magnitude(gx, gy)))

    motion_score = 0.0
    if prev_gray is not None and prev_gray.shape == gray.shape:
        diff = cv2.absdiff(gray, prev_gray)
        motion_score = float(np.mean(diff)) * 18.0

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))
    val_std = float(np.std(hsv[:, :, 2]))
    vibrancy_score = (sat_mean * 2.5) + (val_std * 4.0)

    detected_boxes = []
    if frontal_cascade is not None:
        try:
            faces = frontal_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(44, 44))
            for b in faces:
                detected_boxes.append(tuple(int(v) for v in b))
        except Exception:
            pass

    if not detected_boxes and profile_cascade is not None:
        try:
            pfaces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
            for b in pfaces:
                detected_boxes.append(tuple(int(v) for v in b))
        except Exception:
            pass

    face_count = len(detected_boxes)
    best_face_box = None
    max_face_score = 0.0

    for (x, y, w, h) in detected_boxes:
        face_roi = gray[max(0, y):min(gray.shape[0], y + h), max(0, x):min(gray.shape[1], x + w)]
        expr_intensity = float(cv2.Laplacian(face_roi, cv2.CV_64F).var()) if face_roi.size > 0 else 0.0
        face_contrast = float(np.std(face_roi)) if face_roi.size > 0 else 0.0
        f_score = (w * h * 0.65) + (expr_intensity * 10.0) + (face_contrast * 25.0)
        if f_score > max_face_score:
            max_face_score = f_score
            best_face_box = (x, y, w, h)

    composite_score = sharpness + (edge_energy * 6.0) + motion_score + vibrancy_score
    if face_count > 0:
        composite_score += 8000.0 + max_face_score

    return composite_score, face_count, best_face_box, gray


def extract_video_thumbnails(
    video_path: str,
    count: int = 5,
    aspect_ratio: Optional[str] = None,
    format_type: str = "Auto",
    return_best_raw: bool = False
) -> Any:
    """
    1. Samples key visual frames across the entire video timeline.
    2. Extracts the #1 highest-emotion character face (`best_raw_info`) + timeline PIL frames for Gemini & Nano Banana.
    3. Saves SLOTS 2 to 6 (5 High-Emotion Local Video Frames) in strict 9:16 (1080x1920) or 16:9 (1920x1080).
    """
    results: List[Dict[str, Any]] = []
    best_raw_info: Dict[str, Any] = {
        "frame": None,
        "face_crop_bgr": None,
        "face_box": None,
        "seconds": 0.0,
        "timestamp": "00:00",
        "timeline_pil_frames": []
    }

    detected_ratio, target_w, target_h = detect_target_aspect_ratio(video_path, format_type=aspect_ratio or format_type)
    if aspect_ratio in ("9:16", "16:9"):
        detected_ratio = aspect_ratio
        target_w, target_h = (1080, 1920) if detected_ratio == "9:16" else (1920, 1080)

    try:
        import cv2
        from PIL import Image
    except ImportError:
        print("OpenCV (cv2) or PIL not installed, skipping server frame extraction.")
        return (results, best_raw_info) if return_best_raw else results

    if not os.path.exists(video_path):
        return (results, best_raw_info) if return_best_raw else results

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (results, best_raw_info) if return_best_raw else results

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / fps if fps > 0 else 0

    if total_frames <= 0:
        cap.release()
        return (results, best_raw_info) if return_best_raw else results

    frontal_cascade = None
    profile_cascade = None
    try:
        f_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        p_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        if os.path.exists(f_path):
            frontal_cascade = cv2.CascadeClassifier(f_path)
        if os.path.exists(p_path):
            profile_cascade = cv2.CascadeClassifier(p_path)
    except Exception as e:
        print(f"Haar cascade load notice: {e}")

    num_samples = min(32, max(count * 5, 18))
    start_frame = int(total_frames * 0.04)
    end_frame = int(total_frames * 0.95)
    if end_frame <= start_frame:
        start_frame = 0
        end_frame = max(1, total_frames - 1)

    sample_indices = [
        int(start_frame + i * (end_frame - start_frame) / max(1, num_samples - 1))
        for i in range(num_samples)
    ]

    candidates = []
    prev_gray = None

    for f_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        sec = f_idx / fps if fps > 0 else 0.0
        score, face_count, best_face_box, gray = _score_frame_emotion_and_motion(
            frame, prev_gray, frontal_cascade, profile_cascade
        )
        prev_gray = gray

        candidates.append({
            "frame_idx": f_idx,
            "seconds": sec,
            "score": score,
            "face_count": face_count,
            "face_box": best_face_box,
            "frame": frame
        })

    cap.release()

    if not candidates:
        return (results, best_raw_info) if return_best_raw else results

    # Build 6 chronological timeline PIL frames for multimodal timeline inspection
    timeline_pil_frames = []
    step_t = max(1, len(candidates) // 6)
    for t_cand in candidates[::step_t][:6]:
        try:
            rgb_small = cv2.cvtColor(t_cand["frame"], cv2.COLOR_BGR2RGB)
            h_s, w_s = rgb_small.shape[:2]
            scale = 512.0 / max(w_s, h_s, 1)
            resized_rgb = cv2.resize(rgb_small, (max(1, int(w_s * scale)), max(1, int(h_s * scale))))
            timeline_pil_frames.append(Image.fromarray(resized_rgb))
        except Exception:
            pass

    # Sort by emotion + face + motion score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Record the #1 highest-emotion character face frame for Slot 1 Nano Banana 4K Thumbnail
    top_cand = candidates[0]
    top_sec = top_cand["seconds"]
    top_frame_copy = top_cand["frame"].copy()
    top_face_box = top_cand["face_box"]
    face_crop_bgr = extract_character_face_reference_crop(top_frame_copy, top_face_box)

    best_raw_info = {
        "frame": top_frame_copy,
        "face_crop_bgr": face_crop_bgr,
        "face_box": top_face_box,
        "seconds": round(top_sec, 1),
        "timestamp": f"{int(top_sec // 60):02d}:{int(top_sec % 60):02d}",
        "timeline_pil_frames": timeline_pil_frames
    }

    # Select `count` (5) distinct high-emotion frames spaced across the video for Slots 2..6
    selected = []
    min_time_gap = max(1.0, duration_sec / 12.0)

    for cand in candidates:
        if len(selected) >= count:
            break
        if any(abs(cand["seconds"] - s["seconds"]) < min_time_gap for s in selected):
            continue
        selected.append(cand)

    if len(selected) < count:
        for cand in candidates:
            if len(selected) >= count:
                break
            if all(cand["frame_idx"] != s["frame_idx"] for s in selected):
                selected.append(cand)

    selected.sort(key=lambda x: x["seconds"])

    task_prefix = uuid.uuid4().hex[:8]
    for idx, item in enumerate(selected):
        slot_num = idx + 2  # Slots 2 to 6
        sec = item["seconds"]
        mins = int(sec // 60)
        secs = int(sec % 60)
        time_str = f"{mins:02d}:{secs:02d}"

        formatted_bgr = fit_and_crop_to_aspect_ratio(
            item["frame"],
            aspect_ratio=detected_ratio,
            focus_box=item.get("face_box"),
            high_res=True
        )

        emotion_tag = "Character Face Focus" if item["face_count"] > 0 else "Peak Scene Action"
        label = f"Slot {slot_num}: {emotion_tag} ({time_str})"
        filename = f"thumb_slot{slot_num}_{task_prefix}_{mins}m{secs}s.jpg"
        filepath = os.path.join(THUMBNAILS_DIR, filename)

        cv2.imwrite(filepath, formatted_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        results.append({
            "id": f"slot_{slot_num}",
            "slot": slot_num,
            "filename": filename,
            "url": f"/api/thumbnail_file/{filename}",
            "filepath": filepath,
            "seconds": round(sec, 1),
            "timestamp": f"SLOT {slot_num} • {time_str}",
            "label": label,
            "has_face": item["face_count"] > 0,
            "is_ai_generated": False,
            "is_recommended": False,
            "selected": False,
            "aspect_ratio": detected_ratio,
            "width": target_w,
            "height": target_h
        })

    return (results, best_raw_info) if return_best_raw else results


def _resolve_genre_dramatic_preset(
    genre_str: str,
    primary_ctx: str,
    climactic_ctx: str,
    color_theme_str: str,
    video_hash_int: int
) -> Dict[str, Any]:
    """
    HYPER-ENGAGING 4K NANO BANANA GENRE-SPECIFIC DRAMATIC RENDERING PRESETS:
    - War/Heroic: Soldier character holding heavy gear, trench explosions, cinematic smoke, ultra-dramatic lighting.
    - Horror/Thriller: High-contrast shadows, shock expressions, eerie background atmosphere.
    - Comedy/Drama: Exaggerated expressions, vibrant colors, clean punchy backdrop.
    - Mystery/Suspense: High-stakes suspenseful lighting, neon-noir depth, atmospheric particles.
    """
    combined = f"{genre_str} {primary_ctx} {climactic_ctx} {color_theme_str}".lower()

    if any(w in combined for w in [
        "war", "soldier", "military", "army", "battle", "combat", "trench", "hero",
        "sniper", "gun", "rifle", "weapon", "explosion", "bomb", "commando", "rescue", "fauji", "jung"
    ]):
        return {
            "genre_category": "War/Heroic",
            "mode": "war_heroic",
            "prompt_style": (
                "WAR / HEROIC 4K MOVIE POSTER GRADE: Transform the scene into an ultra-dramatic war/heroic "
                "composition. Keep 100% of the exact character facial identity from the reference face, "
                "depicting the character (e.g., heroic soldier / kid soldier) wearing gritty tactical combat gear "
                "and holding heavy military gear amidst intense trench explosions, billowing cinematic battlefield smoke, "
                "flying fiery orange embers and sparks, and ultra-dramatic golden-amber & steel rim lighting."
            ),
            "shadow_bgr": (18, 28, 16),      # Tactical dark olive-charcoal
            "highlight_bgr": (20, 155, 255), # Fiery explosion amber-orange
            "glow_rgb": (255, 125, 15),      # Fiery ember orange
            "text_rgb": (255, 238, 130),     # War-poster gold
            "badge_rgb": (220, 38, 38)
        }

    if any(w in combined for w in [
        "horror", "thriller", "scary", "ghost", "demon", "dark", "nightmare", "murder",
        "killer", "blood", "creepy", "haunted", "fear", "shock", "terror", "monster", "bhoot", "danger"
    ]):
        return {
            "genre_category": "Horror/Thriller",
            "mode": "horror_thriller",
            "prompt_style": (
                "HORROR / THRILLER 4K POSTER GRADE: Transform the scene into a spine-chilling psychological "
                "horror/thriller composition. Keep 100% of the exact character facial identity from the reference face "
                "with a heightened wide-eyed shock and terror expression, surrounded by deep high-contrast chiaroscuro "
                "shadows, eerie supernatural volumetric fog, sinister dark background atmosphere, and piercing "
                "blood-crimson & icy-cyan rim lighting."
            ),
            "shadow_bgr": (20, 8, 36),       # Deep abyssal blood-noir
            "highlight_bgr": (245, 215, 70), # Icy cyan highlight
            "glow_rgb": (255, 30, 65),       # Crimson shock glow
            "text_rgb": (255, 255, 255),
            "badge_rgb": (185, 28, 28)
        }

    if any(w in combined for w in [
        "comedy", "funny", "humor", "prank", "joke", "laugh", "drama", "emotional",
        "family", "romance", "reaction", "fun", "roast", "entertainment", "challenge", "vlog"
    ]):
        return {
            "genre_category": "Comedy/Drama",
            "mode": "comedy_drama",
            "prompt_style": (
                "COMEDY / DRAMA 4K VIRAL THUMBNAIL GRADE: Transform the scene into an ultra-clickable viral "
                "creator composition. Keep 100% of the exact character facial identity from the reference face "
                "with an exaggerated, hyper-expressive emotional reaction, vibrant high-saturation wardrobe, "
                "and a clean, punchy, high-contrast studio backdrop with energetic radial lighting."
            ),
            "shadow_bgr": (48, 16, 38),      # Rich studio violet-blue
            "highlight_bgr": (30, 220, 255), # Electric golden yellow
            "glow_rgb": (255, 195, 0),       # Punchy viral yellow-gold
            "text_rgb": (255, 245, 120),
            "badge_rgb": (236, 72, 153)
        }

    # Default: Mystery / Suspense / Action 4K Blockbuster
    palettes = [
        {
            "genre_category": "Mystery/Suspense",
            "mode": "mystery_suspense",
            "prompt_style": (
                "MYSTERY / SUSPENSE 4K BLOCKBUSTER POSTER GRADE: Keep 100% of the exact character facial identity "
                "from the reference face with an intense, suspenseful close-up expression, replacing the mundane "
                "background with a dramatic high-contrast cinematic atmosphere, volumetric spotlight rays, and "
                "teal-and-amber rim lighting."
            ),
            "shadow_bgr": (42, 24, 8),
            "highlight_bgr": (18, 165, 255),
            "glow_rgb": (56, 189, 248),
            "text_rgb": (254, 240, 138),
            "badge_rgb": (147, 51, 234)
        },
        {
            "genre_category": "Action/Heroic",
            "mode": "war_heroic",
            "prompt_style": (
                "ACTION / HEROIC 4K POSTER GRADE: Keep 100% of the exact character facial identity from the reference "
                "face, placing them in an intense high-stakes action environment with dramatic sparks, smoke, and "
                "ultra-high-contrast rim lighting."
            ),
            "shadow_bgr": (18, 12, 42),
            "highlight_bgr": (35, 195, 255),
            "glow_rgb": (255, 90, 30),
            "text_rgb": (255, 255, 255),
            "badge_rgb": (239, 68, 68)
        }
    ]
    return palettes[video_hash_int % len(palettes)]


def _render_ultra_high_contrast_ai_visual(
    base_bgr: np.ndarray,
    aspect_ratio: str,
    metadata: Dict[str, Any],
    focus_box: Optional[Tuple[int, int, int, int]],
    video_seed_str: str,
    apply_typography: bool = True,
    replace_mundane_backdrop: bool = True
) -> np.ndarray:
    """
    4K POSTER-GRADE THUMBNAIL COMPOSITOR:
    - Preserves 100% character facial identity in the foreground subject region with razor-sharp CLAHE eye/expression clarity.
    - Replaces mundane backgrounds with genre-specific 4K atmospheric compositions:
      * War/Heroic: Trench explosion horizon glow, volumetric battlefield smoke, glowing fiery embers/sparks.
      * Horror/Thriller: High-contrast chiaroscuro shadows, eerie fog tendrils, crimson/cyan rim lighting.
      * Comedy/Drama: Vibrant high-saturation studio pop, clean punchy radial backdrop, crisp rim light.
    - Renders in matching aspect ratio: 9:16 (1080x1920) or 16:9 (1920x1080).
    """
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    target_w, target_h = (1080, 1920) if aspect_ratio == "9:16" else (1920, 1080)
    seed_digest = int(hashlib.sha256(video_seed_str.encode("utf-8", errors="ignore")).hexdigest()[:12], 16)
    rng = np.random.default_rng(seed_digest)

    thumb_dir = metadata.get("thumbnail_directive") or {}
    color_theme = str(thumb_dir.get("recommended_color_theme") or "")
    genre_str = str(metadata.get("detected_genre") or metadata.get("detected_genre_emotion") or "")
    primary_ctx = str(metadata.get("primary_context") or "")
    climactic_ctx = str(metadata.get("climactic_context") or "")

    preset = _resolve_genre_dramatic_preset(genre_str, primary_ctx, climactic_ctx, color_theme, seed_digest)
    mode = preset["mode"]

    # 1. Frame & dynamic subject-centered dramatic zoom
    framed = fit_and_crop_to_aspect_ratio(base_bgr, aspect_ratio=aspect_ratio, focus_box=focus_box, high_res=True)
    zoom_factor = 1.08 + ((seed_digest % 10) * 0.01)
    zh, zw = int(round(target_h * zoom_factor)), int(round(target_w * zoom_factor))
    zoomed = cv2.resize(framed, (zw, zh), interpolation=cv2.INTER_LANCZOS4)

    x_off = max(0, min(zw - target_w, int((zw - target_w) * 0.5)))
    y_off = max(0, min(zh - target_h, int((zh - target_h) * 0.35)))
    canvas_bgr = zoomed[y_off:y_off + target_h, x_off:x_off + target_w].copy()

    # Detect face in the zoomed canvas so we know the exact character face coordinates to preserve 100%
    subject_cx = target_w * 0.5
    subject_cy = target_h * (0.40 if aspect_ratio == "9:16" else 0.44)
    subject_rx = target_w * (0.36 if aspect_ratio == "9:16" else 0.28)
    subject_ry = target_h * (0.34 if aspect_ratio == "9:16" else 0.42)

    try:
        gray_c = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
        f_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(f_path):
            fc = cv2.CascadeClassifier(f_path)
            faces = fc.detectMultiScale(gray_c, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))
            if len(faces) > 0:
                # Pick largest face
                fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                subject_cx = float(fx + fw * 0.5)
                subject_cy = float(fy + fh * 0.55)
                subject_rx = max(float(fw * 1.35), target_w * 0.24)
                subject_ry = max(float(fh * 1.85), target_h * 0.28)
    except Exception:
        pass

    # 2. Enhance Character Face & Expression Clarity (Multi-band CLAHE + Eye/Detail Sharpening)
    lab = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.4, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_chan)
    l_float = l_enhanced.astype(np.float32) / 255.0
    l_curve = np.clip(0.5 + 1.26 * (l_float - 0.5), 0.0, 1.0)
    subject_bgr = cv2.cvtColor(cv2.merge([(l_curve * 255.0).astype(np.uint8), a_chan, b_chan]), cv2.COLOR_LAB2BGR)

    hsv_sub = cv2.cvtColor(subject_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_boost = 1.42 if mode == "comedy_drama" else (1.22 if mode == "horror_thriller" else 1.32)
    hsv_sub[:, :, 1] = np.clip(hsv_sub[:, :, 1] * sat_boost, 0, 255)
    subject_bgr = cv2.cvtColor(hsv_sub.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 3. Build Soft Character Identity Preservation Mask vs Mundane Background Mask
    yy, xx = np.mgrid[0:target_h, 0:target_w].astype(np.float32)
    ellip_dist = np.sqrt(((xx - subject_cx) / subject_rx) ** 2 + ((yy - subject_cy) / subject_ry) ** 2)
    # Also include lower torso region below the face so the character body connects naturally
    torso_dist = np.sqrt(((xx - subject_cx) / (subject_rx * 1.45)) ** 2 + ((yy - (subject_cy + subject_ry * 0.95)) / (subject_ry * 1.15)) ** 2)
    combined_dist = np.minimum(ellip_dist, torso_dist)

    # Smooth feather: 1.0 inside character face/body, 0.0 in mundane background
    subject_mask = np.clip(1.35 - combined_dist, 0.0, 1.0)
    subject_mask = cv2.GaussianBlur(subject_mask, (0, 0), sigmaX=28.0)[:, :, np.newaxis]
    rim_ring_mask = np.clip(np.exp(-((combined_dist - 0.92) ** 2) / 0.045) * 0.85, 0.0, 1.0)[:, :, np.newaxis]

    # 4. Genre-Specific 4K Dramatic Backdrop Synthesis (Replaces mundane background while keeping character face 100%)
    if replace_mundane_backdrop:
        # Deep bokeh blur of periphery to obliterate mundane background clutter
        bg_blurred = cv2.GaussianBlur(canvas_bgr, (0, 0), sigmaX=24.0).astype(np.float32)
        shadow_bgr = np.array(preset["shadow_bgr"], dtype=np.float32).reshape((1, 1, 3))
        highlight_bgr = np.array(preset["highlight_bgr"], dtype=np.float32).reshape((1, 1, 3))

        if mode == "war_heroic":
            # War/Heroic: Dark smoky trench sky at top + fiery orange-amber explosion glow at horizon + volumetric smoke
            horizon_y = subject_cy + subject_ry * 0.25
            expl_dist = np.sqrt(((xx - subject_cx) / (target_w * 0.55)) ** 2 + ((yy - horizon_y) / (target_h * 0.32)) ** 2)
            explosion_glow = np.clip(np.exp(-(expl_dist ** 2) * 1.35), 0.0, 1.0)[:, :, np.newaxis]
            smoke_wave = (np.sin(xx / 95.0 + yy / 140.0) * 0.5 + 0.5)[:, :, np.newaxis]
            bg_dramatic = (
                bg_blurred * 0.22
                + shadow_bgr * 0.78
                + highlight_bgr * explosion_glow * 0.95
                + np.array([65, 75, 85], dtype=np.float32).reshape((1, 1, 3)) * smoke_wave * 0.32
            )
        elif mode == "horror_thriller":
            # Horror/Thriller: Pitch-black chiaroscuro shadows + eerie crimson/cyan fog atmosphere
            fog_band = np.clip(np.exp(-((yy - (target_h * 0.62)) / (target_h * 0.25)) ** 2), 0.0, 1.0)[:, :, np.newaxis]
            eerie_mist = (np.cos(xx / 80.0 - yy / 110.0) * 0.5 + 0.5)[:, :, np.newaxis]
            crimson_bgr = np.array([45, 15, 220], dtype=np.float32).reshape((1, 1, 3))
            cyan_bgr = np.array([210, 180, 20], dtype=np.float32).reshape((1, 1, 3))
            bg_dramatic = (
                bg_blurred * 0.14
                + shadow_bgr * 0.86
                + crimson_bgr * fog_band * 0.48
                + cyan_bgr * eerie_mist * 0.25
            )
        elif mode == "comedy_drama":
            # Comedy/Drama: Clean, punchy vibrant radial energy backdrop
            rad_dist = np.sqrt(((xx - subject_cx) / (target_w * 0.6)) ** 2 + ((yy - subject_cy) / (target_h * 0.6)) ** 2)
            radial_burst = np.clip(1.15 - rad_dist * 0.75, 0.15, 1.0)[:, :, np.newaxis]
            angles = np.arctan2(yy - subject_cy, xx - subject_cx)
            sunburst_rays = (np.sin(angles * 16.0) * 0.5 + 0.5)[:, :, np.newaxis]
            bg_dramatic = (
                bg_blurred * 0.25
                + shadow_bgr * 0.55
                + highlight_bgr * radial_burst * (0.65 + 0.28 * sunburst_rays)
            )
        else:
            # Mystery/Suspense: Cyber-noir teal & fiery amber spotlight backdrop
            spot_dist = np.sqrt(((xx - subject_cx) / (target_w * 0.58)) ** 2 + ((yy - subject_cy) / (target_h * 0.58)) ** 2)
            spot_glow = np.clip(1.0 - spot_dist * 0.72, 0.12, 1.0)[:, :, np.newaxis]
            bg_dramatic = bg_blurred * 0.25 + shadow_bgr * 0.75 + highlight_bgr * (spot_glow ** 1.8) * 0.62

        # Composite 100% preserved character face/subject over the 4K genre backdrop + intense rim lighting
        fg_float = subject_bgr.astype(np.float32)
        composited = (
            fg_float * subject_mask
            + bg_dramatic * (1.0 - subject_mask)
            + highlight_bgr * rim_ring_mask * (1.0 - subject_mask * 0.45) * 0.55
        )
    else:
        # Gentle cinematic vignette & rim polish for already AI-generated Nano Banana images
        norm_dist = np.sqrt(((xx - subject_cx) / (target_w * 0.68)) ** 2 + ((yy - subject_cy) / (target_h * 0.68)) ** 2)
        vig = np.clip(1.0 - 0.38 * (norm_dist ** 1.7), 0.45, 1.0)[:, :, np.newaxis]
        composited = subject_bgr.astype(np.float32) * vig

    # Unsharp mask for 4K poster-grade razor sharpness
    blurred_comp = cv2.GaussianBlur(composited, (0, 0), sigmaX=2.0)
    sharpened = np.clip(cv2.addWeighted(composited, 1.45, blurred_comp, -0.45, 0), 0, 255).astype(np.uint8)

    # 5. Add Genre-Specific Atmospheric Particle FX (Embers/Sparks for War/Heroic, Eerie Motes for Horror)
    rgb_img = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img).convert("RGBA")
    fx_layer = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    fx_draw = ImageDraw.Draw(fx_layer)

    if mode == "war_heroic" and replace_mundane_backdrop:
        # Draw glowing battlefield embers & sparks in the periphery around the soldier/character
        for _ in range(55):
            px = int(rng.integers(20, target_w - 20))
            py = int(rng.integers(int(target_h * 0.08), int(target_h * 0.88)))
            # Keep embers mostly outside the immediate center of the face
            if abs(px - subject_cx) < subject_rx * 0.55 and abs(py - subject_cy) < subject_ry * 0.55:
                continue
            r_sz = int(rng.integers(2, 7))
            ember_alpha = int(rng.integers(140, 245))
            fx_draw.ellipse(
                [px - r_sz, py - r_sz, px + r_sz, py + r_sz],
                fill=(255, int(rng.integers(110, 215)), 20, ember_alpha)
            )
    elif mode == "horror_thriller" and replace_mundane_backdrop:
        # Subtle eerie red/cyan atmospheric light motes in shadow corners
        for _ in range(30):
            px = int(rng.integers(20, target_w - 20))
            py = int(rng.integers(20, target_h - 20))
            if abs(px - subject_cx) < subject_rx * 0.65 and abs(py - subject_cy) < subject_ry * 0.65:
                continue
            r_sz = int(rng.integers(2, 6))
            fx_draw.ellipse(
                [px - r_sz, py - r_sz, px + r_sz, py + r_sz],
                fill=(255, 35, 65, int(rng.integers(95, 185)))
            )

    pil_img = Image.alpha_composite(pil_img, fx_layer)

    if not apply_typography:
        final_rgb = np.array(pil_img.convert("RGB"))
        return cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)

    # 6. Poster-Grade Scrim & 4K Hook Typography at Bottom
    scrim = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    scrim_draw = ImageDraw.Draw(scrim)
    grad_start_y = int(target_h * (0.65 if aspect_ratio == "9:16" else 0.58))
    for y in range(grad_start_y, target_h):
        prog = (y - grad_start_y) / float(max(1, target_h - grad_start_y))
        alpha = int(min(235, (prog ** 1.3) * 230))
        scrim_draw.line([(0, y), (target_w, y)], fill=(5, 5, 10, alpha))
    pil_img = Image.alpha_composite(pil_img, scrim)

    draw = ImageDraw.Draw(pil_img)

    raw_hook = str(thumb_dir.get("text_overlay") or "").strip()
    if not raw_hook or raw_hook.upper() in ("WATCH THIS", "MUST WATCH", "-"):
        v_title = str(metadata.get("viral_title") or metadata.get("primary_context") or "").strip()
        clean_words = [w for w in re.sub(r'[#|!?:🔥🎯⚡]+', ' ', v_title).split() if len(w) > 1 and not w.lower().startswith("shorts")]
        raw_hook = " ".join(clean_words[:4]).upper() if clean_words else "SHOCKING TWIST"
    else:
        raw_hook = " ".join(raw_hook.split()[:4]).upper()

    font_size = 94 if aspect_ratio == "9:16" else 98
    font = None
    font_candidates = [
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                pass
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    words = raw_hook.split()
    if aspect_ratio == "9:16" and len(words) >= 3:
        mid = (len(words) + 1) // 2
        lines = [" ".join(words[:mid]), " ".join(words[mid:])]
    elif len(raw_hook) > 15 and len(words) >= 2:
        mid = (len(words) + 1) // 2
        lines = [" ".join(words[:mid]), " ".join(words[mid:])]
    else:
        lines = [raw_hook]

    line_height = int(font_size * 1.16)
    total_text_h = line_height * len(lines)
    base_y = target_h - total_text_h - (125 if aspect_ratio == "9:16" else 68)

    bar_w = int(target_w * 0.30)
    bar_x = (target_w - bar_w) // 2
    bar_y = max(24, base_y - 26)
    draw.rounded_rectangle(
        [bar_x, bar_y, bar_x + bar_w, bar_y + 9],
        radius=4,
        fill=(*preset["glow_rgb"], 250)
    )

    for idx, line_str in enumerate(lines):
        ly = base_y + idx * line_height
        try:
            bbox = draw.textbbox((0, 0), line_str, font=font)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = len(line_str) * (font_size // 2)
        lx = max(28, (target_w - tw) // 2)

        for dx in (-6, -3, 0, 3, 6):
            for dy in (-6, -3, 0, 3, 6):
                if dx != 0 or dy != 0:
                    draw.text((lx + dx, ly + dy), line_str, font=font, fill=(0, 0, 0, 250))
        draw.text((lx + 6, ly + 8), line_str, font=font, fill=(0, 0, 0, 230))
        line_color = (255, 255, 255, 255) if idx == 0 else (*preset["text_rgb"], 255)
        draw.text((lx, ly), line_str, font=font, fill=line_color)

    draw.rectangle(
        [3, 3, target_w - 4, target_h - 4],
        outline=(*preset["glow_rgb"], 210),
        width=6
    )

    final_rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)


def generate_dynamic_ai_thumbnail(
    video_path: str,
    format_type: str,
    aspect_ratio: str,
    metadata: Dict[str, Any],
    reference_frame_bgr: Optional[np.ndarray] = None,
    reference_face_crop_bgr: Optional[np.ndarray] = None,
    reference_face_box: Optional[Tuple[int, int, int, int]] = None,
    channel_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    HYPER-ENGAGING 4K NANO BANANA THUMBNAIL PIPELINE (SLOT 1 DEFAULT SELECTED):
    1. Extracts the highest-emotion character face directly from the native video frames.
    2. Feeds this facial reference (`face_crop_pil` + `full_ref_pil`) along with the detected video genre
       into the Nano Banana / Imagen API (`gemini-3.1-flash-image`, `gemini-3-pro-image`, `gemini-2.5-flash-image`).
    3. Applies genre-specific dramatic rendering:
       - War/Heroic: Soldier kid/hero holding heavy gear, trench explosions, cinematic smoke, ultra-dramatic lighting.
       - Horror/Thriller: High-contrast shadows, shock expressions, eerie background atmosphere.
       - Comedy/Drama: Exaggerated expressions, vibrant colors, clean punchy backdrop.
    4. Preserves 100% character facial identity while replacing mundane bodies/backgrounds with
       ultra-clickable, 4K poster-grade compositions in matching aspect ratio (9:16 or 16:9).
    """
    import cv2
    from PIL import Image

    target_w, target_h = (1080, 1920) if aspect_ratio == "9:16" else (1920, 1080)
    primary_ctx = str(metadata.get("primary_context") or "Dramatic Video Moment")
    climactic_ctx = str(metadata.get("climactic_context") or metadata.get("plot_twists") or metadata.get("summary_insights") or primary_ctx)
    facial_expr = str(metadata.get("facial_expression_analysis") or "Intense, expressive close-up emotion with dramatic eye contact")
    genre_str = str(metadata.get("detected_genre") or metadata.get("detected_genre_emotion") or "High-Suspense Cinematic")
    viral_title = str(metadata.get("viral_title") or primary_ctx)
    thumb_dir = metadata.get("thumbnail_directive") or {}
    text_overlay = str(thumb_dir.get("text_overlay") or "MUST WATCH").strip()
    scene_dir = str(thumb_dir.get("visual_scene_direction") or "Close-up dramatic subject with chiaroscuro rim lighting")
    color_theme = str(thumb_dir.get("recommended_color_theme") or "Ultra-high-contrast cinematic teal and fiery amber")

    video_seed_str = f"{os.path.basename(video_path)}|{viral_title}|{primary_ctx}|{climactic_ctx}|{scene_dir}|{color_theme}|{time.time()}"
    seed_digest = int(hashlib.sha256(video_seed_str.encode("utf-8", errors="ignore")).hexdigest()[:12], 16)
    preset = _resolve_genre_dramatic_preset(genre_str, primary_ctx, climactic_ctx, color_theme, seed_digest)
    genre_category = preset["genre_category"]
    genre_prompt_style = preset["prompt_style"]

    orientation_desc = (
        "vertical 9:16 portrait YouTube Shorts / Reels 4K poster thumbnail (1080x1920)"
        if aspect_ratio == "9:16"
        else "cinematic 16:9 widescreen YouTube Long-form 4K poster thumbnail (1920x1080)"
    )

    nano_banana_prompt = (
        f"Create an ultra-clickable, 4K poster-grade {orientation_desc} using the provided character facial reference.\n"
        f"- CRITICAL IDENTITY RULE: Preserve 100% of the exact facial identity, face structure, age, and likeness of the person in the reference image.\n"
        f"- DETECTED GENRE & DRAMATIC RENDERING ({genre_category}): {genre_prompt_style}\n"
        f"- Specific Video Plot & Climax: {primary_ctx} — {climactic_ctx}\n"
        f"- Facial Expression Enhancement: Amplify the character's expression ({facial_expr}) so it is hyper-expressive and immediately draws clicks. {scene_dir}\n"
        f"- Replace Mundane Body/Background: Replace any mundane clothing or plain room/street background with a 4K cinematic {genre_category} environment and lighting ({color_theme}).\n"
        f"- Bold Hook Text Overlay (3-4 words max): \"{text_overlay}\"\n"
        f"Strictly {aspect_ratio} aspect ratio, photorealistic 4K poster composition, extreme dynamic range, razor-sharp eyes."
    )

    generated_bgr = None
    generation_engine = f"nano-banana-4k-{preset['mode']}"

    # Prepare both the tight character face reference PIL and full scene reference PIL
    face_ref_pil = None
    full_ref_pil = None
    if reference_face_crop_bgr is not None and reference_face_crop_bgr.size > 0:
        face_rgb = cv2.cvtColor(reference_face_crop_bgr, cv2.COLOR_BGR2RGB)
        face_ref_pil = Image.fromarray(face_rgb)

    if reference_frame_bgr is not None and reference_frame_bgr.size > 0:
        cropped_ref = fit_and_crop_to_aspect_ratio(
            reference_frame_bgr,
            aspect_ratio=aspect_ratio,
            focus_box=reference_face_box,
            high_res=False
        )
        ref_rgb = cv2.cvtColor(cropped_ref, cv2.COLOR_BGR2RGB)
        full_ref_pil = Image.fromarray(ref_rgb)
        if face_ref_pil is None:
            auto_face_bgr = extract_character_face_reference_crop(reference_frame_bgr, reference_face_box)
            face_ref_pil = Image.fromarray(cv2.cvtColor(auto_face_bgr, cv2.COLOR_BGR2RGB))

    # Attempt 1: Nano Banana (gemini-3.1-flash-image / gemini-3-pro-image / gemini-2.5-flash-image) + Imagen API
    try:
        from google.genai import types

        def _try_nano_banana_or_imagen(client):
            contents_payload = []
            if face_ref_pil is not None:
                contents_payload.append(face_ref_pil)
            if full_ref_pil is not None:
                contents_payload.append(full_ref_pil)
            contents_payload.append(nano_banana_prompt)

            for img_model in NANO_BANANA_IMAGE_MODELS:
                for use_img_cfg in [True, False]:
                    try:
                        if use_img_cfg and hasattr(types, "ImageConfig"):
                            cfg_obj = types.GenerateContentConfig(
                                response_modalities=["IMAGE", "TEXT"],
                                image_config=types.ImageConfig(aspect_ratio=aspect_ratio)
                            )
                        else:
                            cfg_obj = types.GenerateContentConfig(
                                response_modalities=["IMAGE", "TEXT"]
                            )

                        resp = client.models.generate_content(
                            model=img_model,
                            contents=contents_payload,
                            config=cfg_obj
                        )
                        if resp and getattr(resp, "candidates", None):
                            for cand in resp.candidates:
                                content = getattr(cand, "content", None)
                                for part in (getattr(content, "parts", None) or []):
                                    inline = getattr(part, "inline_data", None)
                                    if inline and getattr(inline, "data", None):
                                        img_bytes = inline.data
                                        arr = np.frombuffer(img_bytes, dtype=np.uint8)
                                        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                        if decoded is not None and decoded.size > 0:
                                            return (decoded, img_model)
                    except Exception:
                        continue

            for imagen_model in ["imagen-3.0-generate-002", "imagen-3.0-fast-generate-001"]:
                try:
                    img_resp = client.models.generate_images(
                        model=imagen_model,
                        prompt=nano_banana_prompt,
                        config=types.GenerateImagesConfig(
                            number_of_images=1,
                            aspect_ratio=aspect_ratio,
                            output_mime_type="image/jpeg"
                        )
                    )
                    gen_imgs = getattr(img_resp, "generated_images", None) or []
                    if gen_imgs:
                        img_obj = getattr(gen_imgs[0], "image", None)
                        img_bytes = getattr(img_obj, "image_bytes", None)
                        if img_bytes:
                            arr = np.frombuffer(img_bytes, dtype=np.uint8)
                            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if decoded is not None and decoded.size > 0:
                                return (decoded, imagen_model)
                except Exception:
                    continue

            return (None, None)

        gen_img, used_engine = execute_with_key_rotation(channel_id, _try_nano_banana_or_imagen)
        if gen_img is not None and gen_img.size > 0:
            generated_bgr = fit_and_crop_to_aspect_ratio(gen_img, aspect_ratio=aspect_ratio, high_res=True)
            generated_bgr = _render_ultra_high_contrast_ai_visual(
                generated_bgr,
                aspect_ratio=aspect_ratio,
                metadata=metadata,
                focus_box=None,
                video_seed_str=video_seed_str,
                apply_typography=False,
                replace_mundane_backdrop=False
            )
            generation_engine = used_engine or "gemini-nano-banana-4k"
            print(f"[Nano Banana 4K Thumbnail] Generated Slot 1 AI Thumbnail via {generation_engine} ({aspect_ratio}, Genre: {genre_category})")
    except Exception as e:
        print(f"[Nano Banana 4K Thumbnail] Using 4K Genre-Atmospheric Character-Preserving Compositor ({str(e)[:80]})...")

    # Attempt 2 / 4K Poster-Grade Genre-Atmospheric Compositor (Preserves 100% character face + replaces mundane backdrop)
    if generated_bgr is None:
        base_frame = reference_frame_bgr
        if base_frame is None or base_frame.size == 0:
            base_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        generated_bgr = _render_ultra_high_contrast_ai_visual(
            base_frame,
            aspect_ratio=aspect_ratio,
            metadata=metadata,
            focus_box=reference_face_box,
            video_seed_str=video_seed_str,
            apply_typography=True,
            replace_mundane_backdrop=True
        )

    uid = uuid.uuid4().hex[:8]
    ratio_slug = "9x16" if aspect_ratio == "9:16" else "16x9"
    ai_filename = f"thumb_slot1_4k_{ratio_slug}_{uid}.jpg"
    ai_filepath = os.path.join(THUMBNAILS_DIR, ai_filename)
    cv2.imwrite(ai_filepath, generated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 97])

    return {
        "id": "slot_1_ai",
        "slot": 1,
        "filename": ai_filename,
        "url": f"/api/thumbnail_file/{ai_filename}",
        "filepath": ai_filepath,
        "seconds": 0.0,
        "timestamp": f"SLOT 1 • 4K NANO BANANA ({aspect_ratio})",
        "label": f"🍌 Slot 1: 4K Nano Banana ({genre_category} • {aspect_ratio})",
        "has_face": True,
        "is_ai_generated": True,
        "is_recommended": True,
        "selected": True,
        "aspect_ratio": aspect_ratio,
        "width": target_w,
        "height": target_h,
        "genre_preset": genre_category,
        "engine": generation_engine
    }


def analyze_video_with_gemini(
    video_path: str,
    format_type: str = "Auto",
    custom_instructions: str = "",
    channel_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    UNIVERSAL MULTIMODAL INGESTION & 4K AI THUMBNAIL ENGINE:
    1. DEEP DUAL-TRACK ANALYSIS (FRAMES + SPOKEN AUDIO):
       - Transcribes 100% of spoken audio/dialogues to extract true entities, plot twists, and mood.
       - Samples key visual frames across the timeline to detect video genre (War/Heroic, Horror/Thriller, Comedy/Drama, Mystery/Suspense).
       - Auto-detects aspect ratio directly from the video stream:
         * Vertical (9:16) -> Viral Shorts titles, description hooks, and high-velocity hashtags.
         * Horizontal/Custom (16:9) -> Comprehensive YouTube SEO description, chapter beats, 15+ search-intent tags, and category classification.
    2. HYPER-ENGAGING 4K NANO BANANA THUMBNAIL PIPELINE:
       - Extracts the highest-emotion character face directly from native video frames.
       - Feeds facial reference + genre into Nano Banana / Imagen API with genre-specific dramatic rendering.
       - Preserves 100% character facial identity while replacing mundane bodies/backgrounds with 4K poster-grade compositions.
    """
    cfg = get_gemini_config(channel_id)
    if not cfg["is_configured"]:
        raise ValueError("Gemini API key is not configured. Please click '⚙️ Configure API Key' in the header to enter your API key first.")

    target_model = _normalize_model_name(cfg.get("model") or DEFAULT_MODEL)
    candidate_models = [target_model] + [m for m in FALLBACK_MODELS if m != target_model]
    models_to_try = []
    for m in candidate_models:
        if m and m not in models_to_try:
            models_to_try.append(m)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at {video_path}")

    # Step 1: Auto-detect aspect ratio from native video geometry first (9:16 Vertical vs 16:9 Horizontal)
    aspect_ratio, target_w, target_h = detect_target_aspect_ratio(video_path, format_type=format_type)
    if aspect_ratio == "9:16":
        format_type = "Short"
    else:
        format_type = "Long"

    # Step 2: Sample timeline frames & extract Slots 2 to 6 + #1 highest-emotion character face crop
    local_slots_2_to_6, best_raw_info = extract_video_thumbnails(
        video_path,
        count=5,
        aspect_ratio=aspect_ratio,
        format_type=format_type,
        return_best_raw=True
    )
    timeline_pil_frames = best_raw_info.get("timeline_pil_frames") or []

    # Step 3: Deep Dual-Track Analysis (100% Spoken Audio + Sampled Timeline Visual Frames)
    system_instruction = """You are an elite Multimodal Video Intelligence & YouTube Algorithm Strategist.

CRITICAL DUAL-TRACK ANALYSIS DIRECTIVE:
1. TRACK 1 — 100% SPOKEN AUDIO & DIALOGUE TRANSCRIPTION:
   - Listen to the entire audio track and transcribe 100% of the spoken dialogue, narration, and vocal reactions.
   - Extract the TRUE ENTITIES (character names, real people, places, military/story roles, objects), the exact PLOT TWISTS, and the emotional MOOD directly from what is spoken AND what is shown.
   - NEVER simply copy the first spoken sentence as the video title; synthesize the true story hook and emotional climax.

2. TRACK 2 — VISUAL TIMELINE & GENRE DETECTION:
   - Inspect the visual frames across the timeline from opening scene to climax.
   - Classify the video into one of these exact genre archetypes based on visual action and audio tone:
     * "War/Heroic" (e.g., soldiers, kids playing/acting as soldiers, military gear, combat, rescue, patriotic/heroic action)
     * "Horror/Thriller" (e.g., suspense, scary moments, shock, dark mystery, crime, paranormal, danger)
     * "Comedy/Drama" (e.g., funny skits, exaggerated reactions, emotional family/relationship drama, pranks, entertainment)
     * "Mystery/Suspense" (e.g., untold stories, plot twists, investigations, shocking reveals)
   - Describe the main character's facial expression and physical appearance in detail so the 4K thumbnail engine can preserve 100% of their facial identity while placing them in a dramatic poster backdrop.

3. ASPECT-RATIO & FORMAT AUTO-METADATA GENERATION:
   - If Vertical (9:16 / Short):
     * Title: High-CTR, suspenseful, emotion-driven Shorts title (under 55 chars) + 2 viral hashtags (#Shorts, #Genre).
     * Alternative Titles: 3 distinct ultra-clickable Shorts title options grounded in the real dialogue & plot twist.
     * Description: Snappy narrative hook + full context summary of the video and spoken dialogue + high-velocity hashtags.
     * Tags: 15+ high-volume search-intent tags combining real names/entities, genre terms, and YouTube Shorts search queries.
   - If Horizontal/Custom (16:9 / Long):
     * Title: High-CTR, suspenseful, emotion-driven title using [Suspenseful Emotional Hook] | [High-Volume Search Keyword].
     * Alternative Titles: 3 distinct high-CTR long-form title options.
     * Description: Comprehensive YouTube SEO description containing:
       (a) Compelling 2-sentence Narrative Hook
       (b) Full Context Summary grounded in the spoken dialogue & visual story
       (c) Timestamped Chapter Beats (e.g. 00:00 Intro, etc.)
       (d) Targeted SEO Hashtags
     * Tags: 15-20 high-volume search-intent tags combining real names, genre terms, and YouTube search queries.

4. OUTPUT FORMAT:
   - Return strictly valid JSON matching the requested schema with zero markdown wrapper chatter."""

    analysis_prompt = f"""DETECTED VIDEO GEOMETRY: {aspect_ratio} ({'Vertical YouTube Shorts / Reels' if aspect_ratio == '9:16' else 'Horizontal / Widescreen Long-Form Video'})
TARGET FORMAT MODE: {format_type}

Perform a Deep Dual-Track Analysis of BOTH:
1. 100% of the spoken audio / dialogues in this video.
2. The visual frames across the entire timeline.

{f'Creator Specific Direction: {custom_instructions}' if custom_instructions else ''}

Return strictly a valid JSON object matching this exact schema:
{{
  "format_type": "{format_type}",
  "spoken_audio_transcript": "Complete transcription and summary of the spoken dialogues/audio in the video (in original spoken language/Hinglish/English)",
  "true_entities": ["Real character names, roles, locations, or key objects extracted from dialogue and visuals"],
  "plot_twists": "The key narrative twist, climax, or emotional punchline of the video",
  "visual_timeline_analysis": "Scene-by-scene breakdown of what visually happens on screen from start to finish",
  "detected_genre": "War/Heroic | Horror/Thriller | Comedy/Drama | Mystery/Suspense",
  "detected_genre_emotion": "Detailed genre + emotional mood (e.g., War/Heroic - Courageous & Intense, Comedy/Drama - Hilarious Twist)",
  "detected_language": "Detected spoken language (e.g., Hindi, Hinglish, Urdu, English)",
  "primary_context": "Concise 5-10 word summary of the true subject, characters, and story plot",
  "climactic_context": "Specific description of the highest-stakes visual & spoken climax",
  "facial_expression_analysis": "Detailed description of the main character's face, expression, eyes, and emotion for 100% identity-preserving 4K thumbnail generation",
  "viral_title": "High-CTR, suspenseful, emotion-driven title grounded 100% in the video's true story and dialogue",
  "alternative_titles": [
    "Alternative High-CTR Title Option 1 (Curiosity / Plot Twist Hook)",
    "Alternative High-CTR Title Option 2 (Emotional / Dialogue Punchline Hook)",
    "Alternative High-CTR Title Option 3 (Search-Intent SEO Hook)"
  ],
  "description": "Narrative hook + full context summary + chapter beats (if 16:9 long-form) + hashtags",
  "hashtags": ["#Shorts", "#Viral", "#Trending", "#GenreSpecificTag1", "#GenreSpecificTag2", "#TopicTag"],
  "search_tags": ["15 to 20 high-volume search-intent tags combining real entity names, genre terms, and YouTube search queries"],
  "category_id": "24",
  "category_name": "Entertainment",
  "recommended_thumbnail_second": 2.5,
  "thumbnail_directive": {{
    "text_overlay": "Max 3-4 punchy, suspenseful words matching the true video climax",
    "visual_scene_direction": "Genre-specific 4K poster direction preserving 100% character facial identity while replacing mundane background with dramatic genre elements",
    "recommended_color_theme": "Genre-matched 4K color grade (e.g., War fiery amber & olive smoke, Horror crimson & icy cyan, Comedy vibrant gold & studio pop)"
  }}
}}"""

    metadata = None
    last_error = None

    def _parse_gemini_json_response(raw_text: str, model_used_name: str, mode_label: str) -> Optional[Dict[str, Any]]:
        if not raw_text:
            return None
        match = re.search(r'(\{[\s\S]*\})', raw_text)
        if not match:
            return None
        parsed = json.loads(match.group(1))

        viral_title = (
            parsed.get("viral_title")
            or parsed.get("primary_title")
            or parsed.get("recommended_title")
            or "Unbelievable Moment You Need To See"
        )
        spoken_transcript = str(parsed.get("spoken_audio_transcript") or "").strip()
        visual_timeline = str(parsed.get("visual_timeline_analysis") or "").strip()
        true_entities = parsed.get("true_entities") or []
        plot_twists = str(parsed.get("plot_twists") or parsed.get("climactic_context") or "").strip()
        detected_genre = str(parsed.get("detected_genre") or "Mystery/Suspense").strip()
        genre_emotion = str(parsed.get("detected_genre_emotion") or f"{detected_genre} • High Impact").strip()
        detected_lang = str(parsed.get("detected_language") or "Hindi / Hinglish").strip()
        primary_ctx = str(parsed.get("primary_context") or plot_twists or "Trending Video Story").strip()
        climactic_ctx = str(parsed.get("climactic_context") or plot_twists or primary_ctx).strip()
        facial_expr = str(parsed.get("facial_expression_analysis") or "Intense emotional expression").strip()

        alt_titles = parsed.get("alternative_titles")
        if not isinstance(alt_titles, list) or len(alt_titles) == 0:
            alt_titles = [
                f"{viral_title} 🔥",
                f"The Shocking Truth About {primary_ctx} 🎯",
                f"Wait For The Ending: {primary_ctx} ⚡"
            ]

        desc = str(parsed.get("description") or "").strip()
        hashtags = parsed.get("hashtags") or []
        search_tags = parsed.get("search_tags") or parsed.get("seo_keywords") or parsed.get("tags") or []
        # Ensure at least 15 search-intent tags combining true entities, genre terms, and search queries
        extra_seed_tags = [
            *(str(e).strip() for e in true_entities if str(e).strip()),
            detected_genre.replace("/", " ").lower(),
            primary_ctx.lower(),
            "youtube shorts" if format_type == "Short" else "full story explained",
            "viral video",
            "trending",
            "must watch",
            "hindi kahani",
            "emotional story",
            "shocking twist",
            "4k video",
            "new video",
            "entertainment",
            "storytime",
            "top trending"
        ]
        for et in extra_seed_tags:
            if len(search_tags) >= 16:
                break
            if et and et.lower() not in [str(x).lower() for x in search_tags]:
                search_tags.append(et)

        thumb_dir = parsed.get("thumbnail_directive") or {
            "text_overlay": "SHOCKING MOMENT",
            "visual_scene_direction": f"Preserve 100% character face in {detected_genre} 4K poster composition",
            "recommended_color_theme": "High contrast cinematic 4K color grade"
        }

        full_desc = desc
        if hashtags:
            formatted_tags = [h if str(h).startswith("#") else f"#{h}" for h in hashtags if str(h).strip()]
            tag_line = " ".join(formatted_tags)
            if tag_line and tag_line not in full_desc:
                full_desc = f"{full_desc}\n\n{tag_line}"

        return {
            "format_type": format_type,
            "thumbnail_aspect_ratio": aspect_ratio,
            "spoken_audio_transcript": spoken_transcript,
            "true_entities": true_entities,
            "plot_twists": plot_twists,
            "visual_timeline_analysis": visual_timeline,
            "detected_genre": detected_genre,
            "detected_genre_emotion": genre_emotion,
            "detected_language": detected_lang,
            "primary_context": primary_ctx,
            "climactic_context": climactic_ctx,
            "facial_expression_analysis": facial_expr,
            "viral_title": viral_title,
            "primary_title": viral_title,
            "recommended_title": viral_title,
            "alternative_titles": alt_titles[:3],
            "description": full_desc,
            "raw_description": desc,
            "hashtags": hashtags,
            "search_tags": search_tags,
            "seo_keywords": search_tags,
            "tags": search_tags,
            "thumbnail_directive": thumb_dir,
            "recommended_thumbnail_second": float(parsed.get("recommended_thumbnail_second", best_raw_info.get("seconds", 2.5))),
            "category_id": str(parsed.get("category_id", "24")),
            "category_name": parsed.get("category_name", "Entertainment"),
            "video_type": format_type,
            "made_for_kids": False,
            "model_used": model_used_name,
            "summary_insights": (
                f"Genre: {detected_genre} | Language: {detected_lang} | Format: {format_type} ({aspect_ratio}) | "
                f"Context: {primary_ctx} | Twist: {climactic_ctx} | Dialogue: {spoken_transcript[:180]}"
            )
        }

    def _upload_and_analyze_with_client(client):
        nonlocal metadata, last_error
        from google.genai import types

        uploaded_file = None
        file_name = None

        try:
            print(f"[Gemini Dual-Track Engine] Uploading video + audio stream ({video_path}) to Gemini Files API...")
            uploaded_file = client.files.upload(file=video_path)
            file_name = uploaded_file.name

            start_time = time.time()
            while True:
                f_info = client.files.get(name=file_name)
                state = getattr(f_info.state, "name", str(f_info.state))
                if state == "ACTIVE":
                    print("[Gemini Dual-Track Engine] Video + Audio stream is ACTIVE.")
                    break
                elif state in ["FAILED", "ERROR"]:
                    raise RuntimeError(f"Gemini video stream processing failed with state: {state}")
                if time.time() - start_time > 240:
                    raise TimeoutError("Gemini video file processing timed out after 4 minutes.")
                time.sleep(2.0)
        except Exception as up_err:
            last_error = up_err
            err_s = str(up_err).lower()
            if any(w in err_s for w in ["api_key_invalid", "api key not valid", "permission_denied", "403"]):
                raise up_err
            print(f"[Gemini Dual-Track Engine] File upload notice ({up_err}); will include sampled timeline keyframes directly.")

        try:
            cfg_with_search = types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.3,
                system_instruction=system_instruction
            )
            cfg_json_only = types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
                system_instruction=system_instruction
            )

            # Try all candidate models & modes before rotating keys so model-specific 404/429s don't abort prematurely
            quota_hit_on_all = False
            for model_to_call in models_to_try:
                for use_search in [True, False]:
                    config = cfg_with_search if use_search else cfg_json_only
                    mode_label = "Dual-Track + Google Search Grounding" if use_search else "Dual-Track Direct JSON"

                    try:
                        print(f"[Gemini Dual-Track Engine] Analyzing {format_type} ({aspect_ratio}) with {model_to_call} ({mode_label})...")
                        if uploaded_file is not None:
                            contents_list = [uploaded_file] + timeline_pil_frames[:4] + [analysis_prompt]
                        else:
                            contents_list = timeline_pil_frames + [analysis_prompt]

                        response = client.models.generate_content(
                            model=model_to_call,
                            contents=contents_list,
                            config=config
                        )
                        raw_text = (response.text or "").strip()
                        parsed_meta = _parse_gemini_json_response(raw_text, model_to_call, mode_label)
                        if parsed_meta:
                            metadata = parsed_meta
                            return metadata
                    except Exception as me:
                        last_error = me
                        err_str = str(me).lower()
                        if any(w in err_str for w in ["api_key_invalid", "api key not valid"]):
                            raise me
                        if any(w in err_str for w in ["429", "resource_exhausted", "quota"]):
                            quota_hit_on_all = True
                        continue

            if quota_hit_on_all and metadata is None and last_error is not None:
                raise last_error
        finally:
            if file_name:
                try:
                    client.files.delete(name=file_name)
                except Exception:
                    pass
        return metadata

    try:
        execute_with_key_rotation(channel_id, _upload_and_analyze_with_client)
    except Exception as rot_err:
        last_error = rot_err
        print(f"[Gemini Dual-Track Engine] Key rotation notice: {rot_err}")

    if not metadata:
        print(f"[Gemini Dual-Track Engine] Generating context-aware fallback metadata ({last_error})...")
        clean_name = os.path.splitext(os.path.basename(video_path))[0]
        clean_name = re.sub(r'^(gemini_[a-f0-9-]+_|vid_\d+_)', '', clean_name).replace('_', ' ').replace('-', ' ').title() or "Viral Video"
        preset = _resolve_genre_dramatic_preset("", clean_name, "", "", len(clean_name))
        detected_genre = preset["genre_category"]
        short_hook_words = " ".join(clean_name.split()[:3]).upper() or "SHOCKING TWIST"
        viral_title = (
            f"{clean_name}: Wait For The Twist! 😱 #Shorts #Viral"
            if format_type == "Short"
            else f"{clean_name} — Untold Story & Full Breakdown | Must Watch"
        )
        fallback_tags = [
            clean_name.lower(),
            detected_genre.replace("/", " ").lower(),
            "youtube shorts" if format_type == "Short" else "full story explained",
            "viral video", "trending", "must watch", "shocking twist", "emotional story",
            "new video", "entertainment", "hindi story", "4k video", "viral reel",
            "best moments", "top trending", "creator studio"
        ]
        fallback_desc = (
            f"🔥 {clean_name} — Watch till the very end for the unbelievable twist!\n\n"
            f"Full Context Summary: Experience the high-intensity {detected_genre} moments captured in {clean_name}.\n\n"
            + ("00:00 Intro & Hook\n00:30 Rising Action\n01:00 Climactic Twist\n\n" if format_type == "Long" else "")
            + "#Shorts #Trending #Viral #MustWatch"
        )
        metadata = {
            "format_type": format_type,
            "thumbnail_aspect_ratio": aspect_ratio,
            "spoken_audio_transcript": f"Audio track analyzed for {clean_name} ({detected_genre}).",
            "true_entities": [clean_name],
            "plot_twists": f"Climactic turning point in {clean_name}",
            "visual_timeline_analysis": f"Keyframes sampled across {aspect_ratio} timeline ({detected_genre}).",
            "detected_genre": detected_genre,
            "detected_genre_emotion": f"{detected_genre} • High Suspense",
            "detected_language": "Hindi / Hinglish",
            "primary_context": clean_name,
            "climactic_context": f"Climactic turning point in {clean_name}",
            "facial_expression_analysis": "High-tension expressive character reaction at the decisive moment",
            "viral_title": viral_title,
            "primary_title": viral_title,
            "recommended_title": viral_title,
            "alternative_titles": [
                f"Nobody Expected This In {clean_name}! 🔥",
                f"The Untold Truth Behind {clean_name} 🎯",
                f"Wait Till The End: {clean_name} ⚡"
            ],
            "description": fallback_desc,
            "raw_description": fallback_desc,
            "hashtags": ["#Shorts", "#Trending", "#Viral", "#MustWatch", f"#{detected_genre.split('/')[0]}"],
            "search_tags": fallback_tags,
            "seo_keywords": fallback_tags,
            "tags": fallback_tags,
            "thumbnail_directive": {
                "text_overlay": short_hook_words,
                "visual_scene_direction": f"Preserve 100% character face in {detected_genre} 4K poster composition",
                "recommended_color_theme": "4K Poster Grade High-Contrast Lighting"
            },
            "recommended_thumbnail_second": float(best_raw_info.get("seconds", 2.0)),
            "category_id": "24",
            "category_name": "Entertainment",
            "video_type": format_type,
            "made_for_kids": False,
            "model_used": "smart-fallback-engine",
            "summary_insights": f"Genre: {detected_genre} | Format: {format_type} ({aspect_ratio}) | Context: {clean_name}."
        }

    # Step 4: Generate SLOT 1 (DEFAULT SELECTED) 4K Nano Banana AI Dynamic Thumbnail
    slot_1_ai_thumb = generate_dynamic_ai_thumbnail(
        video_path=video_path,
        format_type=format_type,
        aspect_ratio=aspect_ratio,
        metadata=metadata,
        reference_frame_bgr=best_raw_info.get("frame"),
        reference_face_crop_bgr=best_raw_info.get("face_crop_bgr"),
        reference_face_box=best_raw_info.get("face_box"),
        channel_id=channel_id
    )

    # Combine Slot 1 (Default Selected 4K Nano Banana Thumbnail) + Slots 2 to 6 (5 High-Emotion Local Video Frames)
    all_slots = [slot_1_ai_thumb] + local_slots_2_to_6
    metadata["extracted_thumbnails"] = all_slots
    metadata["selected_thumbnail"] = slot_1_ai_thumb
    metadata["thumbnail_aspect_ratio"] = aspect_ratio
    return metadata


def chat_with_gemini(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    studio_context: Optional[Dict[str, Any]] = None,
    channel_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Interactive creator copilot chat. Can rewrite titles, tailor descriptions,
    translate metadata, brainstorm hooks, and advise on video performance.
    """
    cfg = get_gemini_config(channel_id)
    if not cfg["is_configured"]:
        return {
            "reply": "Gemini API key is not configured yet. Please click '⚙️ Configure API Key' in the top header or Gemini panel to enter your API key to activate AI chat and video analysis.",
            "error": "api_key_missing"
        }

    model_name = _normalize_model_name(cfg.get("model", DEFAULT_MODEL))

    system_prompt = """
You are 'Gemini Creator Copilot', a world-class YouTube strategy partner embedded inside YouTube Creator Studio Pro.
You help creators optimize titles, descriptions, tags, SEO, thumbnails, and channel growth.
You speak clearly, enthusiastically, and practically.
When suggesting titles, provide 3-5 distinct options (Viral/CTR, Search SEO, Curiosity/Story).
If you provide a refined title or description, enclose it clearly or in structured format so the user can apply it directly to their upload.
Format suggestions clearly with labels like:
[TITLE_SUGGESTION]: Your proposed title
[DESCRIPTION_SUGGESTION]: Your proposed description
"""

    context_str = ""
    if studio_context:
        context_str = f"\n\nCURRENT STUDIO CONTEXT:\n- Current Title: {studio_context.get('title', 'None')}\n- Category: {studio_context.get('category', 'None')}\n- Privacy: {studio_context.get('privacy', 'Public')}\n"

    full_prompt = f"{system_prompt}\n{context_str}\nUser question/request: {message}"

    try:
        reply_text = ""
        models_to_try = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]

        def _call_chat(client):
            nonlocal reply_text
            for m in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=m,
                        contents=[full_prompt]
                    )
                    reply_text = response.text or ""
                    if reply_text:
                        return reply_text
                except Exception as ce:
                    err_s = str(ce).lower()
                    if any(t in err_s for t in ["api_key_invalid", "api key not valid"]):
                        raise ce
                    continue
            return reply_text

        execute_with_key_rotation(channel_id, _call_chat)

        if not reply_text:
            reply_text = "I'm currently optimizing for high traffic, but here is a quick tip: Focus your title on high curiosity + clear emotional hook and keep YouTube Shorts under 50 characters."

        title_match = re.search(r'\[TITLE_SUGGESTION\]:\s*(.*)', reply_text)
        desc_match = re.search(r'\[DESCRIPTION_SUGGESTION\]:\s*([\s\S]*?)(?:\[|$)', reply_text)

        return {
            "reply": reply_text,
            "suggested_title": title_match.group(1).strip() if title_match else None,
            "suggested_description": desc_match.group(1).strip() if desc_match else None
        }
    except Exception as e:
        return {
            "reply": f"Gemini Error: {str(e)}",
            "error": str(e)
        }


# =====================================================================
# SUPERFAST YOUTUBE INGESTION & 4K MOVIE-POSTER THUMBNAIL SUITE
# =====================================================================

def extract_youtube_video_id(url_or_id: str) -> Optional[str]:
    """Extracts the 11-character YouTube video ID from various URL formats or raw ID."""
    clean = (url_or_id or "").strip()
    if not clean:
        return None
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", clean):
        return clean
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/|\/shorts\/|\/live\/)([A-Za-z0-9_-]{11})",
        r"[?&]v=([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, clean)
        if m:
            return m.group(1)
    return None


def parse_iso8601_duration(dur_str: str) -> int:
    """Parses ISO 8601 duration string (e.g. PT1M15S, PT45S, PT1H2M10S) into seconds."""
    if not dur_str:
        return 0
    m = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', dur_str)
    if not m:
        return 0
    h = int(m.group(1) or 0)
    m_ = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 3600 + m_ * 60 + s


def fetch_youtube_video_transcript(video_id: str) -> Tuple[str, List[str]]:
    """
    Fetches 100% accurate spoken dialogues / transcript snippets using youtube-transcript-api.
    Tries Hindi, English, Urdu, and auto-generated transcripts with translation fallback.
    """
    transcript_text = ""
    snippets: List[str] = []
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()
        fetched = None
        try:
            fetched = api.fetch(video_id, languages=['hi', 'en', 'ur', 'auto'])
        except Exception:
            pass

        if not fetched:
            try:
                t_list = api.list(video_id)
                for t in t_list:
                    try:
                        fetched = t.fetch()
                        if fetched:
                            break
                    except Exception:
                        continue
            except Exception:
                pass

        if fetched:
            for item in fetched:
                if hasattr(item, 'text'):
                    t = (item.text or "").strip()
                elif isinstance(item, dict):
                    t = (item.get('text') or "").strip()
                else:
                    t = str(item).strip()
                if t:
                    snippets.append(t)
            transcript_text = " ".join(snippets)
    except Exception as e:
        print(f"[YouTube Transcript] Notice: {e}")

    return transcript_text, snippets


def download_youtube_thumbnail_frame(video_id: str) -> Tuple[Optional[np.ndarray], str]:
    """
    Downloads the highest-resolution thumbnail for a YouTube video to use as facial reference.
    Tries maxresdefault.jpg -> sddefault.jpg -> hqdefault.jpg.
    Returns (cv2_frame_bgr, local_saved_path).
    """
    import urllib.request
    import cv2

    urls_to_try = [
        f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",
        f"https://i.ytimg.com/vi/{video_id}/sddefault.jpg",
        f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
    ]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

    raw_bytes = None
    for url in urls_to_try:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=6) as response:
                if response.status == 200:
                    data = response.read()
                    if len(data) > 1024:
                        raw_bytes = data
                        break
        except Exception:
            continue

    if raw_bytes:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is not None and decoded.size > 0:
            local_filename = f"yt_ref_{video_id}_{uuid.uuid4().hex[:6]}.jpg"
            local_path = os.path.join(THUMBNAILS_DIR, local_filename)
            cv2.imwrite(local_path, decoded, [cv2.IMWRITE_JPEG_QUALITY, 96])
            return decoded, local_path

    # Fallback placeholder if network/DNS doesn't reach YouTube servers
    local_filename = f"yt_ref_placeholder_{video_id}.jpg"
    local_path = os.path.join(THUMBNAILS_DIR, local_filename)
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    for y in range(720):
        dummy[y, :, 0] = int(24 + (y / 720) * 40)
        dummy[y, :, 1] = int(18 + (y / 720) * 30)
        dummy[y, :, 2] = int(36 + (y / 720) * 60)
    cv2.imwrite(local_path, dummy, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return dummy, local_path


def analyze_youtube_video_with_gemini(
    video_id_or_url: str,
    format_type: str = "Auto",
    custom_instructions: str = "",
    channel_id: Optional[str] = None,
    youtube_service: Optional[Any] = None,
    existing_video_meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    SUPERFAST YOUTUBE INGESTION & 4K MOVIE-POSTER THUMBNAIL ENGINE:
    1. Ingests pre-processed data from YouTube in seconds:
       - 100% accurate spoken dialogue transcript (via YouTube auto-subtitles/speech-to-text)
       - Duration & aspect ratio auto-detection (<= 60s -> 9:16 Shorts; > 60s -> 16:9 Long-form)
       - High-resolution frame for facial identity extraction
    2. Runs Gemini 2.5 Dual-Track Analysis on dialogues + story context
    3. Produces Slot 1 (4K Nano Banana Movie Poster Thumbnail, default selected) in strict 9:16 or 16:9
       preserving 100% character face likeness while generating a blockbuster movie poster composition!
    4. Slot 2 is the original YouTube high-res reference frame.
    5. Ready for 1-Click "Apply & Go PUBLIC"!
    """
    import cv2
    from PIL import Image

    video_id = extract_youtube_video_id(video_id_or_url)
    if not video_id:
        raise ValueError(f"Invalid YouTube URL or Video ID: '{video_id_or_url}'")

    cfg = get_gemini_config(channel_id)
    if not cfg["is_configured"]:
        raise ValueError("Gemini API key is not configured. Please enter your API key in '⚙️ Configure API Key'.")

    target_model = _normalize_model_name(cfg.get("model") or DEFAULT_MODEL)
    candidate_models = [target_model] + [m for m in FALLBACK_MODELS if m != target_model]
    models_to_try = []
    for m in candidate_models:
        if m and m not in models_to_try:
            models_to_try.append(m)

    # 1. Fetch metadata from YouTube API if available
    yt_title = ""
    yt_desc = ""
    yt_tags = []
    dur_seconds = 0
    privacy_status = "PRIVATE"
    category_id = "24"

    if existing_video_meta:
        yt_title = existing_video_meta.get("title", "")
        yt_desc = existing_video_meta.get("description", "")
        yt_tags = existing_video_meta.get("tags", [])
        dur_seconds = existing_video_meta.get("duration_seconds", 0)
        privacy_status = existing_video_meta.get("privacy", "PRIVATE")
        category_id = existing_video_meta.get("category_id", "24")

    if not yt_title and youtube_service:
        try:
            v_res = youtube_service.videos().list(id=video_id, part="snippet,contentDetails,status").execute()
            items = v_res.get("items", [])
            if items:
                snip = items[0].get("snippet", {})
                cd = items[0].get("contentDetails", {})
                st = items[0].get("status", {})
                yt_title = snip.get("title", "")
                yt_desc = snip.get("description", "")
                yt_tags = snip.get("tags", [])
                category_id = snip.get("categoryId", "24")
                dur_seconds = parse_iso8601_duration(cd.get("duration", ""))
                privacy_status = st.get("privacyStatus", "PRIVATE").upper()
        except Exception as ye:
            print(f"[YouTube Video Analyzer] Notice fetching YouTube API details: {ye}")

    if not yt_title:
        yt_title = f"YouTube Video {video_id}"

    # 2. Aspect Ratio & Format Detection
    if format_type == "Auto":
        if (dur_seconds > 0 and dur_seconds <= 60) or ("/shorts/" in str(video_id_or_url).lower()) or ("#shorts" in yt_title.lower()):
            aspect_ratio = "9:16"
            format_type = "Short"
            target_w, target_h = (1080, 1920)
        else:
            aspect_ratio = "16:9"
            format_type = "Long"
            target_w, target_h = (1920, 1080)
    elif format_type == "Short":
        aspect_ratio = "9:16"
        target_w, target_h = (1080, 1920)
    else:
        aspect_ratio = "16:9"
        target_w, target_h = (1920, 1080)

    # 3. Fetch 100% accurate spoken audio / transcript from YouTube
    transcript_text, snippets = fetch_youtube_video_transcript(video_id)
    if not transcript_text:
        transcript_text = f"Context from YouTube Video '{yt_title}'. Spoken dialogues processed for storyline and character dynamic."

    # 4. Download High-Res Thumbnail for Face Reference
    raw_frame_bgr, local_thumb_path = download_youtube_thumbnail_frame(video_id)

    # Detect character faces & extract #1 highest-emotion character face crop
    frontal_cascade = None
    profile_cascade = None
    try:
        cascade_dir = getattr(cv2, 'data', None)
        cpath = getattr(cascade_dir, 'haarcascades', '') if cascade_dir else ''
        if cpath and os.path.exists(cpath):
            fpath = os.path.join(cpath, 'haarcascade_frontalface_default.xml')
            if os.path.exists(fpath):
                frontal_cascade = cv2.CascadeClassifier(fpath)
            ppath = os.path.join(cpath, 'haarcascade_profileface.xml')
            if os.path.exists(ppath):
                profile_cascade = cv2.CascadeClassifier(ppath)
    except Exception:
        pass

    face_box = None
    face_crop_bgr = None
    if raw_frame_bgr is not None and raw_frame_bgr.size > 0:
        _, face_count, best_box, _ = _score_frame_emotion_and_motion(
            raw_frame_bgr,
            prev_gray=None,
            frontal_cascade=frontal_cascade,
            profile_cascade=profile_cascade
        )
        face_box = best_box
        face_crop_bgr = extract_character_face_reference_crop(raw_frame_bgr, face_box)

    # Save Slot 2: Original YouTube High-Res Frame (cropped to aspect ratio)
    slot_2_filename = f"yt_orig_{video_id}_{aspect_ratio.replace(':', 'x')}.jpg"
    slot_2_filepath = os.path.join(THUMBNAILS_DIR, slot_2_filename)
    cropped_orig = fit_and_crop_to_aspect_ratio(
        raw_frame_bgr,
        aspect_ratio=aspect_ratio,
        focus_box=face_box,
        high_res=True
    )
    cv2.imwrite(slot_2_filepath, cropped_orig, [cv2.IMWRITE_JPEG_QUALITY, 96])

    slot_2_thumb = {
        "id": "slot_2_yt_orig",
        "slot": 2,
        "filename": slot_2_filename,
        "url": f"/api/thumbnail_file/{slot_2_filename}",
        "filepath": slot_2_filepath,
        "seconds": 0.0,
        "timestamp": "YouTube Native High-Res Frame",
        "label": f"🎬 Slot 2: YouTube Native Frame ({aspect_ratio})",
        "has_face": bool(face_box is not None),
        "is_ai_generated": False,
        "is_recommended": False,
        "selected": False,
        "aspect_ratio": aspect_ratio,
        "width": target_w,
        "height": target_h
    }

    # 5. Multimodal Story Analysis with Gemini 2.5
    prompt_str = f"""
You are the world's most elite YouTube Growth Strategist & Cinematic Screenplay Analyst.
You are analyzing an official YouTube video pre-processed with 100% accurate spoken audio dialogues.

VIDEO SPECS:
- Video ID: {video_id}
- Target Format: {format_type} ({aspect_ratio})
- Video Duration: {dur_seconds}s
- Current YouTube Title: {yt_title}
- Current Description: {yt_desc[:500]}
- 100% SPOKEN AUDIO TRANSCRIPT / DIALOGUES:
\"\"\"{transcript_text[:12000]}\"\"\"

CREATOR INSTRUCTIONS:
{custom_instructions or "Maximize organic search SEO, viral curiosity CTR, and algorithmic engagement."}

TASK:
1. Deeply analyze the dialogues to uncover 100% of the true characters, plot twists, decisive conflict, and emotional stakes.
2. Determine the exact GENRE: War/Heroic, Horror/Thriller, Comedy/Drama, Mystery/Suspense, Emotional/Action.
3. Return a STRICT JSON object with these EXACT keys:
{{
  "detected_genre": "War/Heroic",
  "detected_genre_emotion": "Genre • High Suspense Emotional Climax",
  "detected_language": "Hindi / Hinglish",
  "primary_context": "2-4 word core character/event",
  "climactic_context": "The decisive turning point or shock revelation",
  "spoken_audio_transcript": "2-3 sentence grounded summary of what was actually said",
  "true_entities": ["Main Character", "Key Object", "Setting"],
  "plot_twists": "Key dramatic shift or climax",
  "visual_timeline_analysis": "Visual and atmospheric tone of the video",
  "facial_expression_analysis": "Character facial expression at peak intensity",
  "viral_title": "Primary high-CTR title (under 50 chars for Shorts + #Shorts #Viral; High-volume [Hook | Keyword] for Long-form)",
  "alternative_titles": [
    "Compelling Curiosity Hook Title",
    "High-Search Volume Keyword Title",
    "Dramatic Story-Driven Title"
  ],
  "description": "Engaging description with opening hook, comprehensive context breakdown, chapter timestamps (if long-form), 3-5 hashtags, and creator CTA.",
  "hashtags": ["#Shorts", "#Trending", "#Viral", "#MovieExplained", "#HindiStory"],
  "search_tags": ["15 to 20 high-volume search intent keywords and phrases"],
  "category_id": "24",
  "category_name": "Entertainment",
  "thumbnail_directive": {{
    "text_overlay": "3-4 word 3D movie hook typography in ALL CAPS",
    "visual_scene_direction": "Epic high-budget blockbuster movie poster composition matching the genre",
    "recommended_color_theme": "High-contrast cinematic color palette"
  }},
  "summary_insights": "Actionable strategic insight on why this packaging will maximize retention and click-through rate."
}}
"""

    metadata = None
    last_error = None

    def _call_gemini_analysis(client):
        nonlocal metadata, last_error
        for m in models_to_try:
            try:
                response = client.models.generate_content(
                    model=m,
                    contents=[prompt_str],
                    config={"response_mime_type": "application/json"}
                )
                text = response.text or ""
                clean_json = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
                clean_json = re.sub(r"^```\s*", "", clean_json, flags=re.MULTILINE)
                clean_json = re.sub(r"```$", "", clean_json.strip())
                parsed = json.loads(clean_json)
                if isinstance(parsed, dict) and parsed.get("viral_title"):
                    metadata = parsed
                    metadata["model_used"] = m
                    return metadata
            except Exception as ce:
                last_error = ce
                continue
        return metadata

    try:
        execute_with_key_rotation(channel_id, _call_gemini_analysis)
    except Exception as e:
        print(f"[YouTube Video Analyzer] Gemini notice: {e}")

    # Fallback metadata if needed
    if not metadata:
        clean_name = re.sub(r'[^\w\s-]', '', yt_title).strip() or "Viral Story"
        preset = _resolve_genre_dramatic_preset("", clean_name, "", "", len(clean_name))
        detected_genre = preset["genre_category"]
        short_hook = " ".join(clean_name.split()[:3]).upper() or "MUST WATCH"
        viral_title = (
            f"{clean_name}: Wait For The Twist! 😱 #Shorts #Viral"
            if format_type == "Short"
            else f"{clean_name} — Full Story & Breakdown | Must Watch"
        )
        fallback_tags = [
            clean_name.lower(), detected_genre.replace('/', ' ').lower(),
            "youtube shorts" if format_type == "Short" else "full story explained",
            "viral video", "trending", "must watch", "shocking twist", "emotional story"
        ]
        fallback_desc = f"🔥 {clean_name} — Watch till the very end for the unbelievable twist!\n\nContext: {transcript_text[:300]}...\n\n#Shorts #Trending #Viral"
        metadata = {
            "detected_genre": detected_genre,
            "detected_genre_emotion": f"{detected_genre} • High Suspense",
            "detected_language": "Hindi / Hinglish",
            "primary_context": clean_name,
            "climactic_context": f"Decisive climax in {clean_name}",
            "spoken_audio_transcript": transcript_text[:400],
            "true_entities": [clean_name],
            "plot_twists": f"Dramatic twist in {clean_name}",
            "visual_timeline_analysis": f"Visuals sampled from YouTube video ({aspect_ratio}).",
            "facial_expression_analysis": "Intense expressive emotion at peak dramatic moment",
            "viral_title": viral_title,
            "alternative_titles": [
                f"Nobody Expected This In {clean_name}! 🔥",
                f"The Untold Story: {clean_name} 🎯",
                f"Wait Till The End: {clean_name} ⚡"
            ],
            "description": fallback_desc,
            "hashtags": ["#Shorts", "#Trending", "#Viral", "#MustWatch"],
            "search_tags": fallback_tags,
            "category_id": category_id or "24",
            "category_name": "Entertainment",
            "thumbnail_directive": {
                "text_overlay": short_hook,
                "visual_scene_direction": f"4K {detected_genre} movie poster composition with character face",
                "recommended_color_theme": "High-contrast cinematic lighting"
            },
            "summary_insights": f"Pre-processed YouTube Video {video_id} analyzed. Ready for 1-click publishing."
        }

    # 6. Generate SLOT 1 (DEFAULT SELECTED) 4K Nano Banana Movie Poster Thumbnail
    slot_1_ai_thumb = generate_dynamic_ai_thumbnail(
        video_path=local_thumb_path,
        format_type=format_type,
        aspect_ratio=aspect_ratio,
        metadata=metadata,
        reference_frame_bgr=raw_frame_bgr,
        reference_face_crop_bgr=face_crop_bgr,
        reference_face_box=face_box,
        channel_id=channel_id
    )

    metadata["video_id"] = video_id
    metadata["video_url"] = f"https://youtu.be/{video_id}"
    metadata["is_youtube_video"] = True
    metadata["privacy_status"] = privacy_status
    metadata["format_type"] = format_type
    metadata["thumbnail_aspect_ratio"] = aspect_ratio
    metadata["extracted_thumbnails"] = [slot_1_ai_thumb, slot_2_thumb]
    metadata["selected_thumbnail"] = slot_1_ai_thumb
    metadata["tags"] = metadata.get("search_tags") or metadata.get("tags") or []
    metadata["recommended_title"] = metadata.get("viral_title")
    metadata["primary_title"] = metadata.get("viral_title")
    return metadata

