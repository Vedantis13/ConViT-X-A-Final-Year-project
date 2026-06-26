
"""
app.py — ConViT-X Flask Web Server (Windows Inference)

Features:
  - Serves the frontend (frontend/index.html)
  - REST API for prediction and health check
  - Accepts only image files (rejects PDF, MP3, video, docs, etc.)
  - Max file size: 16 MB
  - GPU if available, CPU otherwise (auto-detected via config.py)
  - Model hot-reloads when checkpoint changes — retrain on server,
    copy convitx_best.pth here, restart app — no code changes needed
  - Full CORS for LAN access (mobile camera on same Wi-Fi)
"""

import os
import io
import base64
import imghdr
import logging
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from PIL import Image, UnidentifiedImageError

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

import config
from inference import ConViTXInference, is_chest_xray, pil_to_base64

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("convitx")

# ─── Flask setup ──────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend", static_url_path="")

MAX_FILE_MB = 16
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024

# ─── CORS ─────────────────────────────────────────────────────────────────────
# Allow all origins so mobile devices on the same Wi-Fi can reach the API.
# This is required for the "Camera (Mobile)" feature to work on phones.
CORS(app, resources={
    r"/api/*": {
        "origins":       "*",
        "methods":       ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "X-Requested-With"],
        "max_age":       600,
    }
})

# ─── File type rules ──────────────────────────────────────────────────────────
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
BLOCKED_EXT = {
    ".pdf", ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".tar", ".gz", ".exe", ".py",
    ".txt", ".csv", ".json", ".xml", ".html", ".htm",
}

# ─── Model engine (singleton, auto-reloads on checkpoint change) ──────────────
CHECKPOINT = os.path.join(config.CHECKPOINT_DIR, "convitx_best.pth")
_engine    = None
_ckpt_mtime = 0.0


def get_engine():
    """
    Returns the loaded inference engine.
    Automatically reloads if convitx_best.pth has changed on disk.
    This means: after retraining, just copy the new .pth file and
    the app picks it up on the next request — no restart required.
    """
    global _engine, _ckpt_mtime

    if not os.path.exists(CHECKPOINT):
        return None

    mtime = os.path.getmtime(CHECKPOINT)
    if _engine is None or mtime != _ckpt_mtime:
        log.info(f"Loading checkpoint: {CHECKPOINT}")
        try:
            _engine     = ConViTXInference(CHECKPOINT)
            _ckpt_mtime = mtime
        except Exception as e:
            log.error(f"Failed to load checkpoint: {e}")
            _engine = None

    return _engine


# ─── Error handlers ───────────────────────────────────────────────────────────

@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return jsonify({
        "success": False,
        "error":   f"File too large. Maximum size is {MAX_FILE_MB} MB.",
        "code":    "FILE_TOO_LARGE",
    }), 413


@app.errorhandler(404)
def not_found(_):
    return jsonify({"success": False, "error": "Endpoint not found."}), 404


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend."""
    return send_from_directory("frontend", "frontend.html")


@app.route("/api/health", methods=["GET"])
def health():
    """
    Health check — the frontend polls this every 30 seconds.
    Returns model status, device info, and allowed file size.
    After retraining and copying a new .pth file, model_ready
    automatically becomes true without any code change.
    """
    import torch
    ckpt_exists = os.path.exists(CHECKPOINT)
    gpu_name    = torch.cuda.get_device_name(0) if config.DEVICE == "cuda" else "CPU"
    return jsonify({
        "status":      "ok",
        "model_ready": ckpt_exists,
        "device":      config.DEVICE,
        "device_name": gpu_name,
        "diseases":    config.DISEASES,
        "num_classes": config.NUM_CLASSES,
        "max_file_mb": MAX_FILE_MB,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    ─────────────────
    Accepts:
      • multipart/form-data  →  field name: 'image'
      • application/json     →  { "image_b64": "data:image/png;base64,..." }

    Validation chain (server-side — in addition to client-side checks):
      1. File is present
      2. Extension is an allowed image extension (not .pdf, .mp3, etc.)
      3. MIME type starts with 'image/'
      4. Magic bytes confirm it's a real image (catches renamed files)
      5. File size is within limit
      6. PIL can decode it (not corrupted)
      7. Grayscale heuristic confirms it looks like a chest X-ray

    Response JSON:
    {
      "success":      true,
      "predictions":  { "Atelectasis": 72.4, ... },   // 0-100 percent
      "top_findings": [["Atelectasis", 72.4], ...],    // sorted desc
      "xai_combined": "<base64 PNG>",
      "xai_gradcam":  "<base64 PNG>",
      "xai_rollout":  "<base64 PNG>",
      "original_img": "<base64 PNG>",
      "top_disease":  "Atelectasis",
      "device":       "cuda" / "cpu",
      "warning":      null / "string"
    }
    """
    try:
        # 1. Extract raw bytes
        raw, fname, mime = _extract(request)
        if raw is None:
            return jsonify({
                "success": False,
                "error":   "No image received. Upload a file or use the camera.",
                "code":    "NO_IMAGE",
            }), 400

        # 2. Validate file type
        ok, reason = _validate(raw, fname, mime)
        if not ok:
            return jsonify({
                "success": False,
                "error":   reason,
                "code":    "INVALID_FILE",
            }), 422

        # 3. Decode image
        try:
            img = Image.open(io.BytesIO(raw))
            img.verify()
            img = Image.open(io.BytesIO(raw))
        except (UnidentifiedImageError, Exception) as e:
            return jsonify({
                "success": False,
                "error":   f"Cannot read image file: {e}. The file may be corrupted.",
                "code":    "CORRUPT_IMAGE",
            }), 422

        # 4. X-ray check
        valid, xray_reason = is_chest_xray(img)
        if not valid:
            return jsonify({
                "success": False,
                "error":   xray_reason,
                "code":    "NOT_XRAY",
                "hint":    "Please upload a grayscale PA or AP chest X-ray.",
                "preview": pil_to_base64(img.convert("RGB").resize((224, 224))),
            }), 422

        # 5. Inference
        engine  = get_engine()
        warning = None

        if engine is None:
            warning = (
                "Trained model not found. Showing demo predictions. "
                "Copy convitx_best.pth to the checkpoints/ folder."
            )
            result = _demo(img)
        else:
            result = engine.predict(img)

        result.update({"success": True, "warning": warning, "device": config.DEVICE})
        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"success": False, "error": str(e), "code": "VALIDATION_ERROR"}), 422
    except Exception:
        log.exception("Unhandled prediction error")
        return jsonify({
            "success": False,
            "error":   "Internal server error. Please try again.",
            "code":    "INTERNAL_ERROR",
        }), 500


@app.route("/api/reload", methods=["POST"])
def reload_model():
    """
    Force-reload the model from disk.
    Useful after copying a new convitx_best.pth checkpoint.
    The engine also auto-reloads on the next /api/predict call anyway.
    """
    global _engine, _ckpt_mtime
    _engine, _ckpt_mtime = None, 0.0
    if get_engine() is None:
        return jsonify({"success": False, "error": "Checkpoint not found."}), 404
    return jsonify({"success": True, "message": "Model reloaded.", "device": config.DEVICE})


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _extract(req):
    """Pull raw bytes, filename, and MIME type from the request."""
    if "image" in req.files:
        f = req.files["image"]
        if not f or f.filename == "":
            return None, None, None
        return f.read(), f.filename or "upload", f.content_type or ""

    data = req.get_json(silent=True)
    if data and "image_b64" in data:
        b64  = data["image_b64"]
        mime = "image/png"
        if b64.startswith("data:"):
            header, b64 = b64.split(",", 1)
            mime = header.split(":")[1].split(";")[0]
        try:
            return base64.b64decode(b64), "capture.png", mime
        except Exception:
            return None, None, None

    return None, None, None


def _validate(raw, fname, mime):
    """
    Three-layer file type validation:
      Layer 1 — Extension  : blocks known non-image types
      Layer 2 — MIME type  : must start with 'image/'
      Layer 3 — Magic bytes: imghdr sniffs actual file content
    Returns (True, "OK") or (False, reason_string).
    """
    # Layer 1: extension
    if fname and "." in fname:
        ext = Path(fname).suffix.lower()
        if ext in BLOCKED_EXT:
            return False, (
                f"'{ext}' files are not accepted. "
                "Please upload a chest X-ray image (JPG, PNG, TIFF, BMP, or WebP)."
            )
        if ext and ext not in ALLOWED_EXT:
            return False, (
                f"File type '{ext}' is not supported. "
                "Accepted: JPG, PNG, TIFF, BMP, WebP."
            )

    # Layer 2: MIME
    if mime and mime != "application/octet-stream":
        if not mime.startswith("image/"):
            return False, (
                f"'{mime}' is not an image type. "
                "Only image files are accepted."
            )

    # Layer 3: magic bytes (most reliable — catches .pdf renamed to .png etc.)
    if imghdr.what(None, h=raw[:32]) is None:
        return False, (
            "The file does not appear to be a valid image. "
            "PDF, audio, video, and document files are not accepted."
        )

    # Size
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        return False, (
            f"File is {size_mb:.1f} MB which exceeds the {MAX_FILE_MB} MB limit."
        )

    return True, "OK"


def _demo(img_pil):
    """Return placeholder predictions when no checkpoint exists yet."""
    import random
    rng   = random.Random(42)
    preds = {d: round(rng.uniform(1, 25), 2) for d in config.DISEASES}
    preds[config.DISEASES[0]] = round(rng.uniform(60, 80), 2)
    preds[config.DISEASES[2]] = round(rng.uniform(35, 55), 2)
    top  = sorted(preds.items(), key=lambda x: x[1], reverse=True)
    orig = pil_to_base64(img_pil.convert("RGB").resize((224, 224)))
    return {
        "predictions":  preds,
        "top_findings": top,
        "xai_combined": orig,
        "xai_gradcam":  orig,
        "xai_rollout":  orig,
        "original_img": orig,
        "top_disease":  top[0][0],
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="ConViT-X Web Server")
    p.add_argument("--host", "-H", default="0.0.0.0",
                   help="0.0.0.0 = accessible on LAN (needed for mobile camera)")
    p.add_argument("--port", "-p", type=int, default=5000)
    p.add_argument("--debug", "-d", action="store_true")
    args = p.parse_args()

    log.info("=" * 54)
    log.info("  ConViT-X Diagnostic Server")
    log.info("=" * 54)
    log.info(f"  Device      : {config.DEVICE.upper()}")
    log.info(f"  Checkpoint  : {CHECKPOINT}")
    log.info(f"  Max upload  : {MAX_FILE_MB} MB  (images only)")
    log.info(f"  Desktop URL : http://localhost:{args.port}")
    log.info(f"  Mobile URL  : http://<your-pc-ip>:{args.port}")
    log.info("=" * 54)

    get_engine()   # pre-load at startup for faster first request

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
