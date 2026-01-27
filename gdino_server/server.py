import os
import time
import io
import torch
from typing import Dict, Any, List
import requests

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from groundingdino.util.inference import load_model, load_image, predict

app = FastAPI()
# -----------------------------
# Django API Endpoint
# -----------------------------
DJANGO_API_URL = "https://stockbot-api-yu48.onrender.com/api/inventory/ingestion/classification/" 

# Sending data to Django endpoint
def send_to_django(payload):
    try:
        response = requests.post(DJANGO_API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"Failed to send to Django: {e}")
        if e.response is not None:
            print(f"Django response content: {e.response.text}")
        return None
    except Exception as e:
        print(f"Failed to send to Django: {e}")
        return None

# -----------------------------
# Grounding DINO config
# -----------------------------
GDINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_WEIGHTS = "weights/groundingdino_swint_ogc.pth"

# Preset grocery prompt 
GROCERY_PROMPT = (
    "apple . banana . orange . "
    "milk carton . cereal box . bread loaf . "
    "pasta bag . chips bag . soda can ."
)

BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.20

print("[SERVER] Loading GroundingDINO...")
gdino_model = load_model(GDINO_CONFIG, GDINO_WEIGHTS)
gdino_model = gdino_model.to("cpu")  # 🔑 FORCE CPU
print("[SERVER] GroundingDINO loaded (CPU mode).")

def pick_top_detection(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not detections:
        return {"object_name": "unknown", "confidence": 0.0}
    top = max(detections, key=lambda d: d.get("confidence", 0.0))
    return {"object_name": top.get("class_name", "unknown"), "confidence": float(top.get("confidence", 0.0))}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect-grocery")
async def detect_grocery(file: UploadFile = File(...)):
    from datetime import datetime

    # Timestamp: when server receives image
    ts_received = datetime.now()
    print(f"[TIMESTAMP] Image received at: {ts_received.isoformat()}")

    image_bytes = await file.read()

    # Use the original uploaded filename
    image_filename = file.filename

    # Validate image
    try:
        Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return JSONResponse({"status": "error", "message": "Invalid image"}, status_code=400, media_type="text/plain")

    # Timestamp: before detection
    ts_detection_start = datetime.now()

    # Run Grounding DINO
    start = time.time()
    _, gdino_image = load_image(io.BytesIO(image_bytes))

    _, logits, phrases = predict(
        model=gdino_model,
        image=gdino_image,
        caption=GROCERY_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device="cpu"
    )
    inference_time_ms = (time.time() - start) * 1000.0

    # Timestamp: after detection
    ts_detection_end = datetime.now()
    detections = [{"class_name": p, "confidence": float(s)} for s, p in zip(logits, phrases)]
    top = pick_top_detection(detections)
    print(f"[TIMESTAMP] Detection done at: {ts_detection_end.isoformat()} | Object: {top['object_name']} | Confidence: {top['confidence']:.3f} | Detection duration: {(ts_detection_end - ts_detection_start).total_seconds():.3f} seconds")

    result = {
        "status": "success",
        "object_name": top["object_name"],
        "confidence": round(top["confidence"], 3),
        "inference_time_ms": round(inference_time_ms, 1),
        "num_detections": len(detections),
        "image_filename": image_filename,
        "prompt_used": GROCERY_PROMPT,
    }

    # Timestamp: before sending to Django
    ts_send_django = datetime.now()

    # Django endpoint
    django_payload = {
        "bot_id": 12345, 
        "image_id": image_filename,
        "classification": top["object_name"]
    }
    
    print(f"[DEBUG] Django payload: {django_payload}")
    django_response = send_to_django(django_payload)
    if django_response and (isinstance(django_response, dict) and django_response.get('status') == 'success' or django_response == 'success'):
        print(f"[INFO] Django received image and returned success for image_id: {image_filename}")

    # Timestamp: after Django response
    ts_django_response = datetime.now()
    print(f"[TIMESTAMP] Django response at: {ts_django_response.isoformat()} | Response: {django_response} | Django roundtrip duration: {(ts_django_response - ts_send_django).total_seconds():.3f} seconds")

    # Sending result back to the esp32 camera
    return JSONResponse(result, media_type="text/plain")
    
