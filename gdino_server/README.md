# FastAPI Grounding DINO Image Upload Server

This project provides a FastAPI server for receiving images (e.g., from an ESP32-CAM), running object detection with Grounding DINO, and forwarding results to a Django REST API.

## Features
- Accepts image uploads via `/upload` endpoint (multipart/form-data)
- Runs Grounding DINO inference with a preset grocery prompt
- Saves uploaded images to `uploads/images/`
- Returns JSON with detection results
- Forwards results (image filename and detected object) to a Django REST API

## Requirements
- Python 3.8+
- pip
- (Recommended) Virtual environment

## Setup Instructions

### 1. Clone or Download the Project
Place all files in a folder, e.g., `gdino_server`.

### 2. Create and Activate a Virtual Environment
```
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```
pip install fastapi uvicorn pillow torch requests
# Install Grounding DINO and its dependencies as per its documentation
```

### 4. Download Grounding DINO Model Weights
- Place the config and weights files in the correct locations:
  - `GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py`
  - `weights/groundingdino_swint_ogc.pth`

### 5. Start the Server
```
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
- The server will be accessible on your local network at `http://<your-ip>:8000/upload`

### 6. Test the Server
- With curl:
  ```
  curl -X POST "http://127.0.0.1:8000/upload" -F "file=@your_image.jpg"
  ```
- Or from an ESP32-CAM using the provided Arduino code.

### 7. Django Integration
- The server will POST detection results to the Django REST API endpoint specified in `server.py` (`DJANGO_API_URL`).
- Update this URL to your actual Django endpoint.
