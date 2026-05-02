import os, io, cv2, time, base64, asyncio, csv
import numpy as np
import torch
import torch.nn as nn
import timm
from pathlib import Path
from collections import deque
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
MODEL_PATH = Path(__file__).parent / "models" / \
             "efficientnet_b2_finetuned_best.pth"
CLASS_NAMES = ["anger","disgust","fear",
               "happiness","neutral","sadness","surprise"]

print(f"✅ Device: {DEVICE}")

# ── Session storage ────────────────────────────────────────────────────────
session_data = {
    "start_time"  : None,
    "emotions"    : [],
    "dominant"    : {},
    "engagement"  : [],
    "prev_emotion": None,
    "streak"      : 0,
}

def reset_session():
    session_data["start_time"]   = datetime.now().isoformat()
    session_data["emotions"]     = []
    session_data["dominant"]     = {e: 0 for e in CLASS_NAMES}
    session_data["engagement"]   = []
    session_data["prev_emotion"] = None
    session_data["streak"]       = 0

def compute_engagement(emotion, scores, confidence):
    base_map = {
        "happiness": 0.85, "surprise": 0.80,
        "anger"    : 0.65, "fear"    : 0.55,
        "disgust"  : 0.50, "sadness" : 0.35,
        "neutral"  : 0.30,
    }
    base       = base_map.get(emotion, 0.5)
    conf_boost = max(-0.1, min(0.15, (confidence - 0.5) * 0.3))
    variety_boost = 0.0
    prev = session_data["prev_emotion"]
    if prev is not None and prev != emotion:
        variety_boost = 0.10
        session_data["streak"] = 0
    else:
        session_data["streak"] += 1
    streak_penalty = min(0.20,
        max(0, (session_data["streak"] - 10) * 0.01))
    neutral_prob  = scores.get("neutral", 0.5)
    express_boost = max(-0.1, min(0.15, (0.5 - neutral_prob) * 0.2))
    score = base + conf_boost + variety_boost \
            - streak_penalty + express_boost
    score = max(0.05, min(1.0, score))
    session_data["prev_emotion"] = emotion
    return round(score, 4)

def log_emotion(emotion, scores, source="vision"):
    if session_data["start_time"] is None:
        reset_session()
    confidence = scores.get(emotion, 0.5)
    entry = {
        "time"      : datetime.now().isoformat(),
        "emotion"   : emotion,
        "scores"    : scores,
        "source"    : source,
        "confidence": round(confidence, 4),
    }
    session_data["emotions"].append(entry)
    session_data["dominant"][emotion] = \
        session_data["dominant"].get(emotion, 0) + 1
    eng_score = compute_engagement(emotion, scores, confidence)
    session_data["engagement"].append({
        "time"   : entry["time"],
        "score"  : eng_score,
        "emotion": emotion,
    })

def get_learning_profile():
    if not session_data["emotions"]:
        return {"error": "No session data yet"}
    total = len(session_data["emotions"])
    distribution = {
        e: round(c / total * 100, 1)
        for e, c in session_data["dominant"].items() if c > 0
    }
    dominant = max(session_data["dominant"],
                   key=session_data["dominant"].get)
    recent_eng  = session_data["engagement"][-20:]
    avg_eng     = round(
        sum(e["score"] for e in recent_eng) /
        len(recent_eng) * 100, 1) if recent_eng else 0
    overall_eng = round(
        sum(e["score"] for e in session_data["engagement"]) /
        len(session_data["engagement"]) * 100, 1)
    eng_label = ("High"   if avg_eng >= 65
                 else "Medium" if avg_eng >= 40 else "Low")
    unique_emotions = len([e for e, c
                           in session_data["dominant"].items()
                           if c > 0])
    return {
        "session_start"     : session_data["start_time"],
        "total_detections"  : total,
        "dominant_emotion"  : dominant,
        "distribution"      : distribution,
        "avg_engagement"    : avg_eng,
        "overall_engagement": overall_eng,
        "engagement_label"  : eng_label,
        "emotion_variety"   : round(unique_emotions / 7 * 100, 1),
        "unique_emotions"   : unique_emotions,
        "timeline"          : session_data["emotions"][-50:],
        "engagement_trend"  : session_data["engagement"][-50:],
    }

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Emotion Recognition API")

# ── DB integration ─────────────────────────────────────────────────────────
from models_db import init_db
from routes import router as api_router

app.include_router(api_router)

@app.on_event("startup")
async def startup():
    await init_db()

# ── Static files ───────────────────────────────────────────────────────────
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Vision model ───────────────────────────────────────────────────────────
vision_model = None

def load_vision_model():
    global vision_model
    try:
        model = timm.create_model(
            "efficientnet_b2", pretrained=False, num_classes=7)
        if hasattr(model, 'classifier'):
            in_f = model.classifier.in_features
            model.classifier = nn.Linear(in_f, 7)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state, strict=False)
        model.eval().to(DEVICE)
        vision_model = model
        print("✅ Vision model loaded")
    except Exception as e:
        print(f"⚠️  Vision model failed: {e}")
        vision_model = None

load_vision_model()

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ── Face detector ──────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ── Helper functions ───────────────────────────────────────────────────────
def predict_face_emotion(frame_bgr):
    if vision_model is None:
        return "neutral", {e: 1/7 for e in CLASS_NAMES}
    try:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        aug = val_transform(image=img)["image"]
        aug = aug.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = vision_model(aug)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        emotion = CLASS_NAMES[np.argmax(probs)]
        scores  = {CLASS_NAMES[i]: round(float(probs[i]), 4)
                   for i in range(len(CLASS_NAMES))}
        return emotion, scores
    except Exception as e:
        print(f"Prediction error: {e}")
        return "neutral", {e: 1/7 for e in CLASS_NAMES}

def detect_and_predict(frame_bgr):
    try:
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(48, 48))
    except Exception:
        return frame_bgr, []
    results = []
    for (x, y, w, h) in faces:
        try:
            face = frame_bgr[y:y+h, x:x+w]
            if face.size == 0:
                continue
            emotion, scores = predict_face_emotion(face)
            results.append({
                "bbox"      : [int(x), int(y), int(w), int(h)],
                "emotion"   : emotion,
                "scores"    : scores,
                "confidence": round(scores.get(emotion, 0), 4),
            })
            color = (0, 255, 0)
            cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), color, 2)
            label = f"{emotion.upper()} " \
                    f"{scores.get(emotion, 0)*100:.0f}%"
            cv2.putText(frame_bgr, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)
        except Exception as e:
            print(f"Face processing error: {e}")
            continue
    return frame_bgr, results

# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status"   : "Emotion Recognition API running",
        "device"   : str(DEVICE),
        "model_loaded": vision_model is not None,
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model": vision_model is not None}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    try:
        html_path = Path(__file__).parent / "static" / "index.html"
        return HTMLResponse(content=html_path.read_text())
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>Error: {e}</h1>", status_code=500)

# ══════════════════════════════════════════════════════════════════════════
# FIXED: API endpoint accepts FILE UPLOAD (not base64 JSON)
# ══════════════════════════════════════════════════════════════════════════
@app.post("/api/detect-emotion")
async def detect_emotion_api(file: UploadFile = File(...)):
    """
    Accepts image file upload from React frontend,
    returns emotion predictions.
    """
    # Read file bytes
    contents = await file.read()
    
    # Decode image
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {
            "success": False,
            "message": "Invalid image",
            "emotions": {e: 0.0 for e in CLASS_NAMES},
            "dominant": "neutral"
        }
    
    # Detect faces and predict emotions
    _, results = detect_and_predict(frame)
    
    if not results:
        return {
            "success": False,
            "message": "No face detected",
            "emotions": {e: 0.0 for e in CLASS_NAMES},
            "dominant": "neutral"
        }
    
    # Get the first face result (largest face)
    face_result = results[0]
    emotion = face_result["emotion"]
    scores = face_result["scores"]
    
    # Log emotion for session tracking
    log_emotion(emotion, scores, source="vision")
    
    # Calculate current engagement
    current_eng = 0.0
    if session_data["engagement"]:
        current_eng = session_data["engagement"][-1]["score"]
    
    # Convert scores to percentages
    emotions_pct = {k: round(v * 100, 1) for k, v in scores.items()}
    
    return {
        "success": True,
        "emotions": emotions_pct,
        "dominant": emotion,
        "confidence": round(scores.get(emotion, 0) * 100, 1),
        "engagement": round(current_eng * 100, 1),
        "bbox": face_result.get("bbox")
    }

# ── Session endpoints ──────────────────────────────────────────────────────
@app.post("/session/start")
async def session_start():
    reset_session()
    return {"status": "Session started", "time": session_data["start_time"]}

@app.post("/session/reset")
async def session_reset():
    reset_session()
    return {"status": "Session reset", "time": session_data["start_time"]}

@app.post("/session/end")
async def session_end():
    """End session and return summary"""
    profile = get_learning_profile()
    return {"status": "Session ended", "profile": profile}

@app.get("/session/profile")
async def session_profile():
    return get_learning_profile()

@app.get("/session/timeline")
async def session_timeline():
    return {
        "emotions"  : session_data["emotions"][-100:],
        "engagement": session_data["engagement"][-100:],
    }

# ── Predict endpoints ──────────────────────────────────────────────────────
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        data  = await file.read()
        arr   = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Invalid image"}
        _, results = detect_and_predict(frame)
        for r in results:
            log_emotion(r["emotion"], r["scores"], source="vision")
        return {"faces": results, "count": len(results)}
    except Exception as e:
        return {"error": str(e)}

# ── Export endpoints ───────────────────────────────────────────────────────
@app.get("/session/export/csv")
async def export_csv():
    if not session_data["emotions"]:
        return {"error": "No session data to export"}
    try:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "timestamp", "emotion", "confidence", "source",
            "anger", "disgust", "fear", "happiness",
            "neutral", "sadness", "surprise", "engagement_score"
        ])
        eng_map = {e["time"]: e["score"]
                   for e in session_data["engagement"]}
        for entry in session_data["emotions"]:
            scores = entry.get("scores", {})
            eng    = eng_map.get(entry["time"], 0)
            writer.writerow([
                entry["time"],
                entry["emotion"],
                entry.get("confidence", ""),
                entry.get("source", "vision"),
                scores.get("anger",     ""),
                scores.get("disgust",   ""),
                scores.get("fear",      ""),
                scores.get("happiness", ""),
                scores.get("neutral",   ""),
                scores.get("sadness",   ""),
                scores.get("surprise",  ""),
                round(eng * 100, 1),
            ])
        output.seek(0)
        fname = f"emotion_session_" \
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition":
                     f"attachment; filename={fname}"})
    except Exception as e:
        return {"error": f"Export failed: {e}"}

@app.get("/session/export/report")
async def export_report():
    if not session_data["emotions"]:
        return {"error": "No session data to export"}
    try:
        profile = get_learning_profile()
        now     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dist_rows = ""
        for e, pct in sorted(profile["distribution"].items(),
                              key=lambda x: -x[1]):
            bar = "█" * int(pct / 5)
            dist_rows += f"""
            <tr>
              <td>{e.capitalize()}</td>
              <td>{pct}%</td>
              <td style="font-family:monospace;
                         color:#7c3aed">{bar}</td>
            </tr>"""
        timeline_rows = ""
        for entry in session_data["engagement"][-20:]:
            t   = entry["time"][11:19]
            eng = round(entry["score"] * 100, 1)
            em  = entry["emotion"]
            timeline_rows += f"""
            <tr>
              <td>{t}</td>
              <td>{em.capitalize()}</td>
              <td>{eng}%</td>
            </tr>"""
        eng_color = ("#22c55e" if profile["avg_engagement"] >= 65
                     else "#eab308" if profile["avg_engagement"] >= 40
                     else "#ef4444")
        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Emotion Session Report</title>
<style>
  body {{ font-family:-apple-system,sans-serif;
          max-width:800px; margin:40px auto;
          padding:0 24px; color:#1a1a1a; }}
  h1   {{ color:#7c3aed;
          border-bottom:2px solid #7c3aed;
          padding-bottom:8px; }}
  h2   {{ color:#444; margin-top:32px; }}
  .stat-grid {{ display:grid;
                grid-template-columns:repeat(3,1fr);
                gap:16px; margin:20px 0; }}
  .stat-box  {{ background:#f5f3ff; border-radius:12px;
                padding:16px; text-align:center;
                border:1px solid #e0d9f9; }}
  .stat-val  {{ font-size:28px; font-weight:700;
                color:#7c3aed; }}
  .stat-lbl  {{ font-size:12px; color:#666;
                margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse;
           margin-top:12px; }}
  th    {{ background:#7c3aed; color:white;
           padding:10px 14px; text-align:left;
           font-size:13px; }}
  td    {{ padding:8px 14px; font-size:13px;
           border-bottom:1px solid #eee; }}
  tr:hover td {{ background:#faf5ff; }}
  .badge {{ display:inline-block; padding:4px 12px;
            border-radius:20px; font-weight:700;
            font-size:16px; background:#f5f3ff;
            color:#7c3aed; border:1px solid #7c3aed; }}
  .footer {{ margin-top:40px; padding-top:16px;
             border-top:1px solid #eee;
             font-size:12px; color:#999; }}
</style>
</head>
<body>
  <h1>🧠 Emotion Recognition — Session Report</h1>
  <p style="color:#666">Generated: {now}</p>
  <p style="color:#666">
    Session start: {profile.get("session_start","N/A")}
  </p>
  <h2>Summary</h2>
  <div class="stat-grid">
    <div class="stat-box">
      <div class="stat-val">{profile["total_detections"]}</div>
      <div class="stat-lbl">Total Detections</div>
    </div>
    <div class="stat-box">
      <div class="stat-val" style="color:{eng_color}">
        {profile["avg_engagement"]}%
      </div>
      <div class="stat-lbl">
        Avg Engagement ({profile["engagement_label"]})
      </div>
    </div>
    <div class="stat-box">
      <div class="stat-val">{profile["unique_emotions"]}/7</div>
      <div class="stat-lbl">Emotion Variety</div>
    </div>
  </div>
  <p>Dominant emotion:
    <span class="badge">
      {profile["dominant_emotion"].upper()}
    </span>
  </p>
  <h2>Emotion Distribution</h2>
  <table>
    <tr>
      <th>Emotion</th><th>Percentage</th><th>Visual</th>
    </tr>
    {dist_rows}
  </table>
  <h2>Recent Engagement Timeline (last 20)</h2>
  <table>
    <tr><th>Time</th><th>Emotion</th><th>Engagement</th></tr>
    {timeline_rows}
  </table>
  <div class="footer">
    Generated by EmotiLearn |
    EfficientNet-B2 Vision Model
  </div>
</body>
</html>"""
        fname = f"emotion_report_" \
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        return StreamingResponse(
            io.BytesIO(html.encode()),
            media_type="text/html",
            headers={"Content-Disposition":
                     f"attachment; filename={fname}"})
    except Exception as e:
        return {"error": f"Report failed: {e}"}

# ── WebSocket Webcam ───────────────────────────────────────────────────────
@app.websocket("/ws/webcam")
async def webcam_ws(websocket: WebSocket):
    await websocket.accept()
    print("✅ Webcam WebSocket connected")
    cap = None
    try:
        loop = asyncio.get_event_loop()
        cap  = cv2.VideoCapture(0)
        if not cap.isOpened():
            await websocket.send_json(
                {"error": "Cannot open webcam — "
                          "check camera permissions"})
            await websocket.close()
            return
        consecutive_errors = 0
        while True:
            try:
                ret, frame = await loop.run_in_executor(
                    None, cap.read)
                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors > 10:
                        break
                    await asyncio.sleep(0.1)
                    continue
                consecutive_errors = 0
                annotated, results = detect_and_predict(frame)
                for r in results:
                    log_emotion(r["emotion"], r["scores"],
                                source="vision")
                _, buffer = cv2.imencode(
                    ".jpg", annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64 = base64.b64encode(buffer).decode("utf-8")
                current_eng = 0.0
                if session_data["engagement"]:
                    current_eng = \
                        session_data["engagement"][-1]["score"]
                await websocket.send_json({
                    "frame"     : b64,
                    "faces"     : results,
                    "count"     : len(results),
                    "engagement": round(current_eng * 100, 1),
                })
                await asyncio.sleep(0.05)
            except (RuntimeError, Exception) as e:
                if "close message" in str(e) \
                        or "disconnect" in str(e).lower():
                    print("✅ Webcam client disconnected cleanly")
                    break
                consecutive_errors += 1
                if consecutive_errors > 5:
                    break
                await asyncio.sleep(0.1)
    except Exception as e:
        if "close message" not in str(e):
            print(f"WebSocket error: {e}")
    finally:
        if cap is not None:
            cap.release()
        print("WebSocket closed")

# ══════════════════════════════════════════════════════════════════════════
# MULTI-FACE EXTENSION — APPENDED ONLY, NOTHING ABOVE THIS LINE WAS CHANGED
# Adds per-face tracking, engagement scoring, and reporting.
# Reuses the existing `predict_face_emotion`, `face_cascade`, and
# `CLASS_NAMES` from the code above. No existing function/variable modified.
# ══════════════════════════════════════════════════════════════════════════

# ── Multi-face session storage (separate from `session_data`) ──────────────
multi_session = {
    "start_time"  : None,
    "faces"       : {},    # face_id -> per-face state
    "active_faces": {},    # face_id -> {"bbox":[...], "last_ts": float}
    "next_face_id": 1,
}

TRACK_IOU_THRESHOLD  = 0.3    # IoU needed to match a face to a previous one
TRACK_EXPIRE_SECONDS = 5.0    # Remove from active tracker after N seconds

def _multi_reset():
    multi_session["start_time"]   = datetime.now().isoformat()
    multi_session["faces"]        = {}
    multi_session["active_faces"] = {}
    multi_session["next_face_id"] = 1

def _iou(a, b):
    """IoU between two [x, y, w, h] boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih   = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

def _assign_face_id(bbox, now_ts):
    """Match a bbox to an existing tracked face via IoU, or create a new id."""
    # Expire faces not seen recently
    expired = [fid for fid, info in multi_session["active_faces"].items()
               if now_ts - info["last_ts"] > TRACK_EXPIRE_SECONDS]
    for fid in expired:
        del multi_session["active_faces"][fid]

    # Best IoU match among active faces
    best_id, best_iou = None, 0.0
    for fid, info in multi_session["active_faces"].items():
        val = _iou(bbox, info["bbox"])
        if val > best_iou:
            best_iou, best_id = val, fid

    if best_id is not None and best_iou >= TRACK_IOU_THRESHOLD:
        multi_session["active_faces"][best_id] = {
            "bbox": bbox, "last_ts": now_ts}
        return best_id

    # New face
    new_id = multi_session["next_face_id"]
    multi_session["next_face_id"] += 1
    multi_session["active_faces"][new_id] = {
        "bbox": bbox, "last_ts": now_ts}
    return new_id

def _compute_engagement_for_face(face_state, emotion, scores, confidence):
    """Same formula as compute_engagement() but per-face state."""
    base_map = {
        "happiness": 0.85, "surprise": 0.80,
        "anger"    : 0.65, "fear"    : 0.55,
        "disgust"  : 0.50, "sadness" : 0.35,
        "neutral"  : 0.30,
    }
    base       = base_map.get(emotion, 0.5)
    conf_boost = max(-0.1, min(0.15, (confidence - 0.5) * 0.3))
    variety_boost = 0.0
    prev = face_state["prev_emotion"]
    if prev is not None and prev != emotion:
        variety_boost = 0.10
        face_state["streak"] = 0
    else:
        face_state["streak"] += 1
    streak_penalty = min(0.20,
        max(0, (face_state["streak"] - 10) * 0.01))
    neutral_prob  = scores.get("neutral", 0.5)
    express_boost = max(-0.1, min(0.15, (0.5 - neutral_prob) * 0.2))
    score = base + conf_boost + variety_boost \
            - streak_penalty + express_boost
    score = max(0.05, min(1.0, score))
    face_state["prev_emotion"] = emotion
    return round(score, 4)

def _log_emotion_for_face(face_id, emotion, scores):
    """Log one detection for a specific face, initialising state if needed."""
    if multi_session["start_time"] is None:
        _multi_reset()

    if face_id not in multi_session["faces"]:
        multi_session["faces"][face_id] = {
            "face_id"     : face_id,
            "first_seen"  : datetime.now().isoformat(),
            "last_seen"   : datetime.now().isoformat(),
            "emotions"    : [],
            "dominant"    : {e: 0 for e in CLASS_NAMES},
            "engagement"  : [],
            "prev_emotion": None,
            "streak"      : 0,
        }

    fs         = multi_session["faces"][face_id]
    confidence = scores.get(emotion, 0.5)
    now_iso    = datetime.now().isoformat()
    entry = {
        "time"      : now_iso,
        "emotion"   : emotion,
        "scores"    : scores,
        "confidence": round(confidence, 4),
    }
    fs["emotions"].append(entry)
    fs["dominant"][emotion] = fs["dominant"].get(emotion, 0) + 1
    fs["last_seen"] = now_iso
    eng = _compute_engagement_for_face(fs, emotion, scores, confidence)
    fs["engagement"].append({
        "time"   : now_iso,
        "score"  : eng,
        "emotion": emotion,
    })
    return eng

def _profile_for_face(face_id):
    """Learning profile for a single face."""
    if face_id not in multi_session["faces"]:
        return None
    fs = multi_session["faces"][face_id]
    if not fs["emotions"]:
        return None
    total = len(fs["emotions"])
    distribution = {e: round(c / total * 100, 1)
                    for e, c in fs["dominant"].items() if c > 0}
    dominant = max(fs["dominant"], key=fs["dominant"].get)
    recent_eng  = fs["engagement"][-20:]
    avg_eng     = round(
        sum(e["score"] for e in recent_eng) /
        len(recent_eng) * 100, 1) if recent_eng else 0
    overall_eng = round(
        sum(e["score"] for e in fs["engagement"]) /
        len(fs["engagement"]) * 100, 1) if fs["engagement"] else 0
    eng_label = ("High"   if avg_eng >= 65
                 else "Medium" if avg_eng >= 40 else "Low")
    unique_emotions = len([e for e, c
                           in fs["dominant"].items() if c > 0])
    return {
        "face_id"           : face_id,
        "first_seen"        : fs["first_seen"],
        "last_seen"         : fs["last_seen"],
        "total_detections"  : total,
        "dominant_emotion"  : dominant,
        "distribution"      : distribution,
        "avg_engagement"    : avg_eng,
        "overall_engagement": overall_eng,
        "engagement_label"  : eng_label,
        "emotion_variety"   : round(unique_emotions / 7 * 100, 1),
        "unique_emotions"   : unique_emotions,
        "timeline"          : fs["emotions"][-50:],
        "engagement_trend"  : fs["engagement"][-50:],
    }

def _multi_profile():
    """Class-wide overview plus per-face breakdown."""
    if not multi_session["faces"]:
        return {"error"        : "No multi-face session data yet",
                "session_start": multi_session["start_time"],
                "face_count"   : 0,
                "faces"        : []}
    faces = [_profile_for_face(fid) for fid in multi_session["faces"]]
    faces = [f for f in faces if f is not None]
    class_avg = round(
        sum(f["avg_engagement"] for f in faces) / len(faces), 1
    ) if faces else 0
    class_label = ("High"   if class_avg >= 65
                   else "Medium" if class_avg >= 40 else "Low")
    return {
        "session_start"         : multi_session["start_time"],
        "face_count"            : len(faces),
        "class_avg_engagement"  : class_avg,
        "class_engagement_label": class_label,
        "faces"                 : faces,
    }

# ── Multi-face inference (re-uses predict_face_emotion + face_cascade) ─────
def _detect_and_predict_multi(frame_bgr):
    """Like detect_and_predict but assigns face_id and logs per-face."""
    now_ts = time.time()
    try:
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(48, 48))
    except Exception:
        return frame_bgr, []

    results = []
    for (x, y, w, h) in faces:
        try:
            face_crop = frame_bgr[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue
            emotion, scores = predict_face_emotion(face_crop)
            bbox    = [int(x), int(y), int(w), int(h)]
            face_id = _assign_face_id(bbox, now_ts)
            eng     = _log_emotion_for_face(face_id, emotion, scores)
            results.append({
                "face_id"   : face_id,
                "bbox"      : bbox,
                "emotion"   : emotion,
                "scores"    : scores,
                "confidence": round(scores.get(emotion, 0), 4),
                "engagement": round(eng * 100, 1),
            })
            color = (0, 255, 0)
            cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), color, 2)
            label = f"#{face_id} {emotion.upper()} " \
                    f"{scores.get(emotion, 0)*100:.0f}%"
            cv2.putText(frame_bgr, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
        except Exception as e:
            print(f"Multi-face processing error: {e}")
            continue
    return frame_bgr, results

# ── Multi-face endpoints ───────────────────────────────────────────────────
@app.post("/api/detect-emotion-multi")
async def detect_emotion_multi(file: UploadFile = File(...)):
    """
    Accepts an image, returns per-face emotion predictions and engagement
    for ALL detected faces. Use this instead of /api/detect-emotion when
    you need multi-person classroom tracking.
    """
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"success": False, "message": "Invalid image",
                "faces": [], "count": 0}

    _, results = _detect_and_predict_multi(frame)
    if not results:
        return {"success": False, "message": "No face detected",
                "faces": [], "count": 0}

    class_eng = round(
        sum(r["engagement"] for r in results) / len(results), 1)
    return {
        "success"         : True,
        "count"           : len(results),
        "faces"           : results,
        "class_engagement": class_eng,
    }

@app.post("/session/multi/start")
async def multi_start():
    _multi_reset()
    return {"status": "Multi-face session started",
            "time"  : multi_session["start_time"]}

@app.post("/session/multi/reset")
async def multi_reset():
    _multi_reset()
    return {"status": "Multi-face session reset",
            "time"  : multi_session["start_time"]}

@app.get("/session/multi/profile")
async def multi_profile():
    return _multi_profile()

@app.get("/session/multi/face/{face_id}")
async def multi_face_profile(face_id: int):
    p = _profile_for_face(face_id)
    if p is None:
        raise HTTPException(status_code=404,
                            detail=f"Face #{face_id} not found")
    return p

@app.get("/session/multi/faces")
async def multi_list_faces():
    faces_list = []
    for fid, fs in multi_session["faces"].items():
        faces_list.append({
            "face_id"         : fid,
            "first_seen"      : fs["first_seen"],
            "last_seen"       : fs["last_seen"],
            "total_detections": len(fs["emotions"]),
            "active"          : fid in multi_session["active_faces"],
        })
    return {"count"        : len(faces_list),
            "faces"        : faces_list,
            "session_start": multi_session["start_time"]}

@app.get("/session/export/multi-csv")
async def export_multi_csv():
    if not multi_session["faces"]:
        return {"error": "No multi-face session data to export"}
    try:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "face_id", "timestamp", "emotion", "confidence",
            "anger", "disgust", "fear", "happiness",
            "neutral", "sadness", "surprise", "engagement_score"
        ])
        for fid, fs in multi_session["faces"].items():
            eng_map = {e["time"]: e["score"] for e in fs["engagement"]}
            for entry in fs["emotions"]:
                scores = entry.get("scores", {})
                eng    = eng_map.get(entry["time"], 0)
                writer.writerow([
                    fid,
                    entry["time"],
                    entry["emotion"],
                    entry.get("confidence", ""),
                    scores.get("anger",     ""),
                    scores.get("disgust",   ""),
                    scores.get("fear",      ""),
                    scores.get("happiness", ""),
                    scores.get("neutral",   ""),
                    scores.get("sadness",   ""),
                    scores.get("surprise",  ""),
                    round(eng * 100, 1),
                ])
        output.seek(0)
        fname = f"emotion_multi_session_" \
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition":
                     f"attachment; filename={fname}"})
    except Exception as e:
        return {"error": f"Multi CSV export failed: {e}"}

@app.get("/session/export/multi-report")
async def export_multi_report():
    if not multi_session["faces"]:
        return {"error": "No multi-face session data to export"}
    try:
        overview = _multi_profile()
        now      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        face_sections = ""
        for p in overview["faces"]:
            dist_rows = ""
            for e, pct in sorted(p["distribution"].items(),
                                  key=lambda x: -x[1]):
                bar = "█" * int(pct / 5)
                dist_rows += f"""
                <tr>
                  <td>{e.capitalize()}</td>
                  <td>{pct}%</td>
                  <td style="font-family:monospace;
                             color:#7c3aed">{bar}</td>
                </tr>"""
            eng_color = ("#22c55e" if p["avg_engagement"] >= 65
                         else "#eab308" if p["avg_engagement"] >= 40
                         else "#ef4444")
            face_sections += f"""
            <div class="face-card">
              <h2>👤 Face #{p["face_id"]}</h2>
              <p style="color:#666;font-size:12px">
                First seen: {p["first_seen"]} &nbsp;|&nbsp;
                Last seen: {p["last_seen"]}
              </p>
              <div class="stat-grid">
                <div class="stat-box">
                  <div class="stat-val">{p["total_detections"]}</div>
                  <div class="stat-lbl">Total Detections</div>
                </div>
                <div class="stat-box">
                  <div class="stat-val" style="color:{eng_color}">
                    {p["avg_engagement"]}%
                  </div>
                  <div class="stat-lbl">
                    Avg Engagement ({p["engagement_label"]})
                  </div>
                </div>
                <div class="stat-box">
                  <div class="stat-val">
                    {p["unique_emotions"]}/7
                  </div>
                  <div class="stat-lbl">Emotion Variety</div>
                </div>
              </div>
              <p>Dominant emotion:
                <span class="badge">
                  {p["dominant_emotion"].upper()}
                </span>
              </p>
              <table>
                <tr>
                  <th>Emotion</th><th>Percentage</th><th>Visual</th>
                </tr>
                {dist_rows}
              </table>
            </div>"""

        class_color = ("#22c55e" if overview["class_avg_engagement"] >= 65
                       else "#eab308" if overview["class_avg_engagement"] >= 40
                       else "#ef4444")
        total_detections = sum(f["total_detections"]
                               for f in overview["faces"])

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Multi-Face Emotion Report</title>
<style>
  body {{ font-family:-apple-system,sans-serif;
          max-width:900px; margin:40px auto;
          padding:0 24px; color:#1a1a1a; }}
  h1   {{ color:#7c3aed;
          border-bottom:2px solid #7c3aed;
          padding-bottom:8px; }}
  h2   {{ color:#444; margin-top:12px; }}
  .stat-grid {{ display:grid;
                grid-template-columns:repeat(3,1fr);
                gap:16px; margin:20px 0; }}
  .stat-box  {{ background:#f5f3ff; border-radius:12px;
                padding:16px; text-align:center;
                border:1px solid #e0d9f9; }}
  .stat-val  {{ font-size:24px; font-weight:700;
                color:#7c3aed; }}
  .stat-lbl  {{ font-size:12px; color:#666; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:12px; }}
  th    {{ background:#7c3aed; color:white;
           padding:10px 14px; text-align:left;
           font-size:13px; }}
  td    {{ padding:8px 14px; font-size:13px;
           border-bottom:1px solid #eee; }}
  tr:hover td {{ background:#faf5ff; }}
  .badge {{ display:inline-block; padding:4px 12px;
            border-radius:20px; font-weight:700;
            font-size:16px; background:#f5f3ff;
            color:#7c3aed; border:1px solid #7c3aed; }}
  .face-card {{ background:white; border:1px solid #e0d9f9;
                border-radius:16px; padding:24px;
                margin-bottom:24px;
                box-shadow:0 2px 8px rgba(124,58,237,0.06); }}
  .class-summary {{ background:#faf5ff;
                    border-left:4px solid #7c3aed;
                    padding:20px; border-radius:8px;
                    margin:20px 0; }}
  .footer {{ margin-top:40px; padding-top:16px;
             border-top:1px solid #eee;
             font-size:12px; color:#999; }}
</style>
</head>
<body>
  <h1>🧠 Multi-Face Emotion Session Report</h1>
  <p style="color:#666">Generated: {now}</p>
  <p style="color:#666">
    Session start: {overview.get("session_start", "N/A")}
  </p>

  <div class="class-summary">
    <h2 style="margin-top:0">📊 Class Overview</h2>
    <div class="stat-grid">
      <div class="stat-box">
        <div class="stat-val">{overview["face_count"]}</div>
        <div class="stat-lbl">Faces Tracked</div>
      </div>
      <div class="stat-box">
        <div class="stat-val" style="color:{class_color}">
          {overview["class_avg_engagement"]}%
        </div>
        <div class="stat-lbl">
          Class Avg ({overview["class_engagement_label"]})
        </div>
      </div>
      <div class="stat-box">
        <div class="stat-val">{total_detections}</div>
        <div class="stat-lbl">Total Detections</div>
      </div>
    </div>
  </div>

  <h2>Per-Face Breakdown</h2>
  {face_sections}

  <div class="footer">
    Generated by EmotiLearn |
    EfficientNet-B2 Vision Model | Multi-Face Mode
  </div>
</body>
</html>"""
        fname = f"emotion_multi_report_" \
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        return StreamingResponse(
            io.BytesIO(html.encode()),
            media_type="text/html",
            headers={"Content-Disposition":
                     f"attachment; filename={fname}"})
    except Exception as e:
        return {"error": f"Multi report failed: {e}"}

# ══════════════════════════════════════════════════════════════════════════
# END MULTI-FACE EXTENSION
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# MULTIMODAL EXTENSION — Whisper (speech→text) + RoBERTa (text→emotion)
# Calls Hugging Face Serverless Inference API — no local model loading.
# Nothing above this line was changed.
# ══════════════════════════════════════════════════════════════════════════

import requests as _requests   # avoid shadowing any existing `requests`

HF_TOKEN = os.getenv("HF_TOKEN", "")

_HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
}

_WHISPER_URL  = ("https://router.huggingface.co/hf-inference/"
                 "models/openai/whisper-large-v3")

_ROBERTA_URL  = ("https://router.huggingface.co/hf-inference/"
                 "models/j-hartmann/emotion-english-distilroberta-base")

# RoBERTa label → our 7-class label mapping
# j-hartmann model outputs: anger, disgust, fear, joy, neutral, sadness, surprise
_ROBERTA_LABEL_MAP = {
    "anger"   : "anger",
    "disgust" : "disgust",
    "fear"    : "fear",
    "joy"     : "happiness",   # map "joy" → "happiness"
    "neutral" : "neutral",
    "sadness" : "sadness",
    "surprise": "surprise",
}


def _call_whisper(audio_bytes: bytes, filename: str = "") -> dict:
    """Send raw audio bytes to HF Whisper and return transcription."""
    # Detect content type from filename
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    ct_map = {
        "wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
        "ogg": "audio/ogg", "webm": "audio/webm", "m4a": "audio/mp4",
        "aiff": "audio/aiff", "aif": "audio/aiff", "mp4": "audio/mp4",
    }
    content_type = ct_map.get(ext, "audio/wav")
    try:
        resp = _requests.post(
            _WHISPER_URL,
            headers={**_HF_HEADERS, "Content-Type": content_type},
            data=audio_bytes,
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            text = data.get("text", "")
            return {"success": True, "text": text.strip()}
        else:
            return {"success": False,
                    "error": f"Whisper API {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _call_roberta(text: str) -> dict:
    """Send text to HF RoBERTa emotion classifier and return scores."""
    try:
        resp = _requests.post(
            _ROBERTA_URL,
            headers={**_HF_HEADERS,
                     "Content-Type": "application/json"},
            json={"inputs": text},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            # Response shape: [[{"label":"joy","score":0.92}, ...]]
            raw_scores = data[0] if isinstance(data, list) and data else data
            if isinstance(raw_scores, list):
                # Map to our label scheme
                mapped = {}
                for item in raw_scores:
                    label = item.get("label", "").lower()
                    our_label = _ROBERTA_LABEL_MAP.get(label, label)
                    mapped[our_label] = round(item.get("score", 0), 4)
                dominant = max(mapped, key=mapped.get) if mapped else "neutral"
                return {
                    "success" : True,
                    "emotion" : dominant,
                    "scores"  : mapped,
                    "confidence": round(mapped.get(dominant, 0), 4),
                }
            return {"success": False, "error": "Unexpected response format"}
        else:
            return {"success": False,
                    "error": f"RoBERTa API {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _fuse_scores(face_scores: dict, text_scores: dict,
                 face_weight: float = 0.6,
                 text_weight: float = 0.4) -> dict:
    """
    Weighted fusion of facial and text emotion scores.
    Both dicts map emotion_name → probability (0-1).
    Returns fused scores + dominant + engagement.
    """
    all_emotions = set(list(face_scores.keys()) +
                       list(text_scores.keys()))
    fused = {}
    for emo in all_emotions:
        f = face_scores.get(emo, 0)
        t = text_scores.get(emo, 0)
        fused[emo] = round(f * face_weight + t * text_weight, 4)

    # Normalise so they sum to ~1
    total = sum(fused.values())
    if total > 0:
        fused = {e: round(v / total, 4) for e, v in fused.items()}

    dominant   = max(fused, key=fused.get) if fused else "neutral"
    confidence = fused.get(dominant, 0)

    # Compute engagement on fused result
    base_map = {
        "happiness": 0.85, "surprise": 0.80,
        "anger"    : 0.65, "fear"    : 0.55,
        "disgust"  : 0.50, "sadness" : 0.35,
        "neutral"  : 0.30,
    }
    eng = base_map.get(dominant, 0.5)
    eng += max(-0.1, min(0.15, (confidence - 0.5) * 0.3))
    eng = max(0.05, min(1.0, eng))

    return {
        "dominant"  : dominant,
        "scores"    : fused,
        "confidence": round(confidence, 4),
        "engagement": round(eng * 100, 1),
    }


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Send an audio file (wav, mp3, flac, webm, ogg) → get text back.
    Uses OpenAI Whisper large-v3 via Hugging Face Inference API.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        return {"success": False, "error": "Empty audio file"}

    result = _call_whisper(audio_bytes, file.filename or "")
    return result


@app.post("/api/text-emotion")
async def text_emotion(payload: dict):
    """
    Send {"text": "..."} → get text-based emotion prediction back.
    Uses j-hartmann/emotion-english-distilroberta-base via HF API.
    """
    text = payload.get("text", "").strip()
    if not text:
        return {"success": False, "error": "No text provided"}

    result = _call_roberta(text)
    return result


@app.post("/api/multimodal-detect")
async def multimodal_detect(
    image: UploadFile = File(...),
    audio: UploadFile = File(None),
):
    """
    Multimodal emotion detection:
      - Always runs facial emotion on the image (EfficientNet-B2, local)
      - If audio is provided, runs Whisper → RoBERTa → fuses with face
      - Returns face result, text result (if any), and fused result

    Frontend sends both a webcam frame and an audio clip.
    """
    # ── 1. Facial emotion (local, always runs) ─────────────────────────
    img_bytes = await image.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    face_result = {"success": False, "message": "No face detected"}
    if frame is not None:
        try:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1,
                minNeighbors=5, minSize=(48, 48))
            if len(faces) > 0:
                x, y, w, h = faces[0]
                crop = frame[y:y+h, x:x+w]
                emotion, scores = predict_face_emotion(crop)
                face_result = {
                    "success"   : True,
                    "emotion"   : emotion,
                    "scores"    : scores,
                    "confidence": round(scores.get(emotion, 0), 4),
                }
        except Exception as e:
            face_result = {"success": False, "error": str(e)}

    # ── 2. Audio → text → text emotion (HF API, only if audio sent) ────
    text_result  = None
    whisper_text = ""
    if audio is not None:
        audio_bytes = await audio.read()
        if audio_bytes and len(audio_bytes) > 100:
            whisper = _call_whisper(audio_bytes, audio.filename or "")
            if whisper.get("success"):
                whisper_text = whisper["text"]
                if whisper_text:
                    text_result = _call_roberta(whisper_text)

    # ── 3. Fusion ──────────────────────────────────────────────────────
    fused = None
    if face_result.get("success") and text_result and text_result.get("success"):
        # Convert face scores (0-100) to 0-1 for fusion
        face_01 = {k: v / 100 if v > 1 else v
                   for k, v in face_result["scores"].items()}
        fused = _fuse_scores(face_01, text_result["scores"])
    elif face_result.get("success"):
        # No audio or text failed — just return face as the result
        fused = {
            "dominant"  : face_result["emotion"],
            "scores"    : face_result["scores"],
            "confidence": face_result["confidence"],
            "engagement": 0,  # will be computed by existing engagement fn
        }

    return {
        "success"       : face_result.get("success", False),
        "face"          : face_result,
        "transcription" : whisper_text or None,
        "text_emotion"  : text_result,
        "fused"         : fused,
        "modalities_used": {
            "face" : face_result.get("success", False),
            "audio": text_result is not None and
                     text_result.get("success", False),
        },
    }


@app.get("/api/multimodal/status")
async def multimodal_status():
    """Health check for the multimodal pipeline."""
    hf_configured = bool(HF_TOKEN)

    # Quick test: ping Whisper with empty request to check auth
    whisper_ok = False
    roberta_ok = False
    if hf_configured:
        try:
            r = _requests.post(_WHISPER_URL, headers=_HF_HEADERS,
                               data=b"", timeout=5)
            # 400 = "bad audio" which means auth works
            # 401/403 = bad token
            whisper_ok = r.status_code in (200, 400, 422)
        except Exception:
            pass
        try:
            r = _requests.post(
                _ROBERTA_URL,
                headers={**_HF_HEADERS,
                         "Content-Type": "application/json"},
                json={"inputs": "test"},
                timeout=5)
            roberta_ok = r.status_code == 200
        except Exception:
            pass

    return {
        "hf_token_configured": hf_configured,
        "whisper_available"  : whisper_ok,
        "roberta_available"  : roberta_ok,
        "whisper_model"      : "openai/whisper-large-v3",
        "roberta_model"      : "j-hartmann/emotion-english-distilroberta-base",
        "fusion_weights"     : {"face": 0.6, "text": 0.4},
    }


# ══════════════════════════════════════════════════════════════════════════
# END MULTIMODAL EXTENSION
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# VOICE EMOTION EXTENSION — wav2vec2 (audio → emotion from tone/pitch)
# Calls HF Serverless Inference API — no local model loading.
# Nothing above this line was changed.
# ══════════════════════════════════════════════════════════════════════════

_VOICE_EMOTION_URL = ("https://router.huggingface.co/hf-inference/"
                      "models/r-f/wav2vec-english-speech-emotion-recognition")

# wav2vec2 model outputs: angry, disgust, fear, happy, neutral, sad, surprise
# These map directly to our 7-class scheme
_VOICE_LABEL_MAP = {
    "angry"   : "anger",
    "disgust" : "disgust",
    "fear"    : "fear",
    "happy"   : "happiness",
    "neutral" : "neutral",
    "sad"     : "sadness",
    "surprise": "surprise",
}


def _call_voice_emotion(audio_bytes: bytes, filename: str = "") -> dict:
    """
    Send raw audio bytes to HF wav2vec2 speech emotion model.
    Analyzes HOW you sound (tone, pitch, energy) — not what you say.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    ct_map = {
        "wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
        "ogg": "audio/ogg", "webm": "audio/webm", "m4a": "audio/mp4",
        "aiff": "audio/aiff", "mp4": "audio/mp4",
    }
    content_type = ct_map.get(ext, "audio/wav")
    try:
        resp = _requests.post(
            _VOICE_EMOTION_URL,
            headers={**_HF_HEADERS, "Content-Type": content_type},
            data=audio_bytes,
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            # Response: [{"label":"happy","score":0.92}, ...]
            if isinstance(data, list) and data:
                mapped = {}
                for item in data:
                    label = item.get("label", "").lower()
                    our_label = _VOICE_LABEL_MAP.get(label, label)
                    mapped[our_label] = round(item.get("score", 0), 4)
                dominant = max(mapped, key=mapped.get) if mapped else "neutral"
                return {
                    "success"   : True,
                    "emotion"   : dominant,
                    "scores"    : mapped,
                    "confidence": round(mapped.get(dominant, 0), 4),
                    "source"    : "voice_tone",
                }
            return {"success": False, "error": "Unexpected response format"}
        else:
            return {"success": False,
                    "error": f"Voice API {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/voice-emotion")
async def voice_emotion(file: UploadFile = File(...)):
    """
    Send an audio file → get emotion prediction based on VOICE TONE
    (pitch, energy, speaking style) — NOT text content.
    Uses wav2vec2 fine-tuned for speech emotion recognition via HF API.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        return {"success": False, "error": "Empty audio file"}

    result = _call_voice_emotion(audio_bytes, file.filename or "")
    return result


@app.post("/api/full-multimodal")
async def full_multimodal_detect(
    image: UploadFile = File(...),
    audio: UploadFile = File(None),
):
    """
    Complete tri-signal multimodal emotion detection:
      1. Face  (EfficientNet-B2, local) — HOW you look
      2. Voice (wav2vec2, HF API)       — HOW you sound
      3. Text  (Whisper→RoBERTa, HF API) — WHAT you say
    Returns individual results + tri-signal fusion.
    """
    # ── 1. Face emotion (local) ──────────────────────────────────────
    img_bytes = await image.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    face_result = {"success": False}
    if frame is not None:
        try:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1,
                minNeighbors=5, minSize=(48, 48))
            if len(faces) > 0:
                x, y, w, h = faces[0]
                crop = frame[y:y+h, x:x+w]
                emotion, scores = predict_face_emotion(crop)
                face_result = {
                    "success"   : True,
                    "emotion"   : emotion,
                    "scores"    : scores,
                    "confidence": round(scores.get(emotion, 0), 4),
                    "source"    : "face",
                }
        except Exception as e:
            face_result = {"success": False, "error": str(e)}

    # ── 2 & 3. Voice tone + Text (only if audio provided) ────────────
    voice_result = None
    text_result  = None
    whisper_text = ""

    if audio is not None:
        audio_bytes = await audio.read()
        if audio_bytes and len(audio_bytes) > 1000:
            # Voice tone emotion (wav2vec2)
            voice_result = _call_voice_emotion(
                audio_bytes, audio.filename or "")

            # Speech → text → text emotion (Whisper + RoBERTa)
            whisper = _call_whisper(audio_bytes, audio.filename or "")
            if whisper.get("success"):
                whisper_text = whisper["text"]
                if whisper_text:
                    text_result = _call_roberta(whisper_text)

    # ── 4. Tri-signal fusion ─────────────────────────────────────────
    fused = None
    signals = []
    weights = []

    if face_result.get("success"):
        face_01 = {k: v / 100 if v > 1 else v
                   for k, v in face_result["scores"].items()}
        signals.append(face_01)
        weights.append(0.50)   # face gets 50%

    if voice_result and voice_result.get("success"):
        signals.append(voice_result["scores"])
        weights.append(0.30)   # voice tone gets 30%

    if text_result and text_result.get("success"):
        signals.append(text_result["scores"])
        weights.append(0.20)   # text content gets 20%

    if signals:
        # Normalise weights to sum to 1
        w_total = sum(weights)
        weights = [w / w_total for w in weights]

        # Weighted average across all signals
        all_emotions = set()
        for s in signals:
            all_emotions.update(s.keys())

        fused_scores = {}
        for emo in all_emotions:
            val = sum(
                s.get(emo, 0) * w
                for s, w in zip(signals, weights))
            fused_scores[emo] = round(val, 4)

        # Normalise
        total = sum(fused_scores.values())
        if total > 0:
            fused_scores = {e: round(v / total, 4)
                           for e, v in fused_scores.items()}

        dominant   = max(fused_scores, key=fused_scores.get)
        confidence = fused_scores.get(dominant, 0)

        base_map = {
            "happiness": 0.85, "surprise": 0.80,
            "anger"    : 0.65, "fear"    : 0.55,
            "disgust"  : 0.50, "sadness" : 0.35,
            "neutral"  : 0.30,
        }
        eng = base_map.get(dominant, 0.5)
        eng += max(-0.1, min(0.15, (confidence - 0.5) * 0.3))
        eng = max(0.05, min(1.0, eng))

        fused = {
            "dominant"  : dominant,
            "scores"    : fused_scores,
            "confidence": round(confidence, 4),
            "engagement": round(eng * 100, 1),
        }

    return {
        "success"       : face_result.get("success", False),
        "face"          : face_result,
        "voice_tone"    : voice_result,
        "transcription" : whisper_text or None,
        "text_emotion"  : text_result,
        "fused"         : fused,
        "signals_used"  : {
            "face" : face_result.get("success", False),
            "voice": voice_result is not None and
                     voice_result.get("success", False),
            "text" : text_result is not None and
                     text_result.get("success", False),
        },
        "fusion_weights": dict(zip(
            ["face", "voice", "text"][:len(weights)],
            [round(w, 2) for w in weights]
        )) if weights else {},
    }


# ══════════════════════════════════════════════════════════════════════════
# END VOICE EMOTION EXTENSION
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — ADVANCED ENGAGEMENT STATES
# Rule-based layer on top of EfficientNet-B2 emotion history.
# Detects: Flow, Boredom, Productive Struggle, Confusion, Disengagement.
# Nothing above this line was changed.
# ══════════════════════════════════════════════════════════════════════════

ENGAGEMENT_STATES = {
    "flow": {
        "label": "Flow",
        "description": "Deep focus — sustained positive emotions with high engagement",
        "color": "#22c55e",
        "icon": "🟢",
    },
    "attentive": {
        "label": "Attentive",
        "description": "Actively engaged — moderate positive signals",
        "color": "#3b82f6",
        "icon": "🔵",
    },
    "boredom": {
        "label": "Boredom",
        "description": "Disinterested — sustained neutral with low engagement",
        "color": "#f97316",
        "icon": "🟠",
    },
    "productive_struggle": {
        "label": "Productive Struggle",
        "description": "Challenged but trying — negative emotions mixed with surprise",
        "color": "#eab308",
        "icon": "🟡",
    },
    "confusion": {
        "label": "Confusion",
        "description": "Lost — rapid emotion switching with declining engagement",
        "color": "#a855f7",
        "icon": "🟣",
    },
    "disengagement": {
        "label": "Disengagement",
        "description": "Checked out — sustained low engagement with negative emotions",
        "color": "#ef4444",
        "icon": "🔴",
    },
}


def _detect_engagement_state(emotion_window: list,
                             bbox_window: list = None,
                             frame_size: tuple = (640, 480)) -> dict:
    """
    Analyze a window of recent emotion detections (last 10-20 entries)
    and classify into an advanced engagement state.

    Each entry in emotion_window: {"emotion": str, "engagement_score": float}
    Each entry in bbox_window (optional): {"bbox": [x,y,w,h], "face_detected": bool}
    frame_size: (width, height) of the webcam frame for position normalization.
    """
    if not emotion_window or len(emotion_window) < 3:
        return {
            "state": "attentive",
            "confidence": 0.5,
            **ENGAGEMENT_STATES["attentive"],
        }

    emotions = [e.get("emotion", "neutral") for e in emotion_window]
    eng_scores = [e.get("engagement_score", 0) for e in emotion_window]

    # Averages
    avg_eng = sum(eng_scores) / len(eng_scores) if eng_scores else 0
    # Normalize to 0-1 if scores are 0-100
    if avg_eng > 1:
        avg_eng = avg_eng / 100

    # Emotion counts
    counts = {}
    for em in emotions:
        counts[em] = counts.get(em, 0) + 1
    total = len(emotions)

    # Percentages
    pct = {e: c / total for e, c in counts.items()}

    # Unique emotions
    unique = len(counts)

    # Transitions (how many times emotion changed)
    transitions = sum(1 for i in range(1, len(emotions))
                      if emotions[i] != emotions[i-1])
    transition_rate = transitions / max(1, len(emotions) - 1)

    # Dominant emotion
    dominant = max(counts, key=counts.get)

    # Positive emotions
    positive_pct = pct.get("happiness", 0) + pct.get("surprise", 0)
    negative_pct = (pct.get("anger", 0) + pct.get("fear", 0) +
                    pct.get("sadness", 0) + pct.get("disgust", 0))
    neutral_pct  = pct.get("neutral", 0)

    # Engagement trend (rising or falling)
    if len(eng_scores) >= 4:
        first_half = sum(eng_scores[:len(eng_scores)//2]) / (len(eng_scores)//2)
        second_half = sum(eng_scores[len(eng_scores)//2:]) / (len(eng_scores) - len(eng_scores)//2)
        eng_trend = second_half - first_half  # positive = rising
    else:
        eng_trend = 0

    # ── Bounding box / attention metrics (optional) ───────────────────
    face_absence_pct = 0.0       # % of frames with no face detected
    avg_center_offset = 0.0      # how far face center is from frame center (0-1)
    face_size_trend = 0.0        # positive = leaning in, negative = pulling back
    position_stability = 1.0     # 1 = perfectly still, 0 = very fidgety

    bbox_metrics_available = False

    if bbox_window and len(bbox_window) >= 3:
        bbox_metrics_available = True
        fw, fh = frame_size
        frame_cx, frame_cy = fw / 2, fh / 2

        detected_boxes = []
        absent_count = 0

        for entry in bbox_window:
            if not entry.get("face_detected", True):
                absent_count += 1
                continue
            bbox = entry.get("bbox")
            if bbox and len(bbox) == 4:
                detected_boxes.append(bbox)
            else:
                absent_count += 1

        face_absence_pct = absent_count / len(bbox_window)

        if detected_boxes:
            # Center offset: how far face center is from frame center
            offsets = []
            sizes = []
            positions_x = []
            positions_y = []
            for (bx, by, bw, bh) in detected_boxes:
                cx = bx + bw / 2
                cy = by + bh / 2
                # Normalize offset to 0-1
                dx = abs(cx - frame_cx) / frame_cx
                dy = abs(cy - frame_cy) / frame_cy
                offsets.append((dx + dy) / 2)
                sizes.append(bw * bh)
                positions_x.append(cx)
                positions_y.append(cy)

            avg_center_offset = sum(offsets) / len(offsets)

            # Face size trend: compare first half vs second half
            if len(sizes) >= 4:
                first_sizes = sum(sizes[:len(sizes)//2]) / (len(sizes)//2)
                second_sizes = sum(sizes[len(sizes)//2:]) / (len(sizes) - len(sizes)//2)
                # Positive = face getting bigger (leaning in)
                face_size_trend = ((second_sizes - first_sizes) /
                                   max(1, first_sizes)) * 10

            # Position stability: low std dev = stable, high = fidgety
            if len(positions_x) >= 3:
                import statistics
                std_x = statistics.stdev(positions_x) / max(1, fw)
                std_y = statistics.stdev(positions_y) / max(1, fh)
                # Convert to 0-1 where 1 = stable
                position_stability = max(0, 1 - (std_x + std_y) * 5)

    # ── State detection rules (emotion + bbox combined) ─────────────────
    state = "attentive"
    confidence = 0.5

    # Apply bbox penalties/bonuses to engagement
    bbox_penalty = 0.0
    if bbox_metrics_available:
        # Looking away from camera → penalty
        if avg_center_offset > 0.4:
            bbox_penalty += 0.15
        # Face frequently absent → penalty
        if face_absence_pct > 0.3:
            bbox_penalty += 0.20
        # Pulling back from screen → penalty
        if face_size_trend < -0.5:
            bbox_penalty += 0.10
        # Very fidgety → slight penalty
        if position_stability < 0.5:
            bbox_penalty += 0.05

    adjusted_eng = max(0, avg_eng - bbox_penalty)

    # FLOW: high engagement + positive emotions + facing camera + stable
    if (adjusted_eng >= 0.65 and positive_pct >= 0.5 and
            unique >= 2 and transition_rate >= 0.1 and
            face_absence_pct < 0.1):
        confidence = min(0.95, adjusted_eng * 0.5 +
                         positive_pct * 0.2 + position_stability * 0.3)
        state = "flow"

    # BOREDOM: low engagement + mostly neutral + few transitions
    # bbox boost: looking away or pulling back strengthens boredom signal
    elif (adjusted_eng < 0.40 and neutral_pct >= 0.5 and
            transition_rate < 0.15):
        state = "boredom"
        bbox_boost = (avg_center_offset * 0.3 +
                      face_absence_pct * 0.2) if bbox_metrics_available else 0
        confidence = min(0.95, (1 - adjusted_eng) * 0.4 +
                         neutral_pct * 0.3 + bbox_boost + 0.1)

    # PRODUCTIVE STRUGGLE: challenged but still facing screen
    elif (negative_pct >= 0.3 and pct.get("surprise", 0) >= 0.1 and
            0.35 <= avg_eng <= 0.65 and face_absence_pct < 0.2):
        state = "productive_struggle"
        confidence = min(0.90, negative_pct * 0.4 +
                         pct.get("surprise", 0) * 0.3 + 0.3)

    # CONFUSION: rapid switching + declining engagement
    elif transition_rate >= 0.5 and eng_trend < -0.05:
        state = "confusion"
        confidence = min(0.90, transition_rate * 0.5 + abs(eng_trend) * 2)

    # DISENGAGEMENT: very low engagement + face often absent or turned away
    elif (adjusted_eng < 0.30 and
            ((negative_pct + neutral_pct) >= 0.7 or face_absence_pct > 0.4)):
        state = "disengagement"
        absence_factor = face_absence_pct * 0.3 if bbox_metrics_available else 0
        confidence = min(0.95, (1 - adjusted_eng) * 0.4 +
                         (negative_pct + neutral_pct) * 0.3 + absence_factor)

    # ATTENTIVE: default moderate state
    else:
        confidence = min(0.85, adjusted_eng * 0.5 + 0.3)

    return {
        "state"     : state,
        "confidence": round(confidence, 3),
        **ENGAGEMENT_STATES[state],
        "metrics"   : {
            "avg_engagement"    : round(avg_eng, 3),
            "adjusted_engagement": round(adjusted_eng, 3),
            "positive_pct"      : round(positive_pct, 3),
            "negative_pct"      : round(negative_pct, 3),
            "neutral_pct"       : round(neutral_pct, 3),
            "transition_rate"   : round(transition_rate, 3),
            "engagement_trend"  : round(eng_trend, 4),
            "unique_emotions"   : unique,
            "dominant_emotion"  : dominant,
            "window_size"       : total,
        },
        "attention_metrics": {
            "available"          : bbox_metrics_available,
            "face_absence_pct"   : round(face_absence_pct, 3),
            "avg_center_offset"  : round(avg_center_offset, 3),
            "face_size_trend"    : round(face_size_trend, 3),
            "position_stability" : round(position_stability, 3),
            "bbox_penalty"       : round(bbox_penalty, 3),
        } if bbox_metrics_available else {"available": False},
    }


@app.post("/api/engagement-state")
async def analyze_engagement_state(payload: dict):
    """
    Accepts emotion detections + optional bbox data → returns advanced
    engagement state with attention metrics.

    Body: {
      "emotions": [{"emotion": "happiness", "engagement_score": 78}, ...],
      "bboxes": [{"bbox": [x,y,w,h], "face_detected": true}, ...],  (optional)
      "frame_size": [640, 480]  (optional)
    }
    """
    emotions = payload.get("emotions", [])
    if not emotions:
        return {"success": False, "error": "No emotion data provided"}

    bboxes = payload.get("bboxes", None)
    frame_size = tuple(payload.get("frame_size", [640, 480]))

    result = _detect_engagement_state(emotions, bboxes, frame_size)
    return {"success": True, **result}


@app.get("/api/engagement-states/info")
async def engagement_states_info():
    """Returns metadata about all detectable engagement states."""
    return {
        "states": ENGAGEMENT_STATES,
        "methodology": (
            "Rule-based classification over a sliding window of "
            "EfficientNet-B2 emotion predictions combined with spatial "
            "attention metrics derived from Haar Cascade face bounding boxes. "
            "Emotion features: mean engagement, emotion valence distribution, "
            "transition rate, and engagement trend. "
            "Attention features: face center offset from frame center "
            "(gaze proxy), face absence duration, face size trend "
            "(leaning in/out), and position stability (fidgeting). "
            "Six learning-relevant cognitive states are identified: "
            "Flow, Attentive, Boredom, Productive Struggle, Confusion, "
            "and Disengagement."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════
# END STEP 3 — ADVANCED ENGAGEMENT STATES
# ══════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — LECTURE RECORDING ANALYTICS
# Upload a video file → extract frames → run each through EfficientNet-B2
# → generate a complete engagement timeline with per-frame emotions.
# Nothing above this line was changed.
# ══════════════════════════════════════════════════════════════════════════

import tempfile
import json as _json

@app.post("/api/lecture/analyze")
async def analyze_lecture_video(
    file: UploadFile = File(...),
    sample_rate: int = 2,
):
    """
    Upload a lecture recording video → get a complete engagement timeline.

    - Extracts one frame every `sample_rate` seconds (default: every 2s)
    - Runs each frame through EfficientNet-B2 for emotion detection
    - Computes engagement score per frame
    - Detects advanced engagement states over sliding windows
    - Returns full timeline + summary statistics

    Supports: mp4, avi, mov, webm, mkv
    """
    # Save uploaded video to a temp file
    suffix = "." + (file.filename or "video.mp4").rsplit(".", 1)[-1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return {"success": False, "error": "Could not open video file"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        frame_interval = int(fps * sample_rate)

        timeline = []
        frame_idx = 0
        analyzed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp_sec = round(frame_idx / fps, 1)

                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1,
                        minNeighbors=5, minSize=(48, 48))

                    if len(faces) > 0:
                        # Analyze largest face
                        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
                        crop = frame[y:y+h, x:x+w]
                        emotion, scores = predict_face_emotion(crop)
                        confidence = scores.get(emotion, 0)

                        # Compute engagement
                        base_map = {
                            "happiness": 0.85, "surprise": 0.80,
                            "anger": 0.65, "fear": 0.55,
                            "disgust": 0.50, "sadness": 0.35,
                            "neutral": 0.30,
                        }
                        eng = base_map.get(emotion, 0.5)
                        eng += max(-0.1, min(0.15,
                                   (confidence / 100 - 0.5) * 0.3))
                        eng = max(0.05, min(1.0, eng))

                        entry = {
                            "timestamp"       : timestamp_sec,
                            "timestamp_fmt"   : f"{int(timestamp_sec//60)}:{int(timestamp_sec%60):02d}",
                            "emotion"         : emotion,
                            "confidence"      : round(confidence, 1),
                            "scores"          : scores,
                            "engagement"      : round(eng * 100, 1),
                            "faces_detected"  : len(faces),
                        }
                        timeline.append(entry)
                        analyzed += 1
                    else:
                        timeline.append({
                            "timestamp"     : timestamp_sec,
                            "timestamp_fmt" : f"{int(timestamp_sec//60)}:{int(timestamp_sec%60):02d}",
                            "emotion"       : None,
                            "faces_detected": 0,
                            "engagement"    : 0,
                        })
                except Exception as e:
                    print(f"Frame {frame_idx} error: {e}")

            frame_idx += 1

        cap.release()

        # ── Compute summary statistics ────────────────────────────────
        detected = [e for e in timeline if e.get("emotion")]
        if detected:
            avg_eng = round(sum(e["engagement"]
                                for e in detected) / len(detected), 1)
            emo_counts = {}
            for e in detected:
                em = e["emotion"]
                emo_counts[em] = emo_counts.get(em, 0) + 1
            dominant = max(emo_counts, key=emo_counts.get)
            distribution = {
                e: round(c / len(detected) * 100, 1)
                for e, c in emo_counts.items()
            }

            # Engagement states over sliding windows
            window_size = min(10, len(detected))
            state_timeline = []
            for i in range(0, len(detected), max(1, window_size // 2)):
                window = detected[i:i+window_size]
                if len(window) >= 3:
                    state = _detect_engagement_state([{
                        "emotion": e["emotion"],
                        "engagement_score": e["engagement"],
                    } for e in window])
                    state_timeline.append({
                        "from_time"  : window[0]["timestamp_fmt"],
                        "to_time"    : window[-1]["timestamp_fmt"],
                        "state"      : state["state"],
                        "label"      : state["label"],
                        "icon"       : state["icon"],
                        "confidence" : state["confidence"],
                    })

            # Engagement segments (high/medium/low periods)
            segments = []
            current_level = None
            seg_start = None
            for e in detected:
                level = ("high" if e["engagement"] >= 65
                         else "medium" if e["engagement"] >= 40
                         else "low")
                if level != current_level:
                    if current_level is not None:
                        segments.append({
                            "level": current_level,
                            "from" : seg_start,
                            "to"   : e["timestamp_fmt"],
                        })
                    current_level = level
                    seg_start = e["timestamp_fmt"]
            if current_level:
                segments.append({
                    "level": current_level,
                    "from" : seg_start,
                    "to"   : detected[-1]["timestamp_fmt"],
                })
        else:
            avg_eng = 0
            dominant = None
            distribution = {}
            state_timeline = []
            segments = []

        return {
            "success"         : True,
            "video_info"      : {
                "duration_sec"    : round(duration_sec, 1),
                "duration_fmt"    : f"{int(duration_sec//60)}:{int(duration_sec%60):02d}",
                "fps"             : round(fps, 1),
                "total_frames"    : total_frames,
                "frames_analyzed" : analyzed,
                "sample_rate_sec" : sample_rate,
            },
            "summary"         : {
                "avg_engagement"  : avg_eng,
                "dominant_emotion": dominant,
                "distribution"    : distribution,
                "total_detections": len(detected),
                "engagement_label": (
                    "High" if avg_eng >= 65
                    else "Medium" if avg_eng >= 40
                    else "Low"),
            },
            "timeline"        : timeline,
            "engagement_states": state_timeline,
            "engagement_segments": segments,
        }

    except Exception as e:
        return {"success": False, "error": f"Video analysis failed: {str(e)}"}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/api/lecture/analyze-multi")
async def analyze_lecture_multi(
    file: UploadFile = File(...),
    sample_rate: int = 2,
):
    """
    Like /api/lecture/analyze but uses multi-face detection.
    Returns per-face timelines for classroom recordings with multiple students.
    """
    suffix = "." + (file.filename or "video.mp4").rsplit(".", 1)[-1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return {"success": False, "error": "Could not open video file"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        frame_interval = int(fps * sample_rate)

        # Reset multi-face tracker for this analysis
        _multi_reset()

        all_results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp_sec = round(frame_idx / fps, 1)
                _, results = _detect_and_predict_multi(frame)
                all_results.append({
                    "timestamp"     : timestamp_sec,
                    "timestamp_fmt" : f"{int(timestamp_sec//60)}:{int(timestamp_sec%60):02d}",
                    "faces"         : results,
                    "count"         : len(results),
                })

            frame_idx += 1

        cap.release()

        # Get the per-face profile from the multi session
        profile = _multi_profile()

        return {
            "success"    : True,
            "video_info" : {
                "duration_sec"   : round(duration_sec, 1),
                "duration_fmt"   : f"{int(duration_sec//60)}:{int(duration_sec%60):02d}",
                "fps"            : round(fps, 1),
                "total_frames"   : total_frames,
                "sample_rate_sec": sample_rate,
            },
            "timeline"   : all_results,
            "profile"    : profile,
        }

    except Exception as e:
        return {"success": False, "error": f"Multi-face analysis failed: {str(e)}"}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════
# END STEP 2 — LECTURE RECORDING ANALYTICS
# ══════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — LMS INTEGRATION API SCAFFOLDING
# RESTful endpoints ready for LMS platforms (Moodle, Canvas, Blackboard)
# to consume. Follows LTI-compatible patterns.
# Nothing above this line was changed.
# ══════════════════════════════════════════════════════════════════════════

class LMSWebhookPayload(BaseModel):
    """Schema for LMS webhook events."""
    platform   : str = "moodle"        # moodle, canvas, blackboard
    event_type : str = "session_end"   # session_end, enrollment, grade_sync
    course_id  : str = ""
    student_id : str = ""
    data       : dict = {}

class LMSCourseSync(BaseModel):
    """Schema for syncing a course from LMS."""
    platform    : str
    external_id : str
    name        : str
    description : str = ""
    instructor  : str = ""
    students    : list = []


@app.get("/api/lms/status")
async def lms_status():
    """
    Returns the LMS integration readiness status.
    Lists which platforms are supported and what capabilities are available.
    """
    return {
        "lms_integration_ready": True,
        "api_version"          : "1.0",
        "supported_platforms"  : {
            "moodle"    : {
                "status"       : "api_ready",
                "auth_method"  : "API Token / LTI 1.3",
                "capabilities" : [
                    "course_sync",
                    "student_roster_import",
                    "engagement_export",
                    "grade_passback",
                    "webhook_events",
                ],
            },
            "canvas"    : {
                "status"       : "api_ready",
                "auth_method"  : "OAuth2 / LTI 1.3",
                "capabilities" : [
                    "course_sync",
                    "student_roster_import",
                    "engagement_export",
                    "grade_passback",
                    "webhook_events",
                ],
            },
            "blackboard": {
                "status"       : "api_ready",
                "auth_method"  : "REST API / LTI 1.3",
                "capabilities" : [
                    "course_sync",
                    "student_roster_import",
                    "engagement_export",
                    "webhook_events",
                ],
            },
        },
        "endpoints": {
            "course_sync"      : "POST /api/lms/course/sync",
            "student_report"   : "GET  /api/lms/student/{id}/report",
            "engagement_export": "GET  /api/lms/course/{id}/engagement",
            "webhook"          : "POST /api/lms/webhook",
            "health"           : "GET  /api/lms/status",
        },
    }


@app.post("/api/lms/course/sync")
async def lms_course_sync(data: LMSCourseSync):
    """
    Sync a course from an external LMS platform.
    Creates or updates the course in EmotiLearn's database.

    In production, this would be called by the LMS via LTI launch
    or REST API integration.
    """
    return {
        "success"    : True,
        "message"    : f"Course '{data.name}' synced from {data.platform}",
        "platform"   : data.platform,
        "external_id": data.external_id,
        "course"     : {
            "name"       : data.name,
            "description": data.description,
            "instructor" : data.instructor,
            "students"   : len(data.students),
        },
        "note": (
            "This is the API scaffolding. In production, this endpoint "
            "would create/update the course in the database and import "
            "the student roster from the LMS."
        ),
    }


@app.get("/api/lms/student/{student_id}/report")
async def lms_student_report(student_id: int):
    """
    LMS-compatible student engagement report.
    Returns data in a format that LMS gradebook integrations can consume.

    Can be used for:
    - Moodle grade passback via LTI
    - Canvas outcome reporting
    - Custom LMS dashboard widgets
    """
    # Pull real data from the session store if available
    profile = None
    try:
        from models_db import AsyncSessionLocal, Session, User
        from sqlalchemy import select, func, desc
        async with AsyncSessionLocal() as db:
            user_res = await db.execute(
                select(User).where(User.id == student_id))
            user = user_res.scalar_one_or_none()
            if not user:
                raise HTTPException(404, "Student not found")

            sess_res = await db.execute(
                select(Session)
                .where(Session.user_id == student_id)
                .order_by(desc(Session.started_at)))
            sessions = sess_res.scalars().all()

            if sessions:
                total = len(sessions)
                avg_eng = sum(s.avg_engagement or 0
                              for s in sessions) / total
                emo_counts = {}
                for s in sessions:
                    if s.dominant_emotion:
                        emo_counts[s.dominant_emotion] = \
                            emo_counts.get(s.dominant_emotion, 0) + 1
                dominant = (max(emo_counts, key=emo_counts.get)
                            if emo_counts else None)

                profile = {
                    "student_id"      : student_id,
                    "student_name"    : user.name,
                    "student_email"   : user.email,
                    "total_sessions"  : total,
                    "avg_engagement"  : round(avg_eng * 100, 1),
                    "dominant_emotion": dominant,
                    "last_session"    : (sessions[0].started_at.isoformat()
                                        if sessions[0].started_at else None),
                }
            else:
                profile = {
                    "student_id"     : student_id,
                    "student_name"   : user.name,
                    "total_sessions" : 0,
                    "avg_engagement" : 0,
                    "dominant_emotion": None,
                }
    except Exception as e:
        profile = {
            "student_id": student_id,
            "error"     : str(e),
        }

    return {
        "success"  : True,
        "format"   : "lti_outcome_compatible",
        "profile"  : profile,
        "lti_score": round(
            (profile.get("avg_engagement", 0) / 100)
            if profile else 0, 2),
        "note": (
            "lti_score is a 0-1 value suitable for LTI grade passback. "
            "Map to your LMS grading scale as needed."
        ),
    }


@app.get("/api/lms/course/{course_id}/engagement")
async def lms_course_engagement(course_id: int):
    """
    Export class-wide engagement data for an LMS course dashboard.
    Returns aggregated metrics suitable for LMS analytics widgets.
    """
    try:
        from models_db import (AsyncSessionLocal, Session, User,
                                Course, Lecture)
        from sqlalchemy import select, func, desc
        async with AsyncSessionLocal() as db:
            course_res = await db.execute(
                select(Course).where(Course.id == course_id))
            course = course_res.scalar_one_or_none()
            if not course:
                raise HTTPException(404, "Course not found")

            # Get all sessions for this course's lectures
            sess_res = await db.execute(
                select(Session, User.name.label("student_name"))
                .join(User, Session.user_id == User.id)
                .join(Lecture, Session.lecture_id == Lecture.id,
                      isouter=True)
                .where(Lecture.course_id == course_id)
                .order_by(desc(Session.started_at))
                .limit(100))
            rows = sess_res.all()

            students = {}
            for r in rows:
                sid = r.Session.user_id
                if sid not in students:
                    students[sid] = {
                        "name"       : r.student_name,
                        "sessions"   : 0,
                        "total_eng"  : 0,
                    }
                students[sid]["sessions"] += 1
                students[sid]["total_eng"] += (r.Session.avg_engagement or 0)

            student_list = []
            for sid, data in students.items():
                avg = round(data["total_eng"] / data["sessions"] * 100, 1) \
                    if data["sessions"] else 0
                student_list.append({
                    "student_id"    : sid,
                    "name"          : data["name"],
                    "sessions"      : data["sessions"],
                    "avg_engagement": avg,
                    "lti_score"     : round(avg / 100, 2),
                })

            class_avg = (round(
                sum(s["avg_engagement"] for s in student_list) /
                len(student_list), 1)
                if student_list else 0)

            return {
                "success"        : True,
                "course"         : {
                    "id"  : course.id,
                    "name": course.name,
                },
                "class_avg_engagement": class_avg,
                "total_students"     : len(student_list),
                "total_sessions"     : sum(s["sessions"]
                                           for s in student_list),
                "students"           : student_list,
            }
    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/lms/webhook")
async def lms_webhook(payload: LMSWebhookPayload):
    """
    Receive webhook events from LMS platforms.
    In production, this would trigger actions like:
    - Auto-start a session when a lecture begins
    - Sync grades when a session ends
    - Update student roster on enrollment changes
    """
    print(f"LMS Webhook: {payload.platform} / {payload.event_type}")
    return {
        "success"   : True,
        "received"  : True,
        "platform"  : payload.platform,
        "event_type": payload.event_type,
        "message"   : f"Webhook from {payload.platform} processed "
                      f"({payload.event_type})",
        "note": (
            "This is the webhook scaffolding. In production, each "
            "event_type would trigger specific actions in the system."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════
# END STEP 4 — LMS INTEGRATION API SCAFFOLDING
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# EXAM MONITORING EXTENSION — Attention & interaction tracking for
# online exams/quizzes. Uses face bbox (gaze region + absence streaks)
# + keystroke/click timestamps (interaction gaps) to produce a
# focus/distraction timeline and integrity score.
# Nothing above this line was changed.
# ══════════════════════════════════════════════════════════════════════════

# In-memory exam session storage (separate from learning sessions)
_exam_sessions = {}

GAZE_REGIONS = {
    "center"      : "Focused on exam area",
    "top_left"    : "Looking at top-left (possible second monitor)",
    "top_right"   : "Looking at top-right (possible second monitor)",
    "bottom_left" : "Looking down-left (possible phone/notes)",
    "bottom_right": "Looking down-right (possible phone/notes)",
    "top"         : "Looking up",
    "bottom"      : "Looking down (phone/desk)",
    "left"        : "Looking left (possible second screen)",
    "right"       : "Looking right (possible second screen)",
    "absent"      : "Face not detected (looking away)",
}

FOCUS_THRESHOLDS = {
    "center_tolerance"       : 0.25,    # 25% of frame from center = still "center"
    "absence_warning_sec"    : 5,       # 5s away = warning
    "absence_critical_sec"   : 15,      # 15s away = critical flag
    "interaction_gap_warning": 30,      # 30s no typing/clicking = warning
    "interaction_gap_critical": 60,     # 60s no interaction = critical
}


def _classify_gaze_region(bbox, frame_w, frame_h):
    """
    Given a face bounding box [x, y, w, h] and the frame dimensions,
    classify which region of the screen the student is looking at.
    """
    if bbox is None:
        return "absent"

    x, y, w, h = bbox
    cx = (x + w / 2) / frame_w    # 0-1 normalized
    cy = (y + h / 2) / frame_h    # 0-1 normalized

    tol = FOCUS_THRESHOLDS["center_tolerance"]

    # Center zone
    if (0.5 - tol) <= cx <= (0.5 + tol) and (0.5 - tol) <= cy <= (0.5 + tol):
        return "center"

    # Edge regions
    if cy < 0.35:
        if cx < 0.35:
            return "top_left"
        elif cx > 0.65:
            return "top_right"
        return "top"
    elif cy > 0.65:
        if cx < 0.35:
            return "bottom_left"
        elif cx > 0.65:
            return "bottom_right"
        return "bottom"
    else:
        if cx < 0.35:
            return "left"
        elif cx > 0.65:
            return "right"
        return "center"


def _analyze_exam_attention(detections: list, interactions: list,
                            frame_size: tuple = (640, 480)) -> dict:
    """
    Full exam attention analysis.

    detections: [{"timestamp": float, "bbox": [x,y,w,h] or None,
                  "face_detected": bool, "emotion": str,
                  "engagement_score": float}, ...]
    interactions: [{"timestamp": float, "type": "keypress"|"click"}, ...]
    """
    fw, fh = frame_size
    total_detections = len(detections)

    if total_detections == 0:
        return {"success": False, "error": "No detection data"}

    # ── Gaze region timeline ──────────────────────────────────────────
    gaze_timeline = []
    region_counts = {}
    for d in detections:
        bbox = d.get("bbox") if d.get("face_detected", True) else None
        region = _classify_gaze_region(bbox, fw, fh)
        gaze_timeline.append({
            "timestamp": d.get("timestamp", 0),
            "region"   : region,
            "emotion"  : d.get("emotion", "neutral"),
        })
        region_counts[region] = region_counts.get(region, 0) + 1

    # Region percentages
    region_pct = {r: round(c / total_detections * 100, 1)
                  for r, c in region_counts.items()}
    focus_pct = region_pct.get("center", 0)

    # ── Face absence streaks ──────────────────────────────────────────
    absence_streaks = []
    current_streak = 0
    streak_start = None
    sample_interval = 3   # seconds between detections

    for i, d in enumerate(detections):
        if not d.get("face_detected", True) or d.get("bbox") is None:
            if current_streak == 0:
                streak_start = d.get("timestamp", i * sample_interval)
            current_streak += 1
        else:
            if current_streak > 0:
                streak_duration = current_streak * sample_interval
                absence_streaks.append({
                    "start_sec"   : streak_start,
                    "duration_sec": streak_duration,
                    "severity"    : (
                        "critical" if streak_duration >= FOCUS_THRESHOLDS["absence_critical_sec"]
                        else "warning" if streak_duration >= FOCUS_THRESHOLDS["absence_warning_sec"]
                        else "normal"),
                })
            current_streak = 0

    # Handle streak at end
    if current_streak > 0:
        streak_duration = current_streak * sample_interval
        absence_streaks.append({
            "start_sec"   : streak_start,
            "duration_sec": streak_duration,
            "severity"    : (
                "critical" if streak_duration >= FOCUS_THRESHOLDS["absence_critical_sec"]
                else "warning" if streak_duration >= FOCUS_THRESHOLDS["absence_warning_sec"]
                else "normal"),
        })

    total_absence_sec = sum(s["duration_sec"] for s in absence_streaks)
    session_duration = (detections[-1].get("timestamp", 0) -
                        detections[0].get("timestamp", 0)) + sample_interval
    absence_pct = round(total_absence_sec / max(1, session_duration) * 100, 1)

    # ── Interaction gap analysis ──────────────────────────────────────
    interaction_gaps = []
    if interactions and len(interactions) >= 2:
        sorted_interactions = sorted(interactions, key=lambda x: x.get("timestamp", 0))
        for i in range(1, len(sorted_interactions)):
            gap = (sorted_interactions[i]["timestamp"] -
                   sorted_interactions[i-1]["timestamp"])
            if gap >= FOCUS_THRESHOLDS["interaction_gap_warning"]:
                interaction_gaps.append({
                    "start_sec"   : sorted_interactions[i-1]["timestamp"],
                    "duration_sec": round(gap, 1),
                    "severity"    : (
                        "critical" if gap >= FOCUS_THRESHOLDS["interaction_gap_critical"]
                        else "warning"),
                })

    avg_interaction_gap = 0
    if interactions and len(interactions) >= 2:
        sorted_ts = sorted(x["timestamp"] for x in interactions)
        gaps = [sorted_ts[i] - sorted_ts[i-1] for i in range(1, len(sorted_ts))]
        avg_interaction_gap = round(sum(gaps) / len(gaps), 1) if gaps else 0

    # ── Distraction events (looking away + not typing) ────────────────
    distraction_events = []
    for streak in absence_streaks:
        if streak["severity"] in ("warning", "critical"):
            # Check if there was also no interaction during this period
            start = streak["start_sec"]
            end = start + streak["duration_sec"]
            had_interaction = any(
                start <= i.get("timestamp", 0) <= end
                for i in (interactions or []))
            distraction_events.append({
                "start_sec"   : start,
                "duration_sec": streak["duration_sec"],
                "looked_away" : True,
                "was_typing"  : had_interaction,
                "severity"    : streak["severity"],
            })

    # ── Overall integrity / focus score (0-100) ───────────────────────
    # Starts at 100, penalties deducted
    score = 100.0

    # Penalty for non-center gaze time
    off_center_pct = 100 - focus_pct
    score -= off_center_pct * 0.3    # max -30

    # Penalty for absence
    score -= absence_pct * 0.4       # max -40

    # Penalty for critical absence streaks
    critical_streaks = [s for s in absence_streaks
                        if s["severity"] == "critical"]
    score -= len(critical_streaks) * 5   # -5 per critical streak

    # Penalty for long interaction gaps
    score -= len(interaction_gaps) * 3   # -3 per long gap

    score = max(0, min(100, round(score, 1)))

    focus_label = ("Excellent" if score >= 85
                   else "Good" if score >= 70
                   else "Moderate" if score >= 50
                   else "Poor" if score >= 30
                   else "Critical")

    return {
        "success"       : True,
        "focus_score"   : score,
        "focus_label"   : focus_label,
        "session_duration_sec": round(session_duration, 1),
        "gaze_analysis" : {
            "focus_pct"    : focus_pct,
            "region_breakdown": region_pct,
            "region_descriptions": {
                r: GAZE_REGIONS.get(r, r) for r in region_pct
            },
        },
        "absence_analysis": {
            "total_absence_sec"  : total_absence_sec,
            "absence_pct"        : absence_pct,
            "streaks"            : absence_streaks,
            "longest_streak_sec" : (max(s["duration_sec"] for s in absence_streaks)
                                    if absence_streaks else 0),
            "critical_count"     : len(critical_streaks),
        },
        "interaction_analysis": {
            "total_interactions"  : len(interactions or []),
            "avg_gap_sec"         : avg_interaction_gap,
            "long_gaps"           : interaction_gaps,
        },
        "distraction_events": distraction_events,
        "gaze_timeline"    : gaze_timeline[-100:],   # last 100 entries
    }


@app.post("/api/exam/start")
async def exam_start(payload: dict):
    """
    Start an exam monitoring session.
    Body: {"exam_id": "quiz_1", "student_id": 1}
    """
    exam_id = payload.get("exam_id", f"exam_{int(time.time())}")
    student_id = payload.get("student_id", 0)
    _exam_sessions[exam_id] = {
        "student_id"  : student_id,
        "started_at"  : datetime.now().isoformat(),
        "detections"  : [],
        "interactions" : [],
        "frame_size"  : (640, 480),
    }
    return {
        "success"   : True,
        "exam_id"   : exam_id,
        "started_at": _exam_sessions[exam_id]["started_at"],
    }


@app.post("/api/exam/detection")
async def exam_add_detection(payload: dict):
    """
    Log a single detection during an exam.
    Body: {
      "exam_id": "quiz_1",
      "timestamp": 12.5,
      "bbox": [x, y, w, h] or null,
      "face_detected": true,
      "emotion": "neutral",
      "engagement_score": 45
    }
    """
    exam_id = payload.get("exam_id", "")
    if exam_id not in _exam_sessions:
        return {"success": False, "error": "Exam session not found"}

    _exam_sessions[exam_id]["detections"].append({
        "timestamp"       : payload.get("timestamp", 0),
        "bbox"            : payload.get("bbox"),
        "face_detected"   : payload.get("face_detected", True),
        "emotion"         : payload.get("emotion", "neutral"),
        "engagement_score": payload.get("engagement_score", 0),
    })

    if payload.get("frame_size"):
        _exam_sessions[exam_id]["frame_size"] = tuple(payload["frame_size"])

    return {"success": True, "detections_count": len(
        _exam_sessions[exam_id]["detections"])}


@app.post("/api/exam/interaction")
async def exam_add_interaction(payload: dict):
    """
    Log keyboard/mouse interactions during an exam.
    Body: {
      "exam_id": "quiz_1",
      "interactions": [
        {"timestamp": 5.2, "type": "keypress"},
        {"timestamp": 7.8, "type": "click"},
        ...
      ]
    }
    """
    exam_id = payload.get("exam_id", "")
    if exam_id not in _exam_sessions:
        return {"success": False, "error": "Exam session not found"}

    new_interactions = payload.get("interactions", [])
    _exam_sessions[exam_id]["interactions"].extend(new_interactions)

    return {"success": True, "total_interactions": len(
        _exam_sessions[exam_id]["interactions"])}


@app.post("/api/exam/end")
async def exam_end(payload: dict):
    """
    End an exam session and get the full attention analysis report.
    Body: {"exam_id": "quiz_1"}
    """
    exam_id = payload.get("exam_id", "")
    if exam_id not in _exam_sessions:
        return {"success": False, "error": "Exam session not found"}

    session = _exam_sessions[exam_id]
    result = _analyze_exam_attention(
        session["detections"],
        session["interactions"],
        session["frame_size"],
    )

    result["exam_id"]    = exam_id
    result["student_id"] = session["student_id"]
    result["started_at"] = session["started_at"]
    result["ended_at"]   = datetime.now().isoformat()

    # Keep session for retrieval, but mark as ended
    session["ended_at"] = result["ended_at"]
    session["result"]   = result

    return result


@app.get("/api/exam/{exam_id}/report")
async def exam_report(exam_id: str):
    """
    Retrieve the attention analysis report for a completed exam.
    """
    if exam_id not in _exam_sessions:
        raise HTTPException(404, "Exam session not found")

    session = _exam_sessions[exam_id]
    if "result" in session:
        return session["result"]

    # Generate report on the fly if exam hasn't been formally ended
    result = _analyze_exam_attention(
        session["detections"],
        session["interactions"],
        session["frame_size"],
    )
    result["exam_id"]    = exam_id
    result["student_id"] = session["student_id"]
    result["started_at"] = session["started_at"]
    result["status"]     = "in_progress"
    return result


@app.post("/api/exam/analyze")
async def exam_analyze_batch(payload: dict):
    """
    One-shot analysis — send all data at once instead of streaming.
    Body: {
      "detections": [{"timestamp": 0, "bbox": [x,y,w,h], "face_detected": true,
                       "emotion": "neutral", "engagement_score": 50}, ...],
      "interactions": [{"timestamp": 5.2, "type": "keypress"}, ...],
      "frame_size": [640, 480]
    }
    """
    detections = payload.get("detections", [])
    interactions = payload.get("interactions", [])
    frame_size = tuple(payload.get("frame_size", [640, 480]))

    if not detections:
        return {"success": False, "error": "No detection data"}

    return _analyze_exam_attention(detections, interactions, frame_size)


# ══════════════════════════════════════════════════════════════════════════
# END EXAM MONITORING EXTENSION
# ══════════════════════════════════════════════════════════════════════════
