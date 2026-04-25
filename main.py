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


def _call_whisper(audio_bytes: bytes) -> dict:
    """Send raw audio bytes to HF Whisper and return transcription."""
    try:
        resp = _requests.post(
            _WHISPER_URL,
            headers=_HF_HEADERS,
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

    result = _call_whisper(audio_bytes)
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
            whisper = _call_whisper(audio_bytes)
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
