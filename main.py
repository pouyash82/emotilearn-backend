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
# MULTI-FACE DETECTION & PER-FACE REPORTS
# Appended to existing code — no original lines modified
# ══════════════════════════════════════════════════════════════════════════
import math

_mf = {"faces": {}, "next_id": 1, "positions": []}

def _mf_reset():
    _mf["faces"] = {}
    _mf["next_id"] = 1
    _mf["positions"] = []

def _mf_assign_id(bbox):
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    now = time.time()
    best, best_d = None, 120
    for p in _mf["positions"]:
        if now - p["t"] > 3.0:
            continue
        d = math.sqrt((cx - p["cx"])**2 + (cy - p["cy"])**2)
        if d < best_d:
            best_d, best = d, p
    if best:
        best["cx"], best["cy"], best["t"] = cx, cy, now
        return best["id"]
    fid = f"face_{_mf['next_id']}"
    _mf["next_id"] += 1
    _mf["positions"].append({"id": fid, "cx": cx, "cy": cy, "t": now})
    return fid

def _mf_get(fid):
    if fid not in _mf["faces"]:
        _mf["faces"][fid] = {
            "emotions": [], "dominant": {e: 0 for e in CLASS_NAMES},
            "engagement": [], "prev": None, "streak": 0,
            "first": datetime.now().isoformat(), "last": datetime.now().isoformat(),
        }
    return _mf["faces"][fid]

def _mf_log(fid, emotion, scores):
    fd = _mf_get(fid)
    conf = scores.get(emotion, 0.5)
    fd["emotions"].append({"time": datetime.now().isoformat(), "emotion": emotion,
        "scores": scores, "confidence": round(conf, 4)})
    fd["dominant"][emotion] = fd["dominant"].get(emotion, 0) + 1
    fd["last"] = datetime.now().isoformat()
    bm = {"happiness":0.85,"surprise":0.80,"anger":0.65,"fear":0.55,
          "disgust":0.50,"sadness":0.35,"neutral":0.30}
    base = bm.get(emotion, 0.5)
    cb = max(-0.1, min(0.15, (conf - 0.5) * 0.3))
    vb = 0.0
    if fd["prev"] and fd["prev"] != emotion:
        vb = 0.10
        fd["streak"] = 0
    else:
        fd["streak"] += 1
    sp = min(0.20, max(0, (fd["streak"] - 10) * 0.01))
    eb = max(-0.1, min(0.15, (0.5 - scores.get("neutral", 0.5)) * 0.2))
    eng = max(0.05, min(1.0, base + cb + vb - sp + eb))
    fd["prev"] = emotion
    fd["engagement"].append({"time": fd["last"], "score": round(eng, 4), "emotion": emotion})
    return round(eng, 4)

def _mf_profile(fid):
    if fid not in _mf["faces"]:
        return {"error": f"No data for {fid}"}
    fd = _mf["faces"][fid]
    if not fd["emotions"]:
        return {"error": f"No detections for {fid}"}
    total = len(fd["emotions"])
    dist = {e: round(c / total * 100, 1) for e, c in fd["dominant"].items() if c > 0}
    dom = max(fd["dominant"], key=fd["dominant"].get)
    recent = fd["engagement"][-20:]
    avg = round(sum(x["score"] for x in recent) / len(recent) * 100, 1) if recent else 0
    overall = round(sum(x["score"] for x in fd["engagement"]) / len(fd["engagement"]) * 100, 1)
    lbl = "High" if avg >= 65 else "Medium" if avg >= 40 else "Low"
    uq = len([e for e, c in fd["dominant"].items() if c > 0])
    return {"face_id": fid, "first_seen": fd["first"], "last_seen": fd["last"],
        "total_detections": total, "dominant_emotion": dom, "distribution": dist,
        "avg_engagement": avg, "overall_engagement": overall, "engagement_label": lbl,
        "emotion_variety": round(uq / 7 * 100, 1), "unique_emotions": uq,
        "timeline": fd["emotions"][-50:], "engagement_trend": fd["engagement"][-50:]}

def _mf_class():
    if not _mf["faces"]:
        return {"error": "No multi-face data yet"}
    profiles, total_det, all_eng = [], 0, []
    for fid in _mf["faces"]:
        p = _mf_profile(fid)
        if "error" not in p:
            profiles.append(p)
            total_det += p["total_detections"]
            all_eng.append(p["avg_engagement"])
    avg_cls = round(sum(all_eng) / len(all_eng), 1) if all_eng else 0
    lbl = "High" if avg_cls >= 65 else "Medium" if avg_cls >= 40 else "Low"
    ranked = sorted(profiles, key=lambda f: f["avg_engagement"], reverse=True)
    return {"total_faces": len(_mf["faces"]), "total_detections": total_det,
        "avg_class_engagement": avg_cls, "engagement_label": lbl,
        "face_profiles": profiles,
        "most_engaged": ranked[0]["face_id"] if ranked else None,
        "least_engaged": ranked[-1]["face_id"] if ranked else None}


# ── Multi-face API endpoints ──────────────────────────────────────────────
@app.post("/api/detect-emotion-multi")
async def detect_emotion_multi(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"success": False, "message": "Invalid image", "faces": []}
    _, results = detect_and_predict(frame)
    face_results = []
    for r in results:
        fid = _mf_assign_id(r["bbox"])
        eng = _mf_log(fid, r["emotion"], r["scores"])
        face_results.append({
            "face_id": fid, "bbox": r["bbox"], "emotion": r["emotion"],
            "emotions": {k: round(v * 100, 1) for k, v in r["scores"].items()},
            "confidence": round(r["confidence"] * 100, 1),
            "engagement": round(eng * 100, 1),
        })
    return {"success": len(results) > 0, "face_count": len(results), "faces": face_results}

@app.post("/session/multi/start")
async def multi_start():
    _mf_reset()
    return {"status": "Multi-face session started"}

@app.get("/session/multi/profile")
async def multi_profile():
    return _mf_class()

@app.get("/session/multi/face/{face_id}")
async def multi_face_profile(face_id: str):
    return _mf_profile(face_id)

@app.get("/session/multi/faces")
async def multi_list_faces():
    faces = []
    for fid in _mf["faces"]:
        p = _mf_profile(fid)
        if "error" not in p:
            faces.append({"face_id": fid, "total_detections": p["total_detections"],
                "dominant_emotion": p["dominant_emotion"], "avg_engagement": p["avg_engagement"],
                "engagement_label": p["engagement_label"]})
    return {"total_faces": len(faces), "faces": faces}

@app.get("/session/export/multi-csv")
async def export_multi_csv():
    if not _mf["faces"]:
        return {"error": "No multi-face data"}
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["face_id", "timestamp", "emotion", "confidence",
        "anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise", "engagement"])
    for fid, fd in _mf["faces"].items():
        emap = {x["time"]: x["score"] for x in fd["engagement"]}
        for entry in fd["emotions"]:
            sc = entry.get("scores", {})
            eng = emap.get(entry["time"], 0)
            writer.writerow([fid, entry["time"], entry["emotion"], entry.get("confidence", ""),
                sc.get("anger", ""), sc.get("disgust", ""), sc.get("fear", ""), sc.get("happiness", ""),
                sc.get("neutral", ""), sc.get("sadness", ""), sc.get("surprise", ""), round(eng * 100, 1)])
    output.seek(0)
    fname = f"multi_face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(io.BytesIO(output.getvalue().encode()), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fname}"})

@app.get("/session/export/multi-report")
async def export_multi_report():
    cd = _mf_class()
    if "error" in cd:
        return {"error": cd["error"]}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cards = ""
    for fp in cd["face_profiles"]:
        ec = "#22c55e" if fp["avg_engagement"] >= 65 else "#eab308" if fp["avg_engagement"] >= 40 else "#ef4444"
        bars = ""
        for em, pct in sorted(fp["distribution"].items(), key=lambda x: -x[1]):
            bars += f'<div style="margin:3px 0;display:flex;align-items:center;gap:6px"><span style="width:70px;font-size:12px">{em.capitalize()}</span><div style="background:#e0d9f9;height:14px;flex:1;border-radius:7px;overflow:hidden"><div style="background:#7c3aed;height:100%;width:{pct}%;border-radius:7px"></div></div><span style="font-size:11px;width:35px;text-align:right">{pct}%</span></div>'
        cards += f'<div style="background:#f5f3ff;border-radius:14px;padding:18px;border:1px solid #e0d9f9;margin-bottom:14px"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px"><h3 style="margin:0;color:#7c3aed">{fp["face_id"].replace("_"," ").title()}</h3><span style="background:{ec};color:white;padding:3px 12px;border-radius:16px;font-weight:700;font-size:13px">{fp["avg_engagement"]}% &mdash; {fp["engagement_label"]}</span></div><div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:10px"><div style="text-align:center"><div style="font-size:18px;font-weight:700;color:#7c3aed">{fp["total_detections"]}</div><div style="font-size:10px;color:#666">Detections</div></div><div style="text-align:center"><div style="font-size:18px;font-weight:700;color:#7c3aed;text-transform:capitalize">{fp["dominant_emotion"]}</div><div style="font-size:10px;color:#666">Dominant</div></div><div style="text-align:center"><div style="font-size:18px;font-weight:700;color:#7c3aed">{fp["unique_emotions"]}/7</div><div style="font-size:10px;color:#666">Variety</div></div><div style="text-align:center"><div style="font-size:18px;font-weight:700;color:{ec}">{fp["overall_engagement"]}%</div><div style="font-size:10px;color:#666">Overall</div></div></div>{bars}</div>'
    ac = "#22c55e" if cd["avg_class_engagement"] >= 65 else "#eab308" if cd["avg_class_engagement"] >= 40 else "#ef4444"
    html = f'''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Multi-Face Report</title>
<style>body{{font-family:-apple-system,sans-serif;max-width:850px;margin:40px auto;padding:0 24px;color:#1a1a1a}}h1{{color:#7c3aed;border-bottom:2px solid #7c3aed;padding-bottom:8px}}h2{{color:#444;margin-top:28px}}.sg{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:18px 0}}.sb{{background:#f5f3ff;border-radius:12px;padding:16px;text-align:center;border:1px solid #e0d9f9}}.sv{{font-size:28px;font-weight:700;color:#7c3aed}}.sl{{font-size:11px;color:#666;margin-top:3px}}.ft{{margin-top:36px;padding-top:14px;border-top:1px solid #eee;font-size:11px;color:#999}}</style>
</head><body><h1>EmotiLearn &mdash; Multi-Face Class Report</h1>
<p style="color:#666">Generated: {now}</p>
<h2>Class Overview</h2>
<div class="sg"><div class="sb"><div class="sv">{cd["total_faces"]}</div><div class="sl">Faces Detected</div></div><div class="sb"><div class="sv" style="color:{ac}">{cd["avg_class_engagement"]}%</div><div class="sl">Avg Engagement ({cd["engagement_label"]})</div></div><div class="sb"><div class="sv">{cd["total_detections"]}</div><div class="sl">Total Detections</div></div></div>
<p>Most engaged: <strong style="color:#22c55e">{(cd.get("most_engaged") or "N/A").replace("_"," ").title()}</strong> | Least engaged: <strong style="color:#ef4444">{(cd.get("least_engaged") or "N/A").replace("_"," ").title()}</strong></p>
<h2>Per-Face Profiles</h2>{cards}
<div class="ft">Generated by EmotiLearn Multi-Face | EfficientNet-B2 + Haar Cascade</div>
</body></html>'''
    fname = f"multi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    return StreamingResponse(io.BytesIO(html.encode()), media_type="text/html",
        headers={"Content-Disposition": f"attachment; filename={fname}"})
