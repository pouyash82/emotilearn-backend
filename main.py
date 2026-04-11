"""
EmotiLearn Backend API
======================
Emotion Recognition with Research-Based Engagement Detection

Engagement model based on:
- Whitehill et al. (2014) - "The Faces of Engagement"
- D'Mello & Graesser (2012) - "Dynamics of Affective States during Complex Learning"
- Bosch et al. (2016) - "Detecting Student Emotions in Computer-Enabled Classrooms"
"""

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
from typing import Optional

# Import research-based engagement module
from engagement_research import (
    engagement_detector,
    compute_engagement_detailed,
    reset_engagement_session,
    get_engagement_summary
)

# ── Config ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
MODEL_PATH = Path(__file__).parent / "models" / \
             "efficientnet_b2_finetuned_best.pth"
CLASS_NAMES = ["anger", "disgust", "fear",
               "happiness", "neutral", "sadness", "surprise"]

print(f"✅ Device: {DEVICE}")

# ── Session storage ────────────────────────────────────────────────────────
session_data = {
    "start_time": None,
    "emotions": [],
    "dominant": {},
    "engagement": [],
    "prev_emotion": None,
    "streak": 0,
}

def reset_session():
    session_data["start_time"] = datetime.now().isoformat()
    session_data["emotions"] = []
    session_data["dominant"] = {e: 0 for e in CLASS_NAMES}
    session_data["engagement"] = []
    session_data["prev_emotion"] = None
    session_data["streak"] = 0
    # Also reset the research-based engagement detector
    reset_engagement_session()

def log_emotion(emotion, scores, source="vision"):
    """Log emotion and compute research-based engagement"""
    if session_data["start_time"] is None:
        reset_session()
    
    confidence = scores.get(emotion, 0.5)
    
    # Get RESEARCH-BASED engagement analysis
    engagement_result = compute_engagement_detailed(emotion, scores, face_detected=True)
    
    entry = {
        "time": datetime.now().isoformat(),
        "emotion": emotion,
        "scores": scores,
        "source": source,
        "confidence": round(confidence, 4),
        # New research-based fields
        "engagement_analysis": engagement_result,
    }
    
    session_data["emotions"].append(entry)
    session_data["dominant"][emotion] = \
        session_data["dominant"].get(emotion, 0) + 1
    
    session_data["engagement"].append({
        "time": entry["time"],
        "score": engagement_result["engagement_raw"],
        "level": engagement_result["engagement_level"],
        "emotion": emotion,
        "attention": engagement_result["attention_state"],
        "cognitive_load": engagement_result["cognitive_load"],
        "is_productive": engagement_result["is_productive"],
    })
    
    session_data["prev_emotion"] = emotion
    
    return engagement_result

def get_learning_profile():
    """Get session profile with research-based insights"""
    if not session_data["emotions"]:
        return {"error": "No session data yet"}
    
    total = len(session_data["emotions"])
    distribution = {
        e: round(c / total * 100, 1)
        for e, c in session_data["dominant"].items() if c > 0
    }
    dominant = max(session_data["dominant"],
                   key=session_data["dominant"].get)
    
    # Calculate engagement stats
    recent_eng = session_data["engagement"][-20:]
    all_eng = session_data["engagement"]
    
    avg_eng = round(
        sum(e["score"] for e in recent_eng) /
        len(recent_eng) * 100, 1) if recent_eng else 0
    overall_eng = round(
        sum(e["score"] for e in all_eng) /
        len(all_eng) * 100, 1) if all_eng else 0
    
    # Research-based engagement label
    if avg_eng >= 75:
        eng_label = "Highly Engaged"
    elif avg_eng >= 50:
        eng_label = "Engaged"
    elif avg_eng >= 25:
        eng_label = "Barely Engaged"
    else:
        eng_label = "Disengaged"
    
    # Calculate attention states distribution
    attention_counts = {}
    for e in all_eng:
        att = e.get("attention", "unknown")
        attention_counts[att] = attention_counts.get(att, 0) + 1
    attention_distribution = {
        k: round(v / total * 100, 1)
        for k, v in attention_counts.items()
    }
    
    # Calculate productive time
    productive_count = sum(1 for e in all_eng if e.get("is_productive", False))
    productive_pct = round(productive_count / total * 100, 1) if total > 0 else 0
    
    # Get detailed summary from research module
    research_summary = get_engagement_summary()
    
    unique_emotions = len([e for e, c in session_data["dominant"].items() if c > 0])
    
    return {
        "session_start": session_data["start_time"],
        "total_detections": total,
        "dominant_emotion": dominant,
        "distribution": distribution,
        "avg_engagement": avg_eng,
        "overall_engagement": overall_eng,
        "engagement_label": eng_label,
        "emotion_variety": round(unique_emotions / 7 * 100, 1),
        "unique_emotions": unique_emotions,
        
        # Research-based additions
        "attention_distribution": attention_distribution,
        "productive_time_pct": productive_pct,
        "productive_struggles": research_summary.get("productive_struggles", 0),
        "cognitive_engagement": research_summary.get("avg_engagement", 0),
        
        "timeline": session_data["emotions"][-50:],
        "engagement_trend": session_data["engagement"][-50:],
        
        # Academic reference
        "methodology": {
            "model": "Research-based engagement detection",
            "references": [
                "Whitehill et al. (2014) - IEEE Trans. Affective Computing",
                "D'Mello & Graesser (2012) - Learning and Instruction",
                "Bosch et al. (2016) - IJCAI"
            ]
        }
    }

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="EmotiLearn API - Research-Based Engagement")

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
        raise HTTPException(status_code=500, detail="Vision model not loaded")
    try:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        aug = val_transform(image=img)["image"]
        aug = aug.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = vision_model(aug)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        emotion = CLASS_NAMES[np.argmax(probs)]
        scores = {CLASS_NAMES[i]: round(float(probs[i]), 4)
                  for i in range(len(CLASS_NAMES))}
        return emotion, scores
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

def detect_and_predict(frame_bgr):
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
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
                "bbox": [int(x), int(y), int(w), int(h)],
                "emotion": emotion,
                "scores": scores,
                "confidence": round(scores.get(emotion, 0), 4),
            })
            color = (0, 255, 0)
            cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), color, 2)
            label = f"{emotion.upper()} {scores.get(emotion, 0)*100:.0f}%"
            cv2.putText(frame_bgr, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except Exception as e:
            print(f"Face processing error: {e}")
            continue
    return frame_bgr, results

# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status": "EmotiLearn API running",
        "device": str(DEVICE),
        "model_loaded": vision_model is not None,
        "engagement_model": "Research-based (Whitehill 2014, D'Mello 2012, Bosch 2016)",
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
        return HTMLResponse(content=f"<h1>Error: {e}</h1>", status_code=500)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN API ENDPOINT - Research-Based Emotion + Engagement Detection
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/detect-emotion")
async def detect_emotion_api(file: UploadFile = File(...)):
    """
    Detect emotion and compute RESEARCH-BASED engagement.
    
    Returns:
    - emotion detection results
    - engagement score with academic methodology
    - cognitive load estimate
    - attention state (focused/flow/bored/distracted/frustrated)
    - productive learning indicator
    """
    if vision_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Read and decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    # Detect faces and predict emotions
    _, results = detect_and_predict(frame)
    
    if not results:
        raise HTTPException(status_code=400, detail="No face detected")
    
    # Get first face result
    face_result = results[0]
    emotion = face_result["emotion"]
    scores = face_result["scores"]
    
    # Log emotion and get RESEARCH-BASED engagement analysis
    engagement_result = log_emotion(emotion, scores, source="vision")
    
    # Convert scores to percentages
    emotions_pct = {k: round(v * 100, 1) for k, v in scores.items()}
    
    return {
        "success": True,
        "dominant": emotion,
        "confidence": round(scores.get(emotion, 0) * 100, 1),
        "emotions": emotions_pct,
        "bbox": face_result.get("bbox"),
        
        # Research-based engagement data
        "engagement": engagement_result["engagement_score"],
        "engagement_level": engagement_result["engagement_level"],
        "cognitive_load": engagement_result["cognitive_load"],
        "attention_state": engagement_result["attention_state"],
        "is_productive": engagement_result["is_productive"],
        "emotion_variety": engagement_result["emotion_variety"],
        
        # Debug: show component breakdown
        "engagement_components": engagement_result["components"],
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
    """End session and return research-based summary"""
    profile = get_learning_profile()
    return {"status": "Session ended", "profile": profile}

@app.get("/session/profile")
async def session_profile():
    return get_learning_profile()

@app.get("/session/timeline")
async def session_timeline():
    return {
        "emotions": session_data["emotions"][-100:],
        "engagement": session_data["engagement"][-100:],
    }

# ── Predict endpoints ──────────────────────────────────────────────────────
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
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
            "neutral", "sadness", "surprise",
            "engagement_score", "engagement_level",
            "attention_state", "cognitive_load", "is_productive"
        ])
        eng_map = {e["time"]: e for e in session_data["engagement"]}
        for entry in session_data["emotions"]:
            scores = entry.get("scores", {})
            eng_entry = eng_map.get(entry["time"], {})
            writer.writerow([
                entry["time"],
                entry["emotion"],
                entry.get("confidence", ""),
                entry.get("source", "vision"),
                scores.get("anger", ""),
                scores.get("disgust", ""),
                scores.get("fear", ""),
                scores.get("happiness", ""),
                scores.get("neutral", ""),
                scores.get("sadness", ""),
                scores.get("surprise", ""),
                round(eng_entry.get("score", 0) * 100, 1),
                eng_entry.get("level", ""),
                eng_entry.get("attention", ""),
                eng_entry.get("cognitive_load", ""),
                eng_entry.get("is_productive", ""),
            ])
        output.seek(0)
        fname = f"emotion_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={fname}"})
    except Exception as e:
        return {"error": f"Export failed: {e}"}

@app.get("/session/export/report")
async def export_report():
    if not session_data["emotions"]:
        return {"error": "No session data to export"}
    try:
        profile = get_learning_profile()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        dist_rows = ""
        for e, pct in sorted(profile["distribution"].items(), key=lambda x: -x[1]):
            bar = "█" * int(pct / 5)
            dist_rows += f"""
            <tr>
              <td>{e.capitalize()}</td>
              <td>{pct}%</td>
              <td style="font-family:monospace;color:#7c3aed">{bar}</td>
            </tr>"""
        
        attention_rows = ""
        for att, pct in profile.get("attention_distribution", {}).items():
            attention_rows += f"""
            <tr>
              <td>{att.capitalize()}</td>
              <td>{pct}%</td>
            </tr>"""
        
        timeline_rows = ""
        for entry in session_data["engagement"][-20:]:
            t = entry["time"][11:19]
            eng = round(entry["score"] * 100, 1)
            em = entry["emotion"]
            att = entry.get("attention", "")
            timeline_rows += f"""
            <tr>
              <td>{t}</td>
              <td>{em.capitalize()}</td>
              <td>{eng}%</td>
              <td>{att}</td>
            </tr>"""
        
        eng_color = ("#22c55e" if profile["avg_engagement"] >= 65
                     else "#eab308" if profile["avg_engagement"] >= 40
                     else "#ef4444")
        
        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>EmotiLearn Session Report</title>
<style>
  body {{ font-family:-apple-system,sans-serif; max-width:900px; margin:40px auto; padding:0 24px; color:#1a1a1a; }}
  h1 {{ color:#7c3aed; border-bottom:2px solid #7c3aed; padding-bottom:8px; }}
  h2 {{ color:#444; margin-top:32px; }}
  .stat-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:20px 0; }}
  .stat-box {{ background:#f5f3ff; border-radius:12px; padding:16px; text-align:center; border:1px solid #e0d9f9; }}
  .stat-val {{ font-size:24px; font-weight:700; color:#7c3aed; }}
  .stat-lbl {{ font-size:11px; color:#666; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:12px; }}
  th {{ background:#7c3aed; color:white; padding:10px 14px; text-align:left; font-size:13px; }}
  td {{ padding:8px 14px; font-size:13px; border-bottom:1px solid #eee; }}
  tr:hover td {{ background:#faf5ff; }}
  .badge {{ display:inline-block; padding:4px 12px; border-radius:20px; font-weight:700; font-size:14px; background:#f5f3ff; color:#7c3aed; border:1px solid #7c3aed; }}
  .methodology {{ background:#f0f9ff; border:1px solid #0ea5e9; border-radius:8px; padding:16px; margin-top:24px; }}
  .methodology h3 {{ color:#0369a1; margin-top:0; }}
  .methodology p {{ font-size:12px; color:#475569; margin:4px 0; }}
  .footer {{ margin-top:40px; padding-top:16px; border-top:1px solid #eee; font-size:12px; color:#999; }}
</style>
</head>
<body>
  <h1>🧠 EmotiLearn — Session Report</h1>
  <p style="color:#666">Generated: {now}</p>
  <p style="color:#666">Session start: {profile.get("session_start","N/A")}</p>
  
  <h2>Summary</h2>
  <div class="stat-grid">
    <div class="stat-box">
      <div class="stat-val">{profile["total_detections"]}</div>
      <div class="stat-lbl">Total Detections</div>
    </div>
    <div class="stat-box">
      <div class="stat-val" style="color:{eng_color}">{profile["avg_engagement"]}%</div>
      <div class="stat-lbl">Avg Engagement</div>
    </div>
    <div class="stat-box">
      <div class="stat-val">{profile.get("productive_time_pct", 0)}%</div>
      <div class="stat-lbl">Productive Time</div>
    </div>
    <div class="stat-box">
      <div class="stat-val">{profile.get("productive_struggles", 0)}</div>
      <div class="stat-lbl">Productive Struggles</div>
    </div>
  </div>
  
  <p>Engagement Level: <span class="badge">{profile["engagement_label"]}</span></p>
  <p>Dominant Emotion: <span class="badge">{profile["dominant_emotion"].upper()}</span></p>
  
  <h2>Emotion Distribution</h2>
  <table>
    <tr><th>Emotion</th><th>Percentage</th><th>Visual</th></tr>
    {dist_rows}
  </table>
  
  <h2>Attention States</h2>
  <table>
    <tr><th>State</th><th>Percentage</th></tr>
    {attention_rows}
  </table>
  
  <h2>Recent Timeline (last 20)</h2>
  <table>
    <tr><th>Time</th><th>Emotion</th><th>Engagement</th><th>Attention</th></tr>
    {timeline_rows}
  </table>
  
  <div class="methodology">
    <h3>📚 Research Methodology</h3>
    <p>Engagement detection based on peer-reviewed research:</p>
    <p>• Whitehill, J., et al. (2014). "The Faces of Engagement" - IEEE Trans. Affective Computing</p>
    <p>• D'Mello, S., & Graesser, A. (2012). "Dynamics of Affective States during Complex Learning"</p>
    <p>• Bosch, N., et al. (2016). "Detecting Student Emotions in Computer-Enabled Classrooms" - IJCAI</p>
  </div>
  
  <div class="footer">
    Generated by EmotiLearn | EfficientNet-B2 Vision + Research-Based Engagement Model
  </div>
</body>
</html>"""
        fname = f"emotion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        return StreamingResponse(
            io.BytesIO(html.encode()),
            media_type="text/html",
            headers={"Content-Disposition": f"attachment; filename={fname}"})
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
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            await websocket.send_json({"error": "Cannot open webcam"})
            await websocket.close()
            return
        consecutive_errors = 0
        while True:
            try:
                ret, frame = await loop.run_in_executor(None, cap.read)
                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors > 10:
                        break
                    await asyncio.sleep(0.1)
                    continue
                consecutive_errors = 0
                annotated, results = detect_and_predict(frame)
                
                engagement_data = None
                for r in results:
                    engagement_data = log_emotion(r["emotion"], r["scores"], source="vision")
                
                _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64 = base64.b64encode(buffer).decode("utf-8")
                
                current_eng = engagement_data["engagement_score"] if engagement_data else 0
                
                await websocket.send_json({
                    "frame": b64,
                    "faces": results,
                    "count": len(results),
                    "engagement": current_eng,
                    "engagement_level": engagement_data["engagement_level"] if engagement_data else "",
                    "attention_state": engagement_data["attention_state"] if engagement_data else "",
                })
                await asyncio.sleep(0.05)
            except Exception as e:
                if "close message" in str(e) or "disconnect" in str(e).lower():
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
