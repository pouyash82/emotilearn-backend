import os, io, cv2, time, base64, asyncio, csv
import numpy as np
import torch
import torch.nn as nn
import timm
import whisper
import librosa
import soundfile as sf
from pathlib import Path
from collections import deque
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from transformers import pipeline
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pydantic import BaseModel
from typing import Optional

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

# ── STT model (lazy load) ──────────────────────────────────────────────────
stt_model = None

def get_stt_model():
    global stt_model
    if stt_model is None:
        try:
            print("Loading Whisper...")
            stt_model = whisper.load_model("base")
            print("✅ Whisper loaded")
        except Exception as e:
            print(f"⚠️  Whisper failed: {e}")
    return stt_model

# ── NLP model (lazy load) ──────────────────────────────────────────────────
nlp_pipeline = None

def get_nlp_pipeline():
    global nlp_pipeline
    if nlp_pipeline is None:
        try:
            print("Loading NLP model...")
            nlp_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None, device=-1)
            print("✅ NLP model loaded")
        except Exception as e:
            print(f"⚠️  NLP model failed: {e}")
    return nlp_pipeline

TEXT_TO_VISION = {
    "anger"   : "anger",   "disgust" : "disgust",
    "fear"    : "fear",    "joy"     : "happiness",
    "neutral" : "neutral", "sadness" : "sadness",
    "surprise": "surprise",
}

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
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        emotion = CLASS_NAMES[np.argmax(probs)]
        scores  = {CLASS_NAMES[i]: round(float(probs[i]), 4)
                   for i in range(len(CLASS_NAMES))}
        return emotion, scores
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

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

def audio_to_emotion(audio_path):
    result_data = {
        "transcription": "",
        "text_emotion" : "neutral",
        "audio_emotion": "neutral",
        "final_emotion": "neutral",
        "confidence"   : 0.5,
        "fused_scores" : {e: round(1/7, 4) for e in CLASS_NAMES},
        "text_scores"  : {},
        "audio_scores" : {},
    }
    try:
        stt = get_stt_model()
        if stt:
            result = stt.transcribe(audio_path, language="en")
            text   = result["text"].strip()
        else:
            text = ""
        result_data["transcription"] = text

        text_scores  = {}
        text_emotion = "neutral"
        nlp = get_nlp_pipeline()
        if nlp and text and len(text) > 3:
            nlp_out = nlp(text)[0]
            for item in nlp_out:
                mapped = TEXT_TO_VISION.get(
                    item["label"].lower(), item["label"].lower())
                text_scores[mapped] = round(item["score"], 4)
            if text_scores:
                text_emotion = max(text_scores, key=text_scores.get)
        result_data["text_emotion"] = text_emotion
        result_data["text_scores"]  = text_scores

        y, sr    = librosa.load(audio_path, sr=16000, mono=True)
        rms      = float(np.mean(librosa.feature.rms(y=y)))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo    = float(np.atleast_1d(tempo)[0])
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"))
        f0_clean = f0[~np.isnan(f0)] if f0 is not None \
                   else np.array([200.])
        pitch    = float(np.mean(f0_clean)) \
                   if len(f0_clean) > 0 else 200.

        audio_scores = {e: 0.0 for e in CLASS_NAMES}
        if rms > 0.05 and tempo > 120 and pitch < 220:
            audio_scores["anger"]     += 0.4
        if rms > 0.04 and tempo > 100 and pitch > 200:
            audio_scores["happiness"] += 0.4
        if rms < 0.03 and tempo < 90  and pitch < 180:
            audio_scores["sadness"]   += 0.5
        if pitch > 50  and rms > 0.02 and tempo > 100:
            audio_scores["fear"]      += 0.3
        if pitch > 60:
            audio_scores["surprise"]  += 0.3
        if pitch < 160 and rms < 0.04:
            audio_scores["disgust"]   += 0.3
        total = sum(audio_scores.values())
        if total < 0.1:
            audio_scores["neutral"] = 1.0
            total = 1.0
        audio_scores  = {k: round(v / total, 4)
                         for k, v in audio_scores.items()}
        audio_emotion = max(audio_scores, key=audio_scores.get)
        result_data["audio_emotion"] = audio_emotion
        result_data["audio_scores"]  = audio_scores

        all_e = set(text_scores) | set(audio_scores)
        fused = {}
        for e in all_e:
            fused[e] = round(
                0.4 * text_scores.get(e, 0) +
                0.6 * audio_scores.get(e, 0), 4)
        total = sum(fused.values())
        if total > 0:
            fused = {k: round(v / total, 4)
                     for k, v in fused.items()}
        if fused:
            final = max(fused, key=fused.get)
            result_data["final_emotion"] = final
            result_data["confidence"]    = fused[final]
            result_data["fused_scores"]  = fused

    except Exception as e:
        print(f"Audio analysis error: {e}")
    return result_data

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
# FIXED: Accepts FILE UPLOAD from React frontend (multipart/form-data)
# ══════════════════════════════════════════════════════════════════════════
@app.post("/api/detect-emotion")
async def detect_emotion_api(file: UploadFile = File(...)):
    """
    Accepts image file upload from React frontend,
    returns REAL emotion predictions from the EfficientNet model.
    """
    if vision_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Read file bytes
    contents = await file.read()
    
    # Decode image from bytes
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    # Detect faces and predict emotions using REAL model
    _, results = detect_and_predict(frame)
    
    if not results:
        raise HTTPException(status_code=400, detail="No face detected in image")
    
    # Get the first/largest face result
    face_result = results[0]
    emotion = face_result["emotion"]
    scores = face_result["scores"]
    
    # Log emotion for session tracking
    log_emotion(emotion, scores, source="vision")
    
    # Calculate current engagement
    current_eng = 0.0
    if session_data["engagement"]:
        current_eng = session_data["engagement"][-1]["score"]
    
    # Convert scores to percentages for frontend
    emotions_pct = {k: round(v * 100, 1) for k, v in scores.items()}
    
    return {
        "success": True,
        "dominant": emotion,
        "confidence": round(scores.get(emotion, 0) * 100, 1),
        "emotions": emotions_pct,
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

@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    tmp_path = f"/tmp/audio_{int(time.time())}.wav"
    try:
        data = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(data)
        result = audio_to_emotion(tmp_path)
        if result.get("final_emotion"):
            log_emotion(result["final_emotion"],
                       result.get("fused_scores", {}),
                       source="audio")
        return result
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/predict/combined")
async def predict_combined(
    image: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    img_data = await image.read()
    aud_data = await audio.read()
    arr   = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    vision_results = []
    if frame is not None:
        _, vision_results = detect_and_predict(frame)
    tmp_path = f"/tmp/audio_{int(time.time())}.wav"
    audio_result = {}
    try:
        with open(tmp_path, "wb") as f:
            f.write(aud_data)
        audio_result = audio_to_emotion(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    vision_top = vision_results[0] if vision_results else None
    final_emotion = "neutral"
    final_scores  = {}
    if vision_top and audio_result.get("fused_scores"):
        vs = vision_top["scores"]
        aus = audio_result["fused_scores"]
        all_e = set(vs) | set(aus)
        for e in all_e:
            final_scores[e] = round(
                0.6 * vs.get(e, 0) +
                0.4 * aus.get(e, 0), 4)
        if final_scores:
            final_emotion = max(final_scores, key=final_scores.get)
    elif vision_top:
        final_emotion = vision_top["emotion"]
        final_scores  = vision_top["scores"]
    elif audio_result.get("final_emotion"):
        final_emotion = audio_result["final_emotion"]
        final_scores  = audio_result.get("fused_scores", {})
    if final_scores:
        log_emotion(final_emotion, final_scores, source="combined")
    return {
        "vision"       : vision_results,
        "audio"        : audio_result,
        "final_emotion": final_emotion,
        "final_scores" : final_scores,
    }

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

