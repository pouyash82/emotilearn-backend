import os, io, cv2, time, base64, asyncio, csv, math
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
from typing import Optional, List

# ── Config ─────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
MODEL_PATH = Path(__file__).parent / "models" / \
             "efficientnet_b2_finetuned_best.pth"
CLASS_NAMES = ["anger","disgust","fear",
               "happiness","neutral","sadness","surprise"]

print(f"✅ Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════
# MULTI-FACE SESSION STORAGE
# ══════════════════════════════════════════════════════════════════════════
# Each face gets its own session tracking keyed by face_id
multi_session = {
    "start_time": None,
    "faces": {},           # face_id -> per-face session data
    "next_face_id": 1,     # auto-increment face ID
    "known_positions": [],  # [{id, last_bbox, last_seen}] for tracking
}

def reset_multi_session():
    multi_session["start_time"] = datetime.now().isoformat()
    multi_session["faces"] = {}
    multi_session["next_face_id"] = 1
    multi_session["known_positions"] = []

def get_or_assign_face_id(bbox):
    """
    Simple position-based face tracking.
    If a face is near a previously known position, reuse its ID.
    Otherwise assign a new ID.
    """
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    now = time.time()

    # Try to match with a known face (within 120px distance)
    best_match = None
    best_dist = 120  # max distance threshold

    for known in multi_session["known_positions"]:
        # Skip faces not seen in last 3 seconds (they left)
        if now - known["last_seen"] > 3.0:
            continue
        kx, ky = known["last_center"]
        dist = math.sqrt((cx - kx) ** 2 + (cy - ky) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_match = known

    if best_match:
        best_match["last_bbox"] = bbox
        best_match["last_center"] = (cx, cy)
        best_match["last_seen"] = now
        return best_match["id"]
    else:
        # New face
        face_id = f"face_{multi_session['next_face_id']}"
        multi_session["next_face_id"] += 1
        multi_session["known_positions"].append({
            "id": face_id,
            "last_bbox": bbox,
            "last_center": (cx, cy),
            "last_seen": now,
        })
        return face_id

def get_face_session(face_id):
    """Get or create per-face session data"""
    if face_id not in multi_session["faces"]:
        multi_session["faces"][face_id] = {
            "emotions": [],
            "dominant": {e: 0 for e in CLASS_NAMES},
            "engagement": [],
            "prev_emotion": None,
            "streak": 0,
            "first_seen": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
        }
    return multi_session["faces"][face_id]

def compute_engagement_for_face(face_id, emotion, scores, confidence):
    """Compute engagement per face (independent tracking)"""
    face_data = get_face_session(face_id)

    base_map = {
        "happiness": 0.85, "surprise": 0.80,
        "anger"    : 0.65, "fear"    : 0.55,
        "disgust"  : 0.50, "sadness" : 0.35,
        "neutral"  : 0.30,
    }
    base       = base_map.get(emotion, 0.5)
    conf_boost = max(-0.1, min(0.15, (confidence - 0.5) * 0.3))

    variety_boost = 0.0
    prev = face_data["prev_emotion"]
    if prev is not None and prev != emotion:
        variety_boost = 0.10
        face_data["streak"] = 0
    else:
        face_data["streak"] += 1

    streak_penalty = min(0.20,
        max(0, (face_data["streak"] - 10) * 0.01))
    neutral_prob  = scores.get("neutral", 0.5)
    express_boost = max(-0.1, min(0.15, (0.5 - neutral_prob) * 0.2))

    score = base + conf_boost + variety_boost \
            - streak_penalty + express_boost
    score = max(0.05, min(1.0, score))
    face_data["prev_emotion"] = emotion
    return round(score, 4)

def log_emotion_for_face(face_id, emotion, scores, source="vision"):
    """Log emotion for a specific face"""
    if multi_session["start_time"] is None:
        reset_multi_session()

    face_data = get_face_session(face_id)
    confidence = scores.get(emotion, 0.5)

    entry = {
        "time"      : datetime.now().isoformat(),
        "emotion"   : emotion,
        "scores"    : scores,
        "source"    : source,
        "confidence": round(confidence, 4),
        "face_id"   : face_id,
    }
    face_data["emotions"].append(entry)
    face_data["dominant"][emotion] = \
        face_data["dominant"].get(emotion, 0) + 1
    face_data["last_seen"] = entry["time"]

    eng_score = compute_engagement_for_face(
        face_id, emotion, scores, confidence)
    face_data["engagement"].append({
        "time"   : entry["time"],
        "score"  : eng_score,
        "emotion": emotion,
    })
    return eng_score

def get_face_profile(face_id):
    """Get learning profile for a specific face"""
    if face_id not in multi_session["faces"]:
        return {"error": f"No data for {face_id}"}

    face_data = multi_session["faces"][face_id]
    if not face_data["emotions"]:
        return {"error": f"No detections for {face_id}"}

    total = len(face_data["emotions"])
    distribution = {
        e: round(c / total * 100, 1)
        for e, c in face_data["dominant"].items() if c > 0
    }
    dominant = max(face_data["dominant"],
                   key=face_data["dominant"].get)
    recent_eng  = face_data["engagement"][-20:]
    avg_eng     = round(
        sum(e["score"] for e in recent_eng) /
        len(recent_eng) * 100, 1) if recent_eng else 0
    overall_eng = round(
        sum(e["score"] for e in face_data["engagement"]) /
        len(face_data["engagement"]) * 100, 1)
    eng_label = ("High"   if avg_eng >= 65
                 else "Medium" if avg_eng >= 40 else "Low")
    unique_emotions = len([e for e, c
                           in face_data["dominant"].items()
                           if c > 0])
    return {
        "face_id"           : face_id,
        "first_seen"        : face_data["first_seen"],
        "last_seen"         : face_data["last_seen"],
        "total_detections"  : total,
        "dominant_emotion"  : dominant,
        "distribution"      : distribution,
        "avg_engagement"    : avg_eng,
        "overall_engagement": overall_eng,
        "engagement_label"  : eng_label,
        "emotion_variety"   : round(unique_emotions / 7 * 100, 1),
        "unique_emotions"   : unique_emotions,
        "timeline"          : face_data["emotions"][-50:],
        "engagement_trend"  : face_data["engagement"][-50:],
    }

def get_class_profile():
    """Get aggregate profile across ALL faces"""
    all_faces = multi_session["faces"]
    if not all_faces:
        return {"error": "No session data yet"}

    face_profiles = []
    total_detections = 0
    all_engagement = []

    for face_id in all_faces:
        profile = get_face_profile(face_id)
        if "error" not in profile:
            face_profiles.append(profile)
            total_detections += profile["total_detections"]
            all_engagement.append(profile["avg_engagement"])

    avg_class_engagement = round(
        sum(all_engagement) / len(all_engagement), 1
    ) if all_engagement else 0

    eng_label = ("High"   if avg_class_engagement >= 65
                 else "Medium" if avg_class_engagement >= 40
                 else "Low")

    # Find most/least engaged faces
    sorted_faces = sorted(face_profiles,
                          key=lambda f: f["avg_engagement"],
                          reverse=True)

    return {
        "session_start"        : multi_session["start_time"],
        "total_faces_detected" : len(all_faces),
        "total_detections"     : total_detections,
        "avg_class_engagement" : avg_class_engagement,
        "engagement_label"     : eng_label,
        "face_profiles"        : face_profiles,
        "most_engaged"         : sorted_faces[0]["face_id"] if sorted_faces else None,
        "least_engaged"        : sorted_faces[-1]["face_id"] if sorted_faces else None,
    }


# ══════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBLE: Keep single-face session for existing frontend
# ══════════════════════════════════════════════════════════════════════════
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
    # Also reset multi-session
    reset_multi_session()

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
app = FastAPI(title="EmotiLearn API — Multi-Face Emotion Recognition")

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
    app.mount("/static", StaticFiles(directory=str(static_path)),
              name="static")

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
    """Detect ALL faces, predict emotions, assign face IDs"""
    try:
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(48, 48))
    except Exception:
        return frame_bgr, []

    results = []
    # Color palette for different faces
    face_colors = [
        (0, 255, 0),    # green
        (255, 165, 0),  # orange
        (0, 191, 255),  # deep sky blue
        (255, 0, 255),  # magenta
        (255, 255, 0),  # yellow
        (0, 255, 255),  # cyan
        (255, 105, 180),# hot pink
        (50, 205, 50),  # lime green
    ]

    for idx, (x, y, w, h) in enumerate(faces):
        try:
            face = frame_bgr[y:y+h, x:x+w]
            if face.size == 0:
                continue
            emotion, scores = predict_face_emotion(face)

            # Assign persistent face ID
            face_id = get_or_assign_face_id(
                [int(x), int(y), int(w), int(h)])

            results.append({
                "face_id"   : face_id,
                "bbox"      : [int(x), int(y), int(w), int(h)],
                "emotion"   : emotion,
                "scores"    : scores,
                "confidence": round(scores.get(emotion, 0), 4),
            })

            # Draw with unique color per face
            color = face_colors[idx % len(face_colors)]
            cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), color, 2)

            # Label with face ID + emotion
            face_num = face_id.replace("face_", "#")
            label = f"{face_num} {emotion.upper()} " \
                    f"{scores.get(emotion, 0)*100:.0f}%"
            # Background for text
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_bgr, (x, y - th - 10),
                          (x + tw + 4, y), color, -1)
            cv2.putText(frame_bgr, label, (x + 2, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 2)
        except Exception as e:
            print(f"Face processing error: {e}")
            continue
    return frame_bgr, results


# ══════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════
@app.get("/")
async def root():
    return {
        "status"      : "EmotiLearn API running — Multi-Face v2.0",
        "device"      : str(DEVICE),
        "model_loaded": vision_model is not None,
        "features"    : [
            "multi-face detection",
            "per-face engagement tracking",
            "per-face learning profiles",
            "class-wide analytics",
        ],
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
# SINGLE-FACE API (backward compatible with existing frontend)
# ══════════════════════════════════════════════════════════════════════════
@app.post("/api/detect-emotion")
async def detect_emotion_api(file: UploadFile = File(...)):
    """
    Backward compatible: returns first face result for existing frontend.
    Also logs ALL faces to multi-session.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {
            "success": False,
            "message": "Invalid image",
            "emotions": {e: 0.0 for e in CLASS_NAMES},
            "dominant": "neutral"
        }

    _, results = detect_and_predict(frame)

    # Log ALL faces to multi-session
    for r in results:
        log_emotion_for_face(
            r["face_id"], r["emotion"], r["scores"], "vision")

    if not results:
        return {
            "success": False,
            "message": "No face detected",
            "emotions": {e: 0.0 for e in CLASS_NAMES},
            "dominant": "neutral"
        }

    # Return first face for backward compatibility
    face_result = results[0]
    emotion = face_result["emotion"]
    scores  = face_result["scores"]

    # Also log to legacy single-face session
    log_emotion(emotion, scores, source="vision")

    current_eng = 0.0
    if session_data["engagement"]:
        current_eng = session_data["engagement"][-1]["score"]

    emotions_pct = {k: round(v * 100, 1) for k, v in scores.items()}

    return {
        "success"   : True,
        "emotions"  : emotions_pct,
        "dominant"  : emotion,
        "confidence": round(scores.get(emotion, 0) * 100, 1),
        "engagement": round(current_eng * 100, 1),
        "bbox"      : face_result.get("bbox"),
        "face_id"   : face_result.get("face_id"),
        "face_count": len(results),
    }


# ══════════════════════════════════════════════════════════════════════════
# MULTI-FACE API (new endpoints)
# ══════════════════════════════════════════════════════════════════════════
@app.post("/api/detect-emotion-multi")
async def detect_emotion_multi(file: UploadFile = File(...)):
    """
    Returns ALL detected faces with per-face emotion + engagement.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"success": False, "message": "Invalid image", "faces": []}

    _, results = detect_and_predict(frame)

    face_results = []
    for r in results:
        face_id = r["face_id"]
        eng = log_emotion_for_face(
            face_id, r["emotion"], r["scores"], "vision")

        face_data = get_face_session(face_id)
        total = len(face_data["emotions"])

        face_results.append({
            "face_id"           : face_id,
            "bbox"              : r["bbox"],
            "emotion"           : r["emotion"],
            "emotions"          : {k: round(v * 100, 1)
                                   for k, v in r["scores"].items()},
            "confidence"        : round(r["confidence"] * 100, 1),
            "engagement"        : round(eng * 100, 1),
            "total_detections"  : total,
            "dominant_emotion"  : max(face_data["dominant"],
                                     key=face_data["dominant"].get)
                                 if total > 0 else "neutral",
        })

    # Also log first face to legacy session
    if results:
        log_emotion(results[0]["emotion"],
                    results[0]["scores"], "vision")

    return {
        "success"   : len(results) > 0,
        "face_count": len(results),
        "faces"     : face_results,
        "message"   : f"{len(results)} face(s) detected"
                      if results else "No faces detected",
    }


# ── Multi-face session endpoints ──────────────────────────────────────────
@app.post("/session/start")
async def session_start():
    reset_session()
    return {"status": "Session started",
            "time": session_data["start_time"]}

@app.post("/session/reset")
async def session_reset():
    reset_session()
    return {"status": "Session reset",
            "time": session_data["start_time"]}

@app.post("/session/end")
async def session_end():
    """End session — returns both single and multi-face profiles"""
    profile = get_learning_profile()
    class_profile = get_class_profile()
    return {
        "status"       : "Session ended",
        "profile"      : profile,
        "class_profile": class_profile,
    }

@app.get("/session/profile")
async def session_profile():
    return get_learning_profile()

@app.get("/session/timeline")
async def session_timeline():
    return {
        "emotions"  : session_data["emotions"][-100:],
        "engagement": session_data["engagement"][-100:],
    }


# ── Multi-face profile endpoints ──────────────────────────────────────────
@app.get("/session/multi/profile")
async def multi_profile():
    """Get class-wide profile with per-face breakdown"""
    return get_class_profile()

@app.get("/session/multi/face/{face_id}")
async def face_profile(face_id: str):
    """Get profile for a specific face"""
    return get_face_profile(face_id)

@app.get("/session/multi/faces")
async def list_faces():
    """List all detected faces with summary stats"""
    faces = []
    for face_id in multi_session["faces"]:
        profile = get_face_profile(face_id)
        if "error" not in profile:
            faces.append({
                "face_id"          : face_id,
                "total_detections" : profile["total_detections"],
                "dominant_emotion" : profile["dominant_emotion"],
                "avg_engagement"   : profile["avg_engagement"],
                "engagement_label" : profile["engagement_label"],
                "first_seen"       : profile["first_seen"],
                "last_seen"        : profile["last_seen"],
            })
    return {
        "total_faces": len(faces),
        "faces"      : faces,
    }

@app.get("/session/multi/engagement-summary")
async def multi_engagement_summary():
    """Quick engagement summary across all faces"""
    faces = multi_session["faces"]
    if not faces:
        return {"error": "No faces detected yet"}

    summary = []
    for face_id, data in faces.items():
        if data["engagement"]:
            avg = round(
                sum(e["score"] for e in data["engagement"]) /
                len(data["engagement"]) * 100, 1)
            latest = round(data["engagement"][-1]["score"] * 100, 1)
        else:
            avg = 0
            latest = 0
        summary.append({
            "face_id"           : face_id,
            "avg_engagement"    : avg,
            "current_engagement": latest,
            "total_detections"  : len(data["emotions"]),
            "dominant_emotion"  : max(data["dominant"],
                                     key=data["dominant"].get)
                                 if data["emotions"] else "neutral",
        })

    return {
        "face_count": len(summary),
        "faces"     : sorted(summary,
                             key=lambda f: f["avg_engagement"],
                             reverse=True),
    }


# ── Predict endpoints ─────────────────────────────────────────────────────
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
            log_emotion_for_face(
                r["face_id"], r["emotion"], r["scores"], "vision")
        return {"faces": results, "count": len(results)}
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════
# MULTI-FACE REPORT EXPORT
# ══════════════════════════════════════════════════════════════════════════
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


@app.get("/session/export/multi-csv")
async def export_multi_csv():
    """Export CSV with per-face data"""
    if not multi_session["faces"]:
        return {"error": "No multi-face data to export"}
    try:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "face_id", "timestamp", "emotion", "confidence",
            "anger", "disgust", "fear", "happiness",
            "neutral", "sadness", "surprise", "engagement_score"
        ])
        for face_id, face_data in multi_session["faces"].items():
            eng_map = {e["time"]: e["score"]
                       for e in face_data["engagement"]}
            for entry in face_data["emotions"]:
                scores = entry.get("scores", {})
                eng    = eng_map.get(entry["time"], 0)
                writer.writerow([
                    face_id,
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
        fname = f"multi_face_session_" \
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
    """Single-face report (backward compatible)"""
    if not session_data["emotions"]:
        return {"error": "No session data to export"}
    try:
        profile = get_learning_profile()
        now     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dist_rows = ""
        for e, pct in sorted(profile["distribution"].items(),
                              key=lambda x: -x[1]):
            bar = "\u2588" * int(pct / 5)
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
  <h1>&#x1F9E0; Emotion Recognition — Session Report</h1>
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


@app.get("/session/export/multi-report")
async def export_multi_report():
    """Multi-face class report with per-face breakdowns"""
    class_data = get_class_profile()
    if "error" in class_data:
        return {"error": class_data["error"]}

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build per-face cards
    face_cards = ""
    for fp in class_data["face_profiles"]:
        eng_color = ("#22c55e" if fp["avg_engagement"] >= 65
                     else "#eab308" if fp["avg_engagement"] >= 40
                     else "#ef4444")
        dist_bars = ""
        for e, pct in sorted(fp["distribution"].items(),
                              key=lambda x: -x[1]):
            bar_w = int(pct * 2)
            dist_bars += f"""
            <div style="margin:4px 0;display:flex;
                        align-items:center;gap:8px">
              <span style="width:80px;font-size:12px">
                {e.capitalize()}
              </span>
              <div style="background:#e0d9f9;height:16px;
                          flex:1;border-radius:8px;overflow:hidden">
                <div style="background:#7c3aed;height:100%;
                            width:{pct}%;border-radius:8px"></div>
              </div>
              <span style="font-size:12px;width:40px;
                           text-align:right">{pct}%</span>
            </div>"""

        face_cards += f"""
        <div style="background:#f5f3ff;border-radius:16px;
                    padding:20px;border:1px solid #e0d9f9;
                    margin-bottom:16px">
          <div style="display:flex;justify-content:space-between;
                      align-items:center;margin-bottom:12px">
            <h3 style="margin:0;color:#7c3aed">
              {fp["face_id"].replace("_", " ").title()}
            </h3>
            <span style="background:{eng_color};color:white;
                         padding:4px 14px;border-radius:20px;
                         font-weight:700;font-size:14px">
              {fp["avg_engagement"]}% — {fp["engagement_label"]}
            </span>
          </div>
          <div style="display:grid;
                      grid-template-columns:repeat(4,1fr);
                      gap:12px;margin-bottom:12px">
            <div style="text-align:center">
              <div style="font-size:20px;font-weight:700;
                          color:#7c3aed">
                {fp["total_detections"]}
              </div>
              <div style="font-size:11px;color:#666">
                Detections
              </div>
            </div>
            <div style="text-align:center">
              <div style="font-size:20px;font-weight:700;
                          color:#7c3aed;text-transform:capitalize">
                {fp["dominant_emotion"]}
              </div>
              <div style="font-size:11px;color:#666">
                Dominant
              </div>
            </div>
            <div style="text-align:center">
              <div style="font-size:20px;font-weight:700;
                          color:#7c3aed">
                {fp["unique_emotions"]}/7
              </div>
              <div style="font-size:11px;color:#666">
                Variety
              </div>
            </div>
            <div style="text-align:center">
              <div style="font-size:20px;font-weight:700;
                          color:{eng_color}">
                {fp["overall_engagement"]}%
              </div>
              <div style="font-size:11px;color:#666">
                Overall Eng.
              </div>
            </div>
          </div>
          {dist_bars}
        </div>"""

    avg_color = ("#22c55e"
                 if class_data["avg_class_engagement"] >= 65
                 else "#eab308"
                 if class_data["avg_class_engagement"] >= 40
                 else "#ef4444")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Multi-Face Session Report</title>
<style>
  body {{ font-family:-apple-system,sans-serif;
          max-width:900px; margin:40px auto;
          padding:0 24px; color:#1a1a1a; }}
  h1   {{ color:#7c3aed;
          border-bottom:2px solid #7c3aed;
          padding-bottom:8px; }}
  h2   {{ color:#444; margin-top:32px; }}
  .stat-grid {{ display:grid;
                grid-template-columns:repeat(3,1fr);
                gap:16px; margin:20px 0; }}
  .stat-box  {{ background:#f5f3ff; border-radius:12px;
                padding:20px; text-align:center;
                border:1px solid #e0d9f9; }}
  .stat-val  {{ font-size:32px; font-weight:700;
                color:#7c3aed; }}
  .stat-lbl  {{ font-size:12px; color:#666;
                margin-top:4px; }}
  .footer    {{ margin-top:40px; padding-top:16px;
                border-top:1px solid #eee;
                font-size:12px; color:#999; }}
</style>
</head>
<body>
  <h1>&#x1F9D1;&#x200D;&#x1F393; EmotiLearn — Multi-Face Class Report</h1>
  <p style="color:#666">Generated: {now}</p>
  <p style="color:#666">
    Session start: {class_data.get("session_start","N/A")}
  </p>

  <h2>Class Overview</h2>
  <div class="stat-grid">
    <div class="stat-box">
      <div class="stat-val">
        {class_data["total_faces_detected"]}
      </div>
      <div class="stat-lbl">Faces Detected</div>
    </div>
    <div class="stat-box">
      <div class="stat-val" style="color:{avg_color}">
        {class_data["avg_class_engagement"]}%
      </div>
      <div class="stat-lbl">
        Avg Class Engagement ({class_data["engagement_label"]})
      </div>
    </div>
    <div class="stat-box">
      <div class="stat-val">
        {class_data["total_detections"]}
      </div>
      <div class="stat-lbl">Total Detections</div>
    </div>
  </div>

  <p>Most engaged:
    <strong style="color:#22c55e">
      {(class_data.get("most_engaged") or "N/A")
       .replace("_"," ").title()}
    </strong>
    &nbsp;|&nbsp; Least engaged:
    <strong style="color:#ef4444">
      {(class_data.get("least_engaged") or "N/A")
       .replace("_"," ").title()}
    </strong>
  </p>

  <h2>Per-Face Profiles</h2>
  {face_cards}

  <div class="footer">
    Generated by EmotiLearn Multi-Face System |
    EfficientNet-B2 + Haar Cascade |
    Position-based face tracking
  </div>
</body>
</html>"""
    fname = f"multi_face_report_" \
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    return StreamingResponse(
        io.BytesIO(html.encode()),
        media_type="text/html",
        headers={"Content-Disposition":
                 f"attachment; filename={fname}"})


# ── WebSocket Webcam (Multi-Face) ─────────────────────────────────────────
@app.websocket("/ws/webcam")
async def webcam_ws(websocket: WebSocket):
    await websocket.accept()
    print("✅ Webcam WebSocket connected (multi-face)")
    cap = None
    try:
        loop = asyncio.get_event_loop()
        cap  = cv2.VideoCapture(0)
        if not cap.isOpened():
            await websocket.send_json(
                {"error": "Cannot open webcam"})
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

                # Log ALL faces
                face_engagements = {}
                for r in results:
                    eng = log_emotion_for_face(
                        r["face_id"], r["emotion"],
                        r["scores"], "vision")
                    face_engagements[r["face_id"]] = eng
                    # Also log first face to legacy session
                if results:
                    log_emotion(results[0]["emotion"],
                                results[0]["scores"], "vision")

                _, buffer = cv2.imencode(
                    ".jpg", annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64 = base64.b64encode(buffer).decode("utf-8")

                current_eng = 0.0
                if session_data["engagement"]:
                    current_eng = \
                        session_data["engagement"][-1]["score"]

                await websocket.send_json({
                    "frame"           : b64,
                    "faces"           : results,
                    "count"           : len(results),
                    "engagement"      : round(current_eng * 100, 1),
                    "face_engagements": {
                        fid: round(eng * 100, 1)
                        for fid, eng in face_engagements.items()
                    },
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

