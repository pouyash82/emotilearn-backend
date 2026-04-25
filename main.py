from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import settings
from app.services.vision_emotion import VisionEmotionService
from app.services.speech_to_text import SpeechToTextService
from app.services.text_emotion import TextEmotionService
from app.services.audio_emotion import AudioEmotionService
from app.api.schemas import *

# Initialize FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Multimodal Emotion Detection: Vision + Audio + Text"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
vision_service = None
stt_service = None
text_emotion_service = None
audio_emotion_service = None


@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup"""
    global vision_service, stt_service, text_emotion_service, audio_emotion_service
    
    logger.info("="*70)
    logger.info("STARTING EMOLEARN BACKEND")
    logger.info("="*70)
    
    try:
        # 1. Load YOUR EfficientNet-B2 vision model
        if settings.VISION_MODEL_PATH.exists():
            vision_service = VisionEmotionService(settings.VISION_MODEL_PATH)
            logger.info("✓ Vision service loaded (EfficientNet-B2, 87% accuracy)")
        else:
            logger.warning(f"⚠️  Vision model not found at {settings.VISION_MODEL_PATH}")
        
        # 2. Load Speech-to-Text
        stt_service = SpeechToTextService(model_size=settings.WHISPER_MODEL)
        logger.info("✓ Speech-to-Text service loaded (Whisper)")
        
        # 3. Load Text Emotion
        text_emotion_service = TextEmotionService()
        logger.info("✓ Text emotion service loaded (DistilRoBERTa)")
        
        # 4. Load Audio Emotion
        audio_emotion_service = AudioEmotionService()
        logger.info("✓ Audio emotion service loaded")
        
        logger.info("="*70)
        logger.info("✅ ALL SERVICES READY!")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "version": settings.VERSION,
        "models_loaded": {
            "vision": vision_service is not None,
            "speech_to_text": stt_service is not None,
            "text_emotion": text_emotion_service is not None,
            "audio_emotion": audio_emotion_service is not None
        }
    }


@app.post("/api/vision/analyze", response_model=VisionResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze image for facial emotions using YOUR EfficientNet-B2 model
    """
    if not vision_service:
        raise HTTPException(status_code=503, detail="Vision service not available")
    
    try:
        contents = await file.read()
        result = await vision_service.analyze_image(contents)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Vision analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/audio/analyze", response_model=AudioResponse)
async def analyze_audio(
    file: UploadFile = File(...),
    include_transcription: bool = True
):
    """
    Analyze audio for:
    1. Speech-to-text transcription
    2. Text-based emotion (from transcript)
    3. Audio-based emotion (from prosody/acoustics)
    4. Combined emotion
    """
    if not all([stt_service, text_emotion_service, audio_emotion_service]):
        raise HTTPException(status_code=503, detail="Audio services not available")
    
    try:
        contents = await file.read()
        
        # 1. Transcribe audio
        transcription = await stt_service.transcribe(contents, method="whisper")
        
        if "error" in transcription:
            raise HTTPException(status_code=400, detail=transcription["error"])
        
        # 2. Analyze text emotion from transcript
        text_emotion = text_emotion_service.comprehensive_analysis(transcription["text"])
        
        # 3. Analyze audio emotion from acoustics
        audio_emotion = await audio_emotion_service.predict_emotion(contents)
        
        # 4. Combine results (weighted fusion)
        combined_emotion = _fuse_audio_emotions(
            text_emotion["emotion_analysis"],
            audio_emotion
        )
        
        return {
            "transcription": {
                "text": transcription["text"],
                "language": transcription.get("language", "en"),
                "confidence": transcription.get("confidence", 0.8)
            },
            "text_emotion": {
                "emotion": text_emotion["combined_emotion"],
                "confidence": text_emotion["combined_confidence"],
                "all_emotions": text_emotion["emotion_analysis"]["all_emotions"],
                "sentiment_scores": text_emotion["sentiment_analysis"]["sentiment_scores"]
            },
            "audio_emotion": audio_emotion,
            "combined_emotion": combined_emotion["emotion"],
            "combined_confidence": combined_emotion["confidence"]
        }
        
    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/multimodal/analyze", response_model=MultimodalResponse)
async def analyze_multimodal(
    image_file: UploadFile = File(None),
    audio_file: UploadFile = File(None)
):
    """
    Multimodal emotion detection:
    - Analyze image (face emotion)
    - Analyze audio (speech + prosody)
    - Fuse results for final prediction
    """
    if not image_file and not audio_file:
        raise HTTPException(status_code=400, detail="Provide at least one input (image or audio)")
    
    try:
        vision_result = None
        audio_result = None
        
        # 1. Analyze image
        if image_file:
            image_bytes = await image_file.read()
            vision_result = await vision_service.analyze_image(image_bytes)
        
        # 2. Analyze audio
        if audio_file:
            audio_bytes = await audio_file.read()
            
            # Transcription
            transcription = await stt_service.transcribe(audio_bytes)
            
            # Text emotion
            text_emotion = text_emotion_service.comprehensive_analysis(transcription["text"])
            
            # Audio emotion
            audio_emotion = await audio_emotion_service.predict_emotion(audio_bytes)
            
            # Combine audio modalities
            audio_combined = _fuse_audio_emotions(
                text_emotion["emotion_analysis"],
                audio_emotion
            )
            
            audio_result = {
                "transcription": transcription,
                "text_emotion": text_emotion,
                "audio_emotion": audio_emotion,
                "combined_emotion": audio_combined["emotion"],
                "combined_confidence": audio_combined["confidence"]
            }
        
        # 3. Fuse vision + audio
        final_result = _fuse_multimodal(vision_result, audio_result)
        
        return {
            "vision": vision_result,
            "audio": audio_result,
            "fusion": final_result["fusion_details"],
            "final_emotion": final_result["emotion"],
            "final_confidence": final_result["confidence"]
        }
        
    except Exception as e:
        logger.error(f"Multimodal analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _fuse_audio_emotions(text_emotion: Dict, audio_emotion: Dict) -> Dict:
    """
    Fuse text-based and audio-based emotion predictions
    Weight: 60% audio acoustics, 40% text sentiment
    """
    # Get probabilities
    text_probs = text_emotion.get("all_emotions", {})
    audio_probs = audio_emotion.get("all_emotions", {})
    
    # Weighted fusion
    fused_probs = {}
    all_emotions = set(list(text_probs.keys()) + list(audio_probs.keys()))
    
    for emotion in all_emotions:
        text_score = text_probs.get(emotion, 0.0)
        audio_score = audio_probs.get(emotion, 0.0)
        fused_probs[emotion] = 0.4 * text_score + 0.6 * audio_score
    
    # Get dominant
    dominant = max(fused_probs.items(), key=lambda x: x[1])
    
    return {
        "emotion": dominant[0],
        "confidence": dominant[1],
        "all_emotions": fused_probs
    }


def _fuse_multimodal(vision_result: Dict, audio_result: Dict) -> Dict:
    """
    Fuse vision and audio predictions
    Weight: 50% vision, 50% audio (adjustable based on confidence)
    """
    if not vision_result and not audio_result:
        return {"emotion": "Neutral", "confidence": 0.0, "fusion_details": {}}
    
    # Only vision
    if vision_result and not audio_result:
        return {
            "emotion": vision_result.get("dominant_emotion", "Neutral"),
            "confidence": vision_result.get("dominant_confidence", 0.0),
            "fusion_details": {"method": "vision_only"}
        }
    
    # Only audio
    if audio_result and not vision_result:
        return {
            "emotion": audio_result["combined_emotion"],
            "confidence": audio_result["combined_confidence"],
            "fusion_details": {"method": "audio_only"}
        }
    
    # Both modalities available
    vision_emotion = vision_result.get("dominant_emotion", "Neutral")
    vision_conf = vision_result.get("dominant_confidence", 0.0)
    
    audio_emotion = audio_result["combined_emotion"]
    audio_conf = audio_result["combined_confidence"]
    
    # Adaptive weighting based on confidence
    total_conf = vision_conf + audio_conf
    if total_conf > 0:
        vision_weight = vision_conf / total_conf
        audio_weight = audio_conf / total_conf
    else:
        vision_weight = 0.5
        audio_weight = 0.5
    
    # If both predict same emotion, boost confidence
    if vision_emotion == audio_emotion:
        return {
            "emotion": vision_emotion,
            "confidence": min(0.95, vision_conf * 0.5 + audio_conf * 0.5 + 0.2),
            "fusion_details": {
                "method": "agreement",
                "vision_weight": vision_weight,
                "audio_weight": audio_weight
            }
        }
    
    # Different predictions - use weighted average
    if vision_conf > audio_conf:
        final_emotion = vision_emotion
        final_conf = vision_conf * vision_weight + audio_conf * 0.3
    else:
        final_emotion = audio_emotion
        final_conf = audio_conf * audio_weight + vision_conf * 0.3
    
    return {
        "emotion": final_emotion,
        "confidence": final_conf,
        "fusion_details": {
            "method": "weighted_fusion",
            "vision": {"emotion": vision_emotion, "confidence": vision_conf, "weight": vision_weight},
            "audio": {"emotion": audio_emotion, "confidence": audio_conf, "weight": audio_weight}
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)