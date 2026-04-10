import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import cv2
import numpy as np

# Emotion labels (7 basic emotions from RAF-DB)
EMOTION_LABELS = ['neutral', 'happiness', 'sadness', 'surprise', 'fear', 'disgust', 'anger']

class EmotionDetector:
    def __init__(self, model_path: str = "models/efficientnet_b2_finetuned_best.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load EfficientNet-B2
        self.model = models.efficientnet_b2(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 7)
        
        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        print("Emotion detector initialized successfully!")
    
    def detect_face(self, image: np.ndarray):
        """Detect face in image and return cropped face region"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None, None
        
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Add padding
        pad = int(0.2 * w)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        
        face_img = image[y1:y2, x1:x2]
        return face_img, (x, y, w, h)
    
    def predict_from_base64(self, base64_string: str) -> dict:
        """Predict emotion from base64 encoded image"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64 to image
            img_bytes = base64.b64decode(base64_string)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_np = np.array(img)
            
            # Detect face
            face_img, bbox = self.detect_face(img_np)
            
            if face_img is None:
                return {
                    "success": False,
                    "message": "No face detected",
                    "emotions": {label: 0.0 for label in EMOTION_LABELS},
                    "dominant": "neutral"
                }
            
            # Convert to PIL and preprocess
            face_pil = Image.fromarray(face_img)
            input_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
            
            # Create response
            emotions = {label: round(float(probs[i]) * 100, 1) for i, label in enumerate(EMOTION_LABELS)}
            dominant = EMOTION_LABELS[torch.argmax(probs).item()]
            
            return {
                "success": True,
                "emotions": emotions,
                "dominant": dominant,
                "bbox": bbox
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {
                "success": False,
                "message": str(e),
                "emotions": {label: 0.0 for label in EMOTION_LABELS},
                "dominant": "neutral"
            }

# Global instance
detector = None

def get_detector():
    global detector
    if detector is None:
        detector = EmotionDetector()
    return detector