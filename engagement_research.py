"""
Research-Based Engagement Detection Module
===========================================

Based on peer-reviewed studies:

1. Whitehill, J., et al. (2014). "The Faces of Engagement: Automatic 
   Recognition of Student Engagement from Facial Expressions"
   IEEE Transactions on Affective Computing.
   
2. D'Mello, S., & Graesser, A. (2012). "Dynamics of Affective States 
   during Complex Learning"
   Learning and Instruction.

3. Bosch, N., et al. (2016). "Detecting Student Emotions in 
   Computer-Enabled Classrooms"
   IJCAI.

Key Research Findings Implemented:
----------------------------------
- Engagement ≠ Happiness (common misconception)
- Confusion can indicate productive cognitive struggle
- Boredom (prolonged neutral + low variation) = true disengagement
- Frustration that resolves = learning; sustained frustration = disengagement
- Flow state often appears as focused neutral expression
- Sudden emotion changes indicate attention/reactivity

Engagement Model:
-----------------
We use a 4-level model based on Bosch et al.:
  1. Disengaged (0-25%)   : Boredom, prolonged inattention
  2. Barely Engaged (25-50%): Passive, minimal response
  3. Engaged (50-75%)      : Active attention, some affect variation
  4. Highly Engaged (75-100%): Flow state, productive struggle, high reactivity
"""

from collections import deque
from datetime import datetime
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH-BASED ENGAGEMENT WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

# Based on D'Mello & Graesser (2012) - Affective states in learning
EMOTION_ENGAGEMENT_MAP = {
    # Emotion: (base_engagement, cognitive_load, is_productive)
    
    # HIGH ENGAGEMENT indicators
    "surprise": (0.80, 0.7, True),    # Indicates attention capture, novelty detection
    "fear": (0.60, 0.8, True),        # Can indicate challenge/productive anxiety
    
    # MODERATE-HIGH - Context dependent
    "happiness": (0.65, 0.4, True),   # Positive but may indicate ease, not struggle
    "anger": (0.55, 0.7, True),       # Frustration CAN be productive (Bosch 2016)
    "disgust": (0.45, 0.5, False),    # Usually negative, rejection of content
    
    # CONTEXT-DEPENDENT - Key insight from Whitehill (2014)
    "neutral": (0.50, 0.5, None),     # DEPENDS on context! Flow state vs boredom
    
    # LOW ENGAGEMENT
    "sadness": (0.30, 0.3, False),    # Withdrawal, giving up
}

# Temporal thresholds (in number of detections)
BOREDOM_THRESHOLD = 15        # Neutral for 15+ detections = boredom
FRUSTRATION_THRESHOLD = 10    # Negative emotion for 10+ = stuck
VARIETY_WINDOW = 20           # Look at last 20 detections for variety

class ResearchBasedEngagement:
    """
    Engagement detector based on affective computing research.
    
    Unlike simple emotion-to-engagement mapping, this considers:
    1. Temporal patterns (how emotions change over time)
    2. Cognitive load indicators
    3. Productive vs unproductive affect
    4. Flow state detection
    5. Boredom detection
    """
    
    def __init__(self):
        self.emotion_history = deque(maxlen=100)
        self.engagement_history = deque(maxlen=100)
        self.session_start = None
        self.face_detection_rate = deque(maxlen=30)  # Track if face visible
        
        # State tracking
        self.current_streak = 0
        self.streak_emotion = None
        self.productive_struggle_count = 0
        self.total_detections = 0
        
    def reset(self):
        """Reset for new session"""
        self.emotion_history.clear()
        self.engagement_history.clear()
        self.session_start = datetime.now().isoformat()
        self.face_detection_rate.clear()
        self.current_streak = 0
        self.streak_emotion = None
        self.productive_struggle_count = 0
        self.total_detections = 0
        
    def compute_engagement(self, emotion: str, scores: dict, 
                           face_detected: bool = True) -> dict:
        """
        Compute engagement using research-based model.
        
        Returns detailed engagement analysis including:
        - engagement_score: 0-100
        - engagement_level: text label
        - cognitive_load: estimated mental effort
        - attention_state: focused/distracted/bored
        - is_productive: whether current state aids learning
        """
        self.total_detections += 1
        self.face_detection_rate.append(1 if face_detected else 0)
        
        # Update emotion tracking
        self._update_streak(emotion)
        self.emotion_history.append({
            "emotion": emotion,
            "scores": scores,
            "time": datetime.now().isoformat()
        })
        
        # ══════════════════════════════════════════════════════════════════
        # COMPONENT 1: Base engagement from emotion (Whitehill 2014)
        # ══════════════════════════════════════════════════════════════════
        emotion_data = EMOTION_ENGAGEMENT_MAP.get(
            emotion, (0.50, 0.5, None)
        )
        base_engagement = emotion_data[0]
        cognitive_load = emotion_data[1]
        is_productive = emotion_data[2]
        
        # ══════════════════════════════════════════════════════════════════
        # COMPONENT 2: Neutral context analysis (Key insight!)
        # Neutral can mean: Flow state (good) OR Boredom (bad)
        # ══════════════════════════════════════════════════════════════════
        neutral_adjustment = 0.0
        attention_state = "focused"
        
        if emotion == "neutral":
            neutral_context = self._analyze_neutral_context()
            
            if neutral_context["is_flow_state"]:
                # Flow state: focused neutral with occasional variations
                neutral_adjustment = +0.25
                attention_state = "flow"
                is_productive = True
                cognitive_load = 0.7
            elif neutral_context["is_boredom"]:
                # Boredom: prolonged unchanging neutral
                neutral_adjustment = -0.30
                attention_state = "bored"
                is_productive = False
                cognitive_load = 0.1
            else:
                # Passive but not bored
                neutral_adjustment = 0.0
                attention_state = "passive"
        
        # ══════════════════════════════════════════════════════════════════
        # COMPONENT 3: Productive struggle detection (D'Mello 2012)
        # Confusion/frustration that leads to resolution = GOOD
        # ══════════════════════════════════════════════════════════════════
        struggle_bonus = 0.0
        
        if emotion in ["anger", "fear", "disgust"]:
            # Check if this is productive struggle
            if self._is_productive_struggle():
                struggle_bonus = +0.15
                is_productive = True
                self.productive_struggle_count += 1
            elif self._is_stuck():
                # Prolonged frustration = giving up
                struggle_bonus = -0.20
                is_productive = False
                attention_state = "frustrated"
        
        # ══════════════════════════════════════════════════════════════════
        # COMPONENT 4: Emotional variety (Bosch 2016)
        # More variety in expressions = more responsive = more engaged
        # ══════════════════════════════════════════════════════════════════
        variety_score = self._compute_variety_score()
        variety_bonus = (variety_score - 0.5) * 0.2  # -0.1 to +0.1
        
        # ══════════════════════════════════════════════════════════════════
        # COMPONENT 5: Attention indicator (face detection rate)
        # Looking at screen = potentially engaged
        # ══════════════════════════════════════════════════════════════════
        face_rate = self._get_face_detection_rate()
        attention_bonus = (face_rate - 0.7) * 0.3  # Penalty if often not looking
        
        if face_rate < 0.5:
            attention_state = "distracted"
        
        # ══════════════════════════════════════════════════════════════════
        # COMPONENT 6: Expression intensity
        # Stronger expressions = more arousal = more engaged
        # ══════════════════════════════════════════════════════════════════
        confidence = scores.get(emotion, 0.5)
        intensity_bonus = (confidence - 0.5) * 0.15
        
        # ══════════════════════════════════════════════════════════════════
        # FINAL CALCULATION
        # ══════════════════════════════════════════════════════════════════
        engagement = (
            base_engagement +
            neutral_adjustment +
            struggle_bonus +
            variety_bonus +
            attention_bonus +
            intensity_bonus
        )
        
        # Clamp to valid range
        engagement = max(0.05, min(1.0, engagement))
        engagement_pct = round(engagement * 100, 1)
        
        # Determine level label
        if engagement_pct >= 75:
            level = "Highly Engaged"
        elif engagement_pct >= 50:
            level = "Engaged"
        elif engagement_pct >= 25:
            level = "Barely Engaged"
        else:
            level = "Disengaged"
        
        # Store result
        result = {
            "engagement_score": engagement_pct,
            "engagement_level": level,
            "engagement_raw": round(engagement, 4),
            "cognitive_load": round(cognitive_load * 100, 1),
            "attention_state": attention_state,
            "is_productive": is_productive,
            "emotion_variety": round(variety_score * 100, 1),
            "face_detection_rate": round(face_rate * 100, 1),
            "components": {
                "base": round(base_engagement, 3),
                "neutral_adj": round(neutral_adjustment, 3),
                "struggle": round(struggle_bonus, 3),
                "variety": round(variety_bonus, 3),
                "attention": round(attention_bonus, 3),
                "intensity": round(intensity_bonus, 3),
            }
        }
        
        self.engagement_history.append({
            "time": datetime.now().isoformat(),
            "score": engagement,
            "level": level,
            "emotion": emotion,
            "attention": attention_state,
        })
        
        return result
    
    def _update_streak(self, emotion: str):
        """Track consecutive same-emotion detections"""
        if emotion == self.streak_emotion:
            self.current_streak += 1
        else:
            self.streak_emotion = emotion
            self.current_streak = 1
    
    def _analyze_neutral_context(self) -> dict:
        """
        Determine if neutral is flow state or boredom.
        
        Flow state indicators:
        - Short neutral periods between other emotions
        - Occasional micro-expressions breaking through
        - Consistent face detection (looking at screen)
        
        Boredom indicators:
        - Very long neutral streak
        - No variety in recent history
        - May have face looking away
        """
        # Check neutral streak length
        is_long_neutral = self.current_streak >= BOREDOM_THRESHOLD
        
        # Check recent variety
        recent = list(self.emotion_history)[-VARIETY_WINDOW:]
        unique_recent = len(set(e["emotion"] for e in recent)) if recent else 1
        has_variety = unique_recent >= 3
        
        # Check face detection rate
        face_rate = self._get_face_detection_rate()
        is_attentive = face_rate >= 0.8
        
        # Flow: attentive + some variety, even with current neutral
        is_flow = is_attentive and has_variety and self.current_streak < 10
        
        # Boredom: long neutral + no variety
        is_boredom = is_long_neutral and not has_variety
        
        return {
            "is_flow_state": is_flow,
            "is_boredom": is_boredom,
            "neutral_streak": self.current_streak,
            "variety_count": unique_recent,
        }
    
    def _is_productive_struggle(self) -> bool:
        """
        Check if current frustration/confusion is productive.
        
        Productive struggle (D'Mello 2012):
        - Short bursts of negative affect
        - Followed by resolution (neutral or positive)
        - Part of learning cycle
        """
        # If we're early in a negative streak, could be productive
        if self.current_streak <= 5:
            return True
        
        # Check if there's been resolution recently
        recent = list(self.emotion_history)[-10:]
        positive_count = sum(
            1 for e in recent 
            if e["emotion"] in ["happiness", "surprise", "neutral"]
        )
        
        # If there's been some positive/neutral, struggle is productive
        return positive_count >= 3
    
    def _is_stuck(self) -> bool:
        """Check if learner is stuck (prolonged frustration)"""
        return self.current_streak >= FRUSTRATION_THRESHOLD
    
    def _compute_variety_score(self) -> float:
        """
        Compute emotional variety in recent history.
        More variety = more responsive = more engaged
        """
        if len(self.emotion_history) < 5:
            return 0.5  # Not enough data
        
        recent = list(self.emotion_history)[-VARIETY_WINDOW:]
        unique_emotions = len(set(e["emotion"] for e in recent))
        
        # Normalize: 1 emotion = 0, 7 emotions = 1
        variety = (unique_emotions - 1) / 6
        return min(1.0, max(0.0, variety))
    
    def _get_face_detection_rate(self) -> float:
        """Get recent face detection rate"""
        if not self.face_detection_rate:
            return 1.0
        return sum(self.face_detection_rate) / len(self.face_detection_rate)
    
    def get_session_summary(self) -> dict:
        """Get overall session engagement summary"""
        if not self.engagement_history:
            return {"error": "No data yet"}
        
        scores = [e["score"] for e in self.engagement_history]
        levels = [e["level"] for e in self.engagement_history]
        
        # Calculate time in each state
        level_counts = {}
        for level in levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        total = len(levels)
        level_distribution = {
            k: round(v / total * 100, 1) 
            for k, v in level_counts.items()
        }
        
        return {
            "session_start": self.session_start,
            "total_detections": self.total_detections,
            "avg_engagement": round(np.mean(scores) * 100, 1),
            "min_engagement": round(min(scores) * 100, 1),
            "max_engagement": round(max(scores) * 100, 1),
            "std_engagement": round(np.std(scores) * 100, 1),
            "level_distribution": level_distribution,
            "productive_struggles": self.productive_struggle_count,
            "engagement_trend": list(self.engagement_history)[-50:],
        }


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# Global instance
engagement_detector = ResearchBasedEngagement()

def compute_engagement(emotion: str, scores: dict, confidence: float) -> float:
    """
    Drop-in replacement for the old compute_engagement function.
    Returns just the score (0-1) for backward compatibility.
    """
    result = engagement_detector.compute_engagement(emotion, scores)
    return result["engagement_raw"]

def compute_engagement_detailed(emotion: str, scores: dict, 
                                face_detected: bool = True) -> dict:
    """
    Full detailed engagement analysis.
    Use this for the API response.
    """
    return engagement_detector.compute_engagement(emotion, scores, face_detected)

def reset_engagement_session():
    """Reset for new session"""
    engagement_detector.reset()

def get_engagement_summary() -> dict:
    """Get session summary"""
    return engagement_detector.get_session_summary()
