import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from same directory as this file
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from sqlalchemy import (
    Column, Integer, String, Float,
    DateTime, ForeignKey, Text, Boolean, JSON
)
from sqlalchemy.ext.asyncio import (
    AsyncSession, create_async_engine, async_sessionmaker
)
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "").replace(
    "postgresql://", "postgresql+asyncpg://")

engine           = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id             = Column(Integer, primary_key=True, index=True)
    email          = Column(String, unique=True, index=True,
                            nullable=False)
    name           = Column(String, nullable=False)
    password_hash  = Column(String, nullable=False)
    role           = Column(String, default="student")
    is_active      = Column(Boolean, default=True)
    created_at     = Column(DateTime, default=datetime.utcnow)
    sessions       = relationship("Session", back_populates="user")

class Course(Base):
    __tablename__ = "courses"
    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    teacher_id  = Column(Integer, ForeignKey("users.id"))
    created_at  = Column(DateTime, default=datetime.utcnow)
    lectures    = relationship("Lecture", back_populates="course")

class Lecture(Base):
    __tablename__ = "lectures"
    id         = Column(Integer, primary_key=True, index=True)
    course_id  = Column(Integer, ForeignKey("courses.id"))
    title      = Column(String, nullable=False)
    started_at = Column(DateTime, nullable=True)
    ended_at   = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    course     = relationship("Course", back_populates="lectures")
    sessions   = relationship("Session", back_populates="lecture")

class Session(Base):
    __tablename__ = "sessions"
    id                = Column(Integer, primary_key=True, index=True)
    user_id           = Column(Integer, ForeignKey("users.id"))
    lecture_id        = Column(Integer, ForeignKey("lectures.id"),
                               nullable=True)
    started_at        = Column(DateTime, default=datetime.utcnow)
    ended_at          = Column(DateTime, nullable=True)
    avg_engagement    = Column(Float, default=0.0)
    overall_engagement= Column(Float, default=0.0)
    dominant_emotion  = Column(String, nullable=True)
    total_detections  = Column(Integer, default=0)
    unique_emotions   = Column(Integer, default=0)
    distribution      = Column(JSON, default={})
    user    = relationship("User", back_populates="sessions")
    lecture = relationship("Lecture", back_populates="sessions")
    logs    = relationship("EmotionLog", back_populates="session")

class EmotionLog(Base):
    __tablename__ = "emotion_logs"
    id               = Column(Integer, primary_key=True, index=True)
    session_id       = Column(Integer, ForeignKey("sessions.id"))
    timestamp        = Column(DateTime, default=datetime.utcnow)
    emotion          = Column(String, nullable=False)
    confidence       = Column(Float, default=0.0)
    source           = Column(String, default="vision")
    scores           = Column(JSON, default={})
    engagement_score = Column(Float, default=0.0)
    session = relationship("Session", back_populates="logs")

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database tables created")


# ══════════════════════════════════════════════════════════════════════════
# NEW TABLES — Course enrollment, Exams, Submissions, Notifications
# Appended below — nothing above was changed.
# ══════════════════════════════════════════════════════════════════════════

class Enrollment(Base):
    __tablename__ = "enrollments"
    id          = Column(Integer, primary_key=True, index=True)
    student_id  = Column(Integer, ForeignKey("users.id"), nullable=False)
    course_id   = Column(Integer, ForeignKey("courses.id"), nullable=False)
    enrolled_at = Column(DateTime, default=datetime.utcnow)
    student     = relationship("User")
    course      = relationship("Course")


class Exam(Base):
    __tablename__ = "exams"
    id          = Column(Integer, primary_key=True, index=True)
    course_id   = Column(Integer, ForeignKey("courses.id"), nullable=False)
    title       = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    questions   = Column(JSON, nullable=False)     # [{q, o, a}, ...]
    time_limit  = Column(Integer, default=600)     # seconds
    is_proctored= Column(Boolean, default=True)
    due_date    = Column(DateTime, nullable=True)
    created_by  = Column(Integer, ForeignKey("users.id"))
    created_at  = Column(DateTime, default=datetime.utcnow)
    course      = relationship("Course")
    creator     = relationship("User")
    submissions = relationship("ExamSubmission", back_populates="exam")


class ExamSubmission(Base):
    __tablename__ = "exam_submissions"
    id             = Column(Integer, primary_key=True, index=True)
    exam_id        = Column(Integer, ForeignKey("exams.id"), nullable=False)
    student_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    answers        = Column(JSON, default={})       # {0: 1, 1: 2, ...}
    score          = Column(Float, default=0.0)     # percentage
    total_correct  = Column(Integer, default=0)
    total_questions= Column(Integer, default=0)
    focus_score    = Column(Float, default=100.0)
    focus_log      = Column(JSON, default=[])       # [{ts, s, d, c}, ...]
    alerts         = Column(JSON, default=[])       # [{t, m}, ...]
    answer_timing  = Column(JSON, default={})       # {0: 12, 1: 25, ...}
    gap_warnings   = Column(JSON, default=[])       # [{q, gap, ts}, ...]
    duration_sec   = Column(Integer, default=0)
    submitted_at   = Column(DateTime, default=datetime.utcnow)
    exam           = relationship("Exam", back_populates="submissions")
    student        = relationship("User")


class Notification(Base):
    __tablename__ = "notifications"
    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    type       = Column(String, default="exam")     # exam, course, system
    title      = Column(String, nullable=False)
    message    = Column(Text, nullable=True)
    link       = Column(String, nullable=True)      # e.g. /exam?id=5
    is_read    = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user       = relationship("User")


# ══════════════════════════════════════════════════════════════════════════
# NEW TABLES — Admin features + Chat system
# Appended below — nothing above was changed.
# ══════════════════════════════════════════════════════════════════════════

class AuditLog(Base):
    """Tracks all significant actions for admin audit trail."""
    __tablename__ = "audit_logs"
    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=True)
    action     = Column(String, nullable=False)       # login, logout, grade_change, user_create, etc.
    target     = Column(String, nullable=True)         # what was acted on (user:5, course:3, etc.)
    details    = Column(JSON, default={})              # extra context
    ip_address = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user       = relationship("User")


class ConsentRecord(Base):
    """Tracks student consent for emotion tracking / data collection."""
    __tablename__ = "consent_records"
    id          = Column(Integer, primary_key=True, index=True)
    student_id  = Column(Integer, ForeignKey("users.id"), nullable=False)
    consent_type= Column(String, nullable=False)       # emotion_tracking, data_storage, video_recording
    granted     = Column(Boolean, default=False)
    granted_at  = Column(DateTime, nullable=True)
    revoked_at  = Column(DateTime, nullable=True)
    ip_address  = Column(String, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)
    student     = relationship("User")


class Conversation(Base):
    """A DM thread between two users (student↔teacher or teacher↔admin)."""
    __tablename__ = "conversations"
    id          = Column(Integer, primary_key=True, index=True)
    user1_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    user2_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    last_message= Column(Text, nullable=True)
    last_at     = Column(DateTime, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)
    user1       = relationship("User", foreign_keys=[user1_id])
    user2       = relationship("User", foreign_keys=[user2_id])
    messages    = relationship("Message", back_populates="conversation",
                               order_by="Message.created_at")


class Message(Base):
    """A single chat message within a conversation."""
    __tablename__ = "messages"
    id              = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"),
                             nullable=False)
    sender_id       = Column(Integer, ForeignKey("users.id"), nullable=False)
    content         = Column(Text, nullable=False)
    is_read         = Column(Boolean, default=False)
    created_at      = Column(DateTime, default=datetime.utcnow)
    conversation    = relationship("Conversation", back_populates="messages")
    sender          = relationship("User")


class SystemAnnouncement(Base):
    """Admin broadcast messages visible to all users."""
    __tablename__ = "system_announcements"
    id         = Column(Integer, primary_key=True, index=True)
    admin_id   = Column(Integer, ForeignKey("users.id"), nullable=False)
    title      = Column(String, nullable=False)
    content    = Column(Text, nullable=False)
    priority   = Column(String, default="normal")      # low, normal, high, critical
    target_role= Column(String, nullable=True)          # null=all, student, teacher
    is_active  = Column(Boolean, default=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    admin      = relationship("User")
