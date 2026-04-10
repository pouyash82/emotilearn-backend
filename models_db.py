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
