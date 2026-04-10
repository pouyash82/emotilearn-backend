from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from models_db import (
    User, Course, Lecture, Session, EmotionLog, get_db)
from auth import (
    hash_password, verify_password, create_token,
    get_current_user, get_current_teacher, get_current_admin)

router = APIRouter()

# ── Schemas ────────────────────────────────────────────────────────────────
class UserRegister(BaseModel):
    email   : str
    name    : str
    password: str
    role    : str = "student"

class UserLogin(BaseModel):
    email   : str
    password: str

class CourseCreate(BaseModel):
    name       : str
    description: Optional[str] = None

class LectureCreate(BaseModel):
    course_id: int
    title    : str

class SessionSave(BaseModel):
    lecture_id        : Optional[int] = None
    avg_engagement    : float
    overall_engagement: float
    dominant_emotion  : str
    total_detections  : int
    unique_emotions   : int
    distribution      : dict
    emotion_logs      : list

# ── Auth ───────────────────────────────────────────────────────────────────
@router.post("/auth/register")
async def register(data: UserRegister,
                   db  : AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(User).where(User.email == data.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=400, detail="Email already registered")
    user = User(
        email        = data.email,
        name         = data.name,
        password_hash= hash_password(data.password),
        role         = data.role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    token = create_token({"sub": str(user.id)})
    return {
        "access_token": token,
        "token_type"  : "bearer",
        "user": {"id": user.id, "name": user.name,
                 "role": user.role, "email": user.email}
    }

@router.post("/auth/login")
async def login(data: UserLogin,
                db  : AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(User).where(User.email == data.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(
            data.password, user.password_hash):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password")
    token = create_token({"sub": str(user.id)})
    return {
        "access_token": token,
        "token_type"  : "bearer",
        "user": {"id": user.id, "name": user.name,
                 "role": user.role, "email": user.email}
    }

@router.get("/auth/me")
async def me(current_user: User = Depends(get_current_user)):
    return {
        "id"        : current_user.id,
        "name"      : current_user.name,
        "email"     : current_user.email,
        "role"      : current_user.role,
        "created_at": current_user.created_at,
    }

# ── Student ────────────────────────────────────────────────────────────────
@router.get("/students/sessions")
async def get_my_sessions(
        current_user: User = Depends(get_current_user),
        db          : AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Session)
        .where(Session.user_id == current_user.id)
        .order_by(desc(Session.started_at))
        .limit(20))
    sessions = result.scalars().all()
    return [{
        "id"              : s.id,
        "started_at"      : s.started_at,
        "ended_at"        : s.ended_at,
        "avg_engagement"  : round(s.avg_engagement * 100, 1),
        "dominant_emotion": s.dominant_emotion,
        "total_detections": s.total_detections,
        "distribution"    : s.distribution,
    } for s in sessions]

@router.get("/students/stats")
async def get_my_stats(
        current_user: User = Depends(get_current_user),
        db          : AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(
            func.count(Session.id).label("total_sessions"),
            func.avg(Session.avg_engagement).label("avg_eng"),
            func.sum(Session.total_detections).label("total_det"),
        ).where(Session.user_id == current_user.id))
    row = result.one()
    return {
        "total_sessions"  : row.total_sessions or 0,
        "avg_engagement"  : round((row.avg_eng or 0) * 100, 1),
        "total_detections": row.total_det or 0,
    }

@router.post("/sessions/save")
async def save_session(
        data        : SessionSave,
        current_user: User = Depends(get_current_user),
        db          : AsyncSession = Depends(get_db)):
    session = Session(
        user_id           = current_user.id,
        lecture_id        = data.lecture_id,
        ended_at          = datetime.utcnow(),
        avg_engagement    = data.avg_engagement / 100,
        overall_engagement= data.overall_engagement / 100,
        dominant_emotion  = data.dominant_emotion,
        total_detections  = data.total_detections,
        unique_emotions   = data.unique_emotions,
        distribution      = data.distribution,
    )
    db.add(session)
    await db.flush()
    for log in data.emotion_logs[-200:]:
        db.add(EmotionLog(
            session_id      = session.id,
            timestamp       = datetime.fromisoformat(
                log.get("time",
                        datetime.utcnow().isoformat())),
            emotion         = log.get("emotion", "neutral"),
            confidence      = log.get("confidence", 0.0),
            source          = log.get("source", "vision"),
            scores          = log.get("scores", {}),
            engagement_score= log.get("engagement_score", 0.0),
        ))
    await db.commit()
    return {"status": "saved", "session_id": session.id}

# ── Teacher ────────────────────────────────────────────────────────────────
@router.post("/courses")
async def create_course(
        data   : CourseCreate,
        teacher: User = Depends(get_current_teacher),
        db     : AsyncSession = Depends(get_db)):
    course = Course(name=data.name,
                    description=data.description,
                    teacher_id=teacher.id)
    db.add(course)
    await db.commit()
    await db.refresh(course)
    return {"id": course.id, "name": course.name}

@router.get("/courses")
async def get_courses(
        teacher: User = Depends(get_current_teacher),
        db     : AsyncSession = Depends(get_db)):
    result  = await db.execute(
        select(Course).where(Course.teacher_id == teacher.id))
    courses = result.scalars().all()
    return [{"id": c.id, "name": c.name,
             "description": c.description} for c in courses]

@router.post("/lectures")
async def create_lecture(
        data   : LectureCreate,
        teacher: User = Depends(get_current_teacher),
        db     : AsyncSession = Depends(get_db)):
    lecture = Lecture(course_id=data.course_id,
                      title=data.title,
                      started_at=datetime.utcnow())
    db.add(lecture)
    await db.commit()
    await db.refresh(lecture)
    return {"id": lecture.id, "title": lecture.title}

@router.get("/teacher/analytics")
async def teacher_analytics(
        teacher: User = Depends(get_current_teacher),
        db     : AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Session, User.name.label("student_name"))
        .join(User, Session.user_id == User.id)
        .join(Lecture, Session.lecture_id == Lecture.id,
              isouter=True)
        .join(Course, Lecture.course_id == Course.id,
              isouter=True)
        .where(Course.teacher_id == teacher.id)
        .order_by(desc(Session.started_at))
        .limit(50))
    rows = result.all()
    return [{
        "session_id"      : r.Session.id,
        "student_name"    : r.student_name,
        "started_at"      : r.Session.started_at,
        "avg_engagement"  : round(r.Session.avg_engagement*100, 1),
        "dominant_emotion": r.Session.dominant_emotion,
        "total_detections": r.Session.total_detections,
    } for r in rows]

# ── Admin ──────────────────────────────────────────────────────────────────
@router.get("/admin/users")
async def get_all_users(
        admin: User = Depends(get_current_admin),
        db   : AsyncSession = Depends(get_db)):
    result = await db.execute(select(User))
    users  = result.scalars().all()
    return [{"id": u.id, "name": u.name,
             "email": u.email, "role": u.role}
            for u in users]

@router.get("/admin/stats")
async def admin_stats(
        admin: User = Depends(get_current_admin),
        db   : AsyncSession = Depends(get_db)):
    users    = await db.execute(select(func.count(User.id)))
    sessions = await db.execute(select(func.count(Session.id)))
    logs     = await db.execute(
        select(func.count(EmotionLog.id)))
    return {
        "total_users"   : users.scalar(),
        "total_sessions": sessions.scalar(),
        "total_logs"    : logs.scalar(),
    }
