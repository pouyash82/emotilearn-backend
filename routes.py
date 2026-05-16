import io, csv, json
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, or_, and_
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
from models_db import (
    User, Course, Lecture, Session, EmotionLog,
    Enrollment, Exam, ExamSubmission, Notification,
    AuditLog, ConsentRecord, Conversation, Message,
    SystemAnnouncement, get_db)
from auth import (
    hash_password, verify_password, create_token,
    get_current_user, get_current_teacher, get_current_admin)

router = APIRouter()

CLASS_NAMES = ["anger", "disgust", "fear",
               "happiness", "neutral", "sadness", "surprise"]

# ══════════════════════════════════════════════════════════════════════════
# Schemas
# ══════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════
# Report helpers (CSV + HTML) — shared between student self-access and
# teacher drill-down so both sides render exactly the same report.
# ══════════════════════════════════════════════════════════════════════════
def _session_csv_response(session, logs, student_name=""):
    """Build a CSV download for a single DB-stored session."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "timestamp", "emotion", "confidence", "source",
        "anger", "disgust", "fear", "happiness",
        "neutral", "sadness", "surprise", "engagement_score"
    ])
    for log in logs:
        scores = log.scores or {}
        ts = log.timestamp.isoformat() if log.timestamp else ""
        writer.writerow([
            ts,
            log.emotion,
            log.confidence,
            log.source,
            scores.get("anger",     ""),
            scores.get("disgust",   ""),
            scores.get("fear",      ""),
            scores.get("happiness", ""),
            scores.get("neutral",   ""),
            scores.get("sadness",   ""),
            scores.get("surprise",  ""),
            round((log.engagement_score or 0) * 100, 1),
        ])
    output.seek(0)
    safe = (student_name or "student").replace(" ", "_").replace("/", "_")
    fname = (f"session_{session.id}_{safe}_"
             f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition":
                 f"attachment; filename={fname}"})

def _session_html_report(session, logs, student, course_name=None):
    """Build an HTML report for a single DB-stored session."""
    distribution = session.distribution or {}
    avg_eng      = round((session.avg_engagement or 0) * 100, 1)
    eng_label = ("High"   if avg_eng >= 65
                 else "Medium" if avg_eng >= 40 else "Low")
    eng_color = ("#22c55e" if avg_eng >= 65
                 else "#eab308" if avg_eng >= 40 else "#ef4444")

    dist_rows = ""
    for e, pct in sorted(distribution.items(), key=lambda x: -x[1]):
        bar = "█" * int(float(pct) / 5)
        dist_rows += f"""
        <tr>
          <td>{e.capitalize()}</td>
          <td>{pct}%</td>
          <td style="font-family:monospace;
                     color:#7c3aed">{bar}</td>
        </tr>"""

    timeline_rows = ""
    for log in logs[-20:]:
        t   = log.timestamp.strftime("%H:%M:%S") if log.timestamp else "-"
        eng = round((log.engagement_score or 0) * 100, 1)
        timeline_rows += f"""
        <tr>
          <td>{t}</td>
          <td>{log.emotion.capitalize()}</td>
          <td>{eng}%</td>
        </tr>"""

    started = (session.started_at.strftime("%Y-%m-%d %H:%M:%S")
               if session.started_at else "N/A")
    ended   = (session.ended_at.strftime("%Y-%m-%d %H:%M:%S")
               if session.ended_at else "—")
    now     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    course_line = (f'<span>Course: {course_name}</span>'
                   if course_name else "")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Session Report — {student.name}</title>
<style>
  body {{ font-family:-apple-system,sans-serif;
          max-width:800px; margin:40px auto;
          padding:0 24px; color:#1a1a1a; }}
  h1   {{ color:#7c3aed;
          border-bottom:2px solid #7c3aed;
          padding-bottom:8px; }}
  h2   {{ color:#444; margin-top:32px; }}
  .info-card {{ background:#f5f3ff; border-radius:12px;
                padding:16px; margin:16px 0;
                border:1px solid #e0d9f9; }}
  .info-row  {{ display:flex; gap:24px; flex-wrap:wrap;
                font-size:13px; color:#666; margin-top:8px; }}
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
  <h1>🧠 Learning Session Report</h1>
  <p style="color:#666">Generated: {now}</p>

  <div class="info-card">
    <strong style="color:#7c3aed">Student:</strong>
    {student.name}
    <span style="color:#666">({student.email})</span>
    <div class="info-row">
      <span>Session #{session.id}</span>
      <span>Started: {started}</span>
      <span>Ended: {ended}</span>
      {course_line}
    </div>
  </div>

  <h2>Summary</h2>
  <div class="stat-grid">
    <div class="stat-box">
      <div class="stat-val">{session.total_detections or 0}</div>
      <div class="stat-lbl">Total Detections</div>
    </div>
    <div class="stat-box">
      <div class="stat-val" style="color:{eng_color}">
        {avg_eng}%
      </div>
      <div class="stat-lbl">
        Avg Engagement ({eng_label})
      </div>
    </div>
    <div class="stat-box">
      <div class="stat-val">
        {session.unique_emotions or 0}/7
      </div>
      <div class="stat-lbl">Emotion Variety</div>
    </div>
  </div>
  <p>Dominant emotion:
    <span class="badge">
      {(session.dominant_emotion or "N/A").upper()}
    </span>
  </p>

  <h2>Emotion Distribution</h2>
  <table>
    <tr>
      <th>Emotion</th><th>Percentage</th><th>Visual</th>
    </tr>
    {dist_rows or
     '<tr><td colspan="3" style="text-align:center;color:#999">'
     'No distribution data</td></tr>'}
  </table>

  <h2>Recent Engagement Timeline (last 20)</h2>
  <table>
    <tr><th>Time</th><th>Emotion</th><th>Engagement</th></tr>
    {timeline_rows or
     '<tr><td colspan="3" style="text-align:center;color:#999">'
     'No timeline data</td></tr>'}
  </table>

  <div class="footer">
    Generated by EmotiLearn |
    EfficientNet-B2 Vision Model | Session #{session.id}
  </div>
</body>
</html>"""
    fname = (f"session_{session.id}_report_"
             f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    return StreamingResponse(
        io.BytesIO(html.encode()),
        media_type="text/html",
        headers={"Content-Disposition":
                 f"attachment; filename={fname}"})

# ══════════════════════════════════════════════════════════════════════════
# Auth (unchanged)
# ══════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════
# Student — existing (unchanged)
# ══════════════════════════════════════════════════════════════════════════
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
        "avg_engagement"  : round((s.avg_engagement or 0) * 100, 1),
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
                        datetime.utcnow().isoformat()
                ).replace("Z", "+00:00")).replace(tzinfo=None),
            emotion         = log.get("emotion", "neutral"),
            confidence      = log.get("confidence", 0.0),
            source          = log.get("source", "vision"),
            scores          = log.get("scores", {}),
            engagement_score= log.get("engagement_score", 0.0),
        ))
    await db.commit()
    return {"status": "saved", "session_id": session.id}

# ══════════════════════════════════════════════════════════════════════════
# Student — NEW: download own report & CSV for a specific past session
# ══════════════════════════════════════════════════════════════════════════
@router.get("/students/sessions/{session_id}/report")
async def student_session_report(
        session_id  : int,
        current_user: User = Depends(get_current_user),
        db          : AsyncSession = Depends(get_db)):
    sess_res = await db.execute(
        select(Session).where(
            Session.id == session_id,
            Session.user_id == current_user.id))
    session = sess_res.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Session not found")

    logs_res = await db.execute(
        select(EmotionLog)
        .where(EmotionLog.session_id == session_id)
        .order_by(EmotionLog.timestamp))
    logs = logs_res.scalars().all()

    return _session_html_report(session, logs, current_user)

@router.get("/students/sessions/{session_id}/csv")
async def student_session_csv(
        session_id  : int,
        current_user: User = Depends(get_current_user),
        db          : AsyncSession = Depends(get_db)):
    sess_res = await db.execute(
        select(Session).where(
            Session.id == session_id,
            Session.user_id == current_user.id))
    session = sess_res.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Session not found")

    logs_res = await db.execute(
        select(EmotionLog)
        .where(EmotionLog.session_id == session_id)
        .order_by(EmotionLog.timestamp))
    logs = logs_res.scalars().all()

    return _session_csv_response(session, logs, current_user.name)

# ══════════════════════════════════════════════════════════════════════════
# Teacher — existing course/lecture/analytics endpoints (unchanged)
# ══════════════════════════════════════════════════════════════════════════
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
        "avg_engagement"  : round((r.Session.avg_engagement or 0)*100, 1),
        "dominant_emotion": r.Session.dominant_emotion,
        "total_detections": r.Session.total_detections,
    } for r in rows]

# ══════════════════════════════════════════════════════════════════════════
# Teacher — NEW: overview stats, students list, per-student drill-down,
# per-session reports/CSV. These are what the new TeacherDashboard.jsx and
# StudentAnalyticsModal call.
# ══════════════════════════════════════════════════════════════════════════
@router.get("/teacher/stats")
async def teacher_stats(
        teacher: User = Depends(get_current_teacher),
        db     : AsyncSession = Depends(get_db)):
    """Top-of-dashboard counts."""
    c_res = await db.execute(
        select(func.count(Course.id))
        .where(Course.teacher_id == teacher.id))
    total_courses = c_res.scalar() or 0

    s_res = await db.execute(
        select(func.count(User.id))
        .where(User.role == "student"))
    total_students = s_res.scalar() or 0

    sess_res = await db.execute(
        select(func.count(Session.id)))
    total_sessions = sess_res.scalar() or 0

    avg_res = await db.execute(
        select(func.avg(Session.avg_engagement)))
    class_avg = round((avg_res.scalar() or 0) * 100, 1)

    return {
        "total_courses"       : total_courses,
        "total_students"      : total_students,
        "total_sessions"      : total_sessions,
        "class_avg_engagement": class_avg,
    }

@router.get("/teacher/courses")
async def teacher_get_courses(
        teacher: User = Depends(get_current_teacher),
        db     : AsyncSession = Depends(get_db)):
    """Same as GET /courses but in the shape TeacherDashboard.jsx expects."""
    result  = await db.execute(
        select(Course).where(Course.teacher_id == teacher.id)
        .order_by(desc(Course.created_at)))
    courses = result.scalars().all()
    return [{
        "id"         : c.id,
        "name"       : c.name,
        "description": c.description,
        "code"       : f"CRS-{c.id:04d}",
        "students"   : [],
    } for c in courses]

@router.post("/teacher/courses")
async def teacher_create_course(
        data   : CourseCreate,
        teacher: User = Depends(get_current_teacher),
        db     : AsyncSession = Depends(get_db)):
    course = Course(name=data.name,
                    description=data.description,
                    teacher_id=teacher.id)
    db.add(course)
    await db.commit()
    await db.refresh(course)
    return {"id": course.id, "name": course.name,
            "description": course.description}

@router.delete("/teacher/courses/{course_id}")
async def teacher_delete_course(
        course_id: int,
        teacher  : User = Depends(get_current_teacher),
        db       : AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Course).where(
            Course.id == course_id,
            Course.teacher_id == teacher.id))
    course = result.scalar_one_or_none()
    if not course:
        raise HTTPException(404, "Course not found")
    await db.delete(course)
    await db.commit()
    return {"status": "deleted"}

@router.get("/teacher/students")
async def teacher_get_students(
        teacher: User = Depends(get_current_teacher),
        db     : AsyncSession = Depends(get_db)):
    """
    List all students plus aggregated stats across every saved session.
    Students with sessions come first (most recent activity at top),
    students who have never recorded follow at the end.
    """
    stu_res  = await db.execute(
        select(User).where(User.role == "student"))
    students = stu_res.scalars().all()

    output = []
    for s in students:
        sess_res = await db.execute(
            select(Session)
            .where(Session.user_id == s.id)
            .order_by(desc(Session.started_at)))
        sessions = sess_res.scalars().all()

        if not sessions:
            output.append({
                "id"              : s.id,
                "name"            : s.name,
                "email"           : s.email,
                "total_sessions"  : 0,
                "avg_engagement"  : 0,
                "dominant_emotion": None,
                "last_active"     : None,
            })
            continue

        total_sessions = len(sessions)
        avg_eng = sum(x.avg_engagement or 0
                      for x in sessions) / total_sessions

        emo_counts = {}
        for x in sessions:
            if x.dominant_emotion:
                emo_counts[x.dominant_emotion] = \
                    emo_counts.get(x.dominant_emotion, 0) + 1
        dominant = (max(emo_counts, key=emo_counts.get)
                    if emo_counts else None)

        output.append({
            "id"              : s.id,
            "name"            : s.name,
            "email"           : s.email,
            "total_sessions"  : total_sessions,
            "avg_engagement"  : round(avg_eng * 100, 1),
            "dominant_emotion": dominant,
            "last_active"     : (sessions[0].started_at.isoformat()
                                 if sessions[0].started_at else None),
        })

    output.sort(key=lambda x: (
        0 if x["total_sessions"] > 0 else 1,
        -(x["total_sessions"]),
    ))
    return output

@router.get("/teacher/students/{student_id}/profile")
async def teacher_student_profile(
        student_id: int,
        teacher   : User = Depends(get_current_teacher),
        db        : AsyncSession = Depends(get_db)):
    """Aggregated profile for one student — used by StudentAnalyticsModal."""
    stu_res = await db.execute(
        select(User).where(
            User.id == student_id, User.role == "student"))
    student = stu_res.scalar_one_or_none()
    if not student:
        raise HTTPException(404, "Student not found")

    sess_res = await db.execute(
        select(Session)
        .where(Session.user_id == student_id)
        .order_by(desc(Session.started_at)))
    sessions = sess_res.scalars().all()

    if not sessions:
        return {
            "id"              : student.id,
            "name"            : student.name,
            "email"           : student.email,
            "total_sessions"  : 0,
            "avg_engagement"  : 0,
            "dominant_emotion": None,
            "distribution"    : {},
            "engagement_trend": [],
        }

    total_sessions = len(sessions)
    avg_eng = sum(s.avg_engagement or 0
                  for s in sessions) / total_sessions

    # Aggregate distribution percentages across all sessions
    combined = {}
    for s in sessions:
        if s.distribution:
            for emo, pct in s.distribution.items():
                combined[emo] = combined.get(emo, 0) + float(pct)
    if combined:
        total = sum(combined.values())
        if total > 0:
            combined = {e: round(v / total * 100, 1)
                        for e, v in combined.items()}

    emo_counts = {}
    for s in sessions:
        if s.dominant_emotion:
            emo_counts[s.dominant_emotion] = \
                emo_counts.get(s.dominant_emotion, 0) + 1
    dominant = (max(emo_counts, key=emo_counts.get)
                if emo_counts else None)

    # Engagement trend: one data point per session (oldest → newest),
    # capped at 50 points so the sparkline stays readable.
    trend = [{
        "time" : s.started_at.isoformat() if s.started_at else "",
        "score": s.avg_engagement or 0,
    } for s in reversed(sessions)][-50:]

    return {
        "id"              : student.id,
        "name"            : student.name,
        "email"           : student.email,
        "total_sessions"  : total_sessions,
        "avg_engagement"  : round(avg_eng * 100, 1),
        "dominant_emotion": dominant,
        "distribution"    : combined,
        "engagement_trend": trend,
    }

@router.get("/teacher/students/{student_id}/sessions")
async def teacher_student_sessions(
        student_id: int,
        teacher   : User = Depends(get_current_teacher),
        db        : AsyncSession = Depends(get_db)):
    """All of a given student's sessions, most recent first."""
    sess_res = await db.execute(
        select(Session)
        .where(Session.user_id == student_id)
        .order_by(desc(Session.started_at))
        .limit(50))
    sessions = sess_res.scalars().all()
    return [{
        "id"              : s.id,
        "started_at"      : s.started_at,
        "ended_at"        : s.ended_at,
        "avg_engagement"  : round((s.avg_engagement or 0) * 100, 1),
        "dominant_emotion": s.dominant_emotion,
        "total_detections": s.total_detections,
        "distribution"    : s.distribution,
    } for s in sessions]

@router.get("/teacher/sessions/{session_id}/report")
async def teacher_session_report(
        session_id: int,
        teacher   : User = Depends(get_current_teacher),
        db        : AsyncSession = Depends(get_db)):
    """HTML report for any student's session — opens in browser/downloads."""
    sess_res = await db.execute(
        select(Session).where(Session.id == session_id))
    session = sess_res.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Session not found")

    stu_res = await db.execute(
        select(User).where(User.id == session.user_id))
    student = stu_res.scalar_one_or_none()
    if not student:
        raise HTTPException(404, "Student not found")

    logs_res = await db.execute(
        select(EmotionLog)
        .where(EmotionLog.session_id == session_id)
        .order_by(EmotionLog.timestamp))
    logs = logs_res.scalars().all()

    course_name = None
    if session.lecture_id:
        lec_res = await db.execute(
            select(Lecture, Course)
            .join(Course, Lecture.course_id == Course.id)
            .where(Lecture.id == session.lecture_id))
        row = lec_res.one_or_none()
        if row:
            course_name = row.Course.name

    return _session_html_report(session, logs, student, course_name)

@router.get("/teacher/sessions/{session_id}/csv")
async def teacher_session_csv(
        session_id: int,
        teacher   : User = Depends(get_current_teacher),
        db        : AsyncSession = Depends(get_db)):
    """CSV export for any student's session."""
    sess_res = await db.execute(
        select(Session).where(Session.id == session_id))
    session = sess_res.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Session not found")

    stu_res = await db.execute(
        select(User).where(User.id == session.user_id))
    student = stu_res.scalar_one_or_none()

    logs_res = await db.execute(
        select(EmotionLog)
        .where(EmotionLog.session_id == session_id)
        .order_by(EmotionLog.timestamp))
    logs = logs_res.scalars().all()

    return _session_csv_response(
        session, logs,
        student.name if student else "student")

# ══════════════════════════════════════════════════════════════════════════
# Admin (unchanged)
# ══════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════
# ENROLLMENT — Students enroll in courses
# ══════════════════════════════════════════════════════════════════════════

@router.post("/courses/{course_id}/enroll")
async def enroll_in_course(
        course_id: int,
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    """Student enrolls in a course."""
    course = await db.get(Course, course_id)
    if not course:
        raise HTTPException(404, "Course not found")
    existing = await db.execute(
        select(Enrollment).where(
            Enrollment.student_id == user.id,
            Enrollment.course_id == course_id))
    if existing.scalar_one_or_none():
        raise HTTPException(400, "Already enrolled")
    enrollment = Enrollment(student_id=user.id, course_id=course_id)
    db.add(enrollment)
    await db.commit()
    return {"success": True, "message": f"Enrolled in {course.name}"}


@router.delete("/courses/{course_id}/enroll")
async def unenroll_from_course(
        course_id: int,
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    enrollment = await db.execute(
        select(Enrollment).where(
            Enrollment.student_id == user.id,
            Enrollment.course_id == course_id))
    e = enrollment.scalar_one_or_none()
    if not e:
        raise HTTPException(404, "Not enrolled")
    await db.delete(e)
    await db.commit()
    return {"success": True}


@router.get("/students/courses")
async def student_get_courses(
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    """Get courses the student is enrolled in."""
    result = await db.execute(
        select(Enrollment, Course)
        .join(Course, Enrollment.course_id == Course.id)
        .where(Enrollment.student_id == user.id))
    rows = result.all()
    courses = []
    for enrollment, course in rows:
        teacher = await db.get(User, course.teacher_id)
        courses.append({
            "id": course.id,
            "name": course.name,
            "description": course.description,
            "teacher_name": teacher.name if teacher else "Unknown",
            "enrolled_at": enrollment.enrolled_at.isoformat()
                           if enrollment.enrolled_at else None,
        })
    return {"courses": courses}


@router.get("/courses/{course_id}/students")
async def get_course_students(
        course_id: int,
        user: User = Depends(get_current_teacher),
        db: AsyncSession = Depends(get_db)):
    """Teacher gets enrolled students for a course."""
    result = await db.execute(
        select(Enrollment, User)
        .join(User, Enrollment.student_id == User.id)
        .where(Enrollment.course_id == course_id))
    return {"students": [
        {"id": u.id, "name": u.name, "email": u.email,
         "enrolled_at": e.enrolled_at.isoformat() if e.enrolled_at else None}
        for e, u in result.all()
    ]}


@router.get("/courses/available")
async def get_available_courses(
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    """List all courses (for enrollment page)."""
    result = await db.execute(select(Course))
    courses = result.scalars().all()
    # Check which ones student is enrolled in
    enrolled = await db.execute(
        select(Enrollment.course_id).where(
            Enrollment.student_id == user.id))
    enrolled_ids = {r[0] for r in enrolled.all()}
    out = []
    for c in courses:
        teacher = await db.get(User, c.teacher_id)
        out.append({
            "id": c.id, "name": c.name,
            "description": c.description,
            "teacher_name": teacher.name if teacher else "Unknown",
            "enrolled": c.id in enrolled_ids,
        })
    return {"courses": out}


# ══════════════════════════════════════════════════════════════════════════
# EXAMS — Teacher creates, student takes
# ══════════════════════════════════════════════════════════════════════════

class ExamCreate(BaseModel):
    course_id  : int
    title      : str
    description: str = ""
    questions  : list       # [{q, o, a}, ...]
    time_limit : int = 600  # seconds
    is_proctored: bool = True
    due_date   : Optional[str] = None


@router.post("/exams")
async def create_exam(
        data: ExamCreate,
        user: User = Depends(get_current_teacher),
        db: AsyncSession = Depends(get_db)):
    """Teacher creates an exam for a course."""
    course = await db.get(Course, data.course_id)
    if not course or course.teacher_id != user.id:
        raise HTTPException(403, "Not your course")
    due = None
    if data.due_date:
        try:
            due = datetime.fromisoformat(
                data.due_date.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            pass
    exam = Exam(
        course_id=data.course_id, title=data.title,
        description=data.description, questions=data.questions,
        time_limit=data.time_limit, is_proctored=data.is_proctored,
        due_date=due, created_by=user.id)
    db.add(exam)
    await db.flush()
    # Notify all enrolled students
    enrollments = await db.execute(
        select(Enrollment.student_id).where(
            Enrollment.course_id == data.course_id))
    for (sid,) in enrollments.all():
        db.add(Notification(
            user_id=sid, type="exam",
            title=f"New Exam: {data.title}",
            message=f"A new exam has been posted for {course.name}.",
            link=f"/exam?id={exam.id}"))
    await db.commit()
    await db.refresh(exam)
    return {"success": True, "exam_id": exam.id,
            "message": f"Exam '{data.title}' created, "
                       f"{enrollments.rowcount if hasattr(enrollments, 'rowcount') else 'all'} students notified"}


@router.get("/courses/{course_id}/exams")
async def get_course_exams(
        course_id: int,
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    """List exams for a course (works for both student and teacher)."""
    result = await db.execute(
        select(Exam).where(Exam.course_id == course_id)
        .order_by(desc(Exam.created_at)))
    exams = result.scalars().all()
    out = []
    for e in exams:
        # Check if student has submitted
        sub = None
        if user.role == "student":
            sub_res = await db.execute(
                select(ExamSubmission).where(
                    ExamSubmission.exam_id == e.id,
                    ExamSubmission.student_id == user.id))
            sub_obj = sub_res.scalar_one_or_none()
            if sub_obj:
                sub = {"score": sub_obj.score,
                       "focus_score": sub_obj.focus_score,
                       "submitted_at": sub_obj.submitted_at.isoformat()
                                       if sub_obj.submitted_at else None}
        out.append({
            "id": e.id, "title": e.title,
            "description": e.description,
            "question_count": len(e.questions) if e.questions else 0,
            "time_limit": e.time_limit,
            "is_proctored": e.is_proctored,
            "due_date": e.due_date.isoformat() if e.due_date else None,
            "created_at": e.created_at.isoformat() if e.created_at else None,
            "submission": sub,
        })
    return {"exams": out}


@router.get("/exams/{exam_id}")
async def get_exam(
        exam_id: int,
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    """Load exam for taking (student) or reviewing (teacher).
    For students: questions without correct answers.
    For teachers: full questions with answers."""
    exam = await db.get(Exam, exam_id)
    if not exam:
        raise HTTPException(404, "Exam not found")
    course = await db.get(Course, exam.course_id)
    # Strip answers for students
    qs = exam.questions or []
    if user.role == "student":
        qs = [{"q": q["q"], "o": q["o"]} for q in qs]
    return {
        "id": exam.id, "title": exam.title,
        "description": exam.description,
        "course_name": course.name if course else "",
        "questions": qs,
        "time_limit": exam.time_limit,
        "is_proctored": exam.is_proctored,
        "due_date": exam.due_date.isoformat() if exam.due_date else None,
    }


@router.delete("/exams/{exam_id}")
async def delete_exam(
        exam_id: int,
        user: User = Depends(get_current_teacher),
        db: AsyncSession = Depends(get_db)):
    exam = await db.get(Exam, exam_id)
    if not exam:
        raise HTTPException(404, "Exam not found")
    course = await db.get(Course, exam.course_id)
    if not course or course.teacher_id != user.id:
        raise HTTPException(403, "Not your exam")
    await db.delete(exam)
    await db.commit()
    return {"success": True}


# ══════════════════════════════════════════════════════════════════════════
# EXAM SUBMISSIONS — Student submits, teacher reviews
# ══════════════════════════════════════════════════════════════════════════

class SubmitExam(BaseModel):
    exam_id       : int
    answers       : dict           # {0: 1, 1: 2, ...}
    focus_score   : float = 100
    focus_log     : list = []
    alerts        : list = []
    answer_timing : dict = {}
    gap_warnings  : list = []
    duration_sec  : int = 0


@router.post("/exams/submit")
async def submit_exam(
        data: SubmitExam,
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    """Student submits exam answers + focus data."""
    exam = await db.get(Exam, data.exam_id)
    if not exam:
        raise HTTPException(404, "Exam not found")
    # Check if already submitted
    existing = await db.execute(
        select(ExamSubmission).where(
            ExamSubmission.exam_id == data.exam_id,
            ExamSubmission.student_id == user.id))
    if existing.scalar_one_or_none():
        raise HTTPException(400, "Already submitted")
    # Calculate score
    qs = exam.questions or []
    correct = 0
    for idx_str, ans in data.answers.items():
        idx = int(idx_str)
        if 0 <= idx < len(qs) and ans == qs[idx].get("a"):
            correct += 1
    score = round(correct / max(1, len(qs)) * 100, 1)
    sub = ExamSubmission(
        exam_id=data.exam_id, student_id=user.id,
        answers=data.answers, score=score,
        total_correct=correct, total_questions=len(qs),
        focus_score=data.focus_score, focus_log=data.focus_log,
        alerts=data.alerts, answer_timing=data.answer_timing,
        gap_warnings=data.gap_warnings, duration_sec=data.duration_sec)
    db.add(sub)
    await db.commit()
    await db.refresh(sub)
    return {
        "success": True, "submission_id": sub.id,
        "score": score, "correct": correct, "total": len(qs),
        "focus_score": data.focus_score,
    }


@router.get("/students/exams")
async def student_get_exams(
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    """Student gets all pending + completed exams across enrolled courses."""
    enrolled = await db.execute(
        select(Enrollment.course_id).where(
            Enrollment.student_id == user.id))
    course_ids = [r[0] for r in enrolled.all()]
    if not course_ids:
        return {"exams": []}
    result = await db.execute(
        select(Exam, Course.name.label("course_name"))
        .join(Course, Exam.course_id == Course.id)
        .where(Exam.course_id.in_(course_ids))
        .order_by(desc(Exam.created_at)))
    exams = []
    for exam, course_name in result.all():
        sub_res = await db.execute(
            select(ExamSubmission).where(
                ExamSubmission.exam_id == exam.id,
                ExamSubmission.student_id == user.id))
        sub = sub_res.scalar_one_or_none()
        exams.append({
            "id": exam.id, "title": exam.title,
            "course_name": course_name,
            "course_id": exam.course_id,
            "question_count": len(exam.questions) if exam.questions else 0,
            "time_limit": exam.time_limit,
            "is_proctored": exam.is_proctored,
            "due_date": exam.due_date.isoformat() if exam.due_date else None,
            "submitted": sub is not None,
            "score": sub.score if sub else None,
            "focus_score": sub.focus_score if sub else None,
            "submitted_at": sub.submitted_at.isoformat()
                           if sub and sub.submitted_at else None,
        })
    return {"exams": exams}


@router.get("/students/submissions/{submission_id}")
async def student_get_submission(
        submission_id: int,
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    """Student views their own submission details."""
    sub = await db.get(ExamSubmission, submission_id)
    if not sub or sub.student_id != user.id:
        raise HTTPException(404, "Submission not found")
    exam = await db.get(Exam, sub.exam_id)
    return {
        "submission": {
            "id": sub.id, "score": sub.score,
            "total_correct": sub.total_correct,
            "total_questions": sub.total_questions,
            "focus_score": sub.focus_score,
            "focus_log": sub.focus_log,
            "alerts": sub.alerts,
            "answer_timing": sub.answer_timing,
            "gap_warnings": sub.gap_warnings,
            "duration_sec": sub.duration_sec,
            "submitted_at": sub.submitted_at.isoformat()
                           if sub.submitted_at else None,
        },
        "exam": {
            "title": exam.title if exam else "",
            "questions": exam.questions if exam else [],
        },
        "answers": sub.answers,
    }


@router.get("/exams/{exam_id}/submissions")
async def teacher_get_submissions(
        exam_id: int,
        user: User = Depends(get_current_teacher),
        db: AsyncSession = Depends(get_db)):
    """Teacher views all submissions for an exam."""
    exam = await db.get(Exam, exam_id)
    if not exam:
        raise HTTPException(404, "Exam not found")
    result = await db.execute(
        select(ExamSubmission, User.name.label("student_name"),
               User.email.label("student_email"))
        .join(User, ExamSubmission.student_id == User.id)
        .where(ExamSubmission.exam_id == exam_id)
        .order_by(desc(ExamSubmission.submitted_at)))
    submissions = []
    for sub, sname, semail in result.all():
        submissions.append({
            "id": sub.id, "student_name": sname,
            "student_email": semail,
            "score": sub.score, "focus_score": sub.focus_score,
            "alerts_count": len(sub.alerts) if sub.alerts else 0,
            "gap_warnings_count": len(sub.gap_warnings)
                                 if sub.gap_warnings else 0,
            "duration_sec": sub.duration_sec,
            "submitted_at": sub.submitted_at.isoformat()
                           if sub.submitted_at else None,
        })
    avg_score = (round(sum(s["score"] for s in submissions)
                       / len(submissions), 1)
                 if submissions else 0)
    avg_focus = (round(sum(s["focus_score"] for s in submissions)
                       / len(submissions), 1)
                 if submissions else 0)
    return {
        "exam_title": exam.title,
        "total_submissions": len(submissions),
        "avg_score": avg_score, "avg_focus": avg_focus,
        "submissions": submissions,
    }


@router.get("/teacher/submissions/{submission_id}")
async def teacher_view_submission(
        submission_id: int,
        user: User = Depends(get_current_teacher),
        db: AsyncSession = Depends(get_db)):
    """Teacher views full submission detail including focus report."""
    sub = await db.get(ExamSubmission, submission_id)
    if not sub:
        raise HTTPException(404, "Submission not found")
    exam = await db.get(Exam, sub.exam_id)
    student = await db.get(User, sub.student_id)
    return {
        "student": {"name": student.name if student else "",
                    "email": student.email if student else ""},
        "exam": {"title": exam.title if exam else "",
                 "questions": exam.questions if exam else []},
        "submission": {
            "answers": sub.answers, "score": sub.score,
            "total_correct": sub.total_correct,
            "total_questions": sub.total_questions,
            "focus_score": sub.focus_score,
            "focus_log": sub.focus_log,
            "alerts": sub.alerts,
            "answer_timing": sub.answer_timing,
            "gap_warnings": sub.gap_warnings,
            "duration_sec": sub.duration_sec,
            "submitted_at": sub.submitted_at.isoformat()
                           if sub.submitted_at else None,
        },
    }


# ══════════════════════════════════════════════════════════════════════════
# NOTIFICATIONS
# ══════════════════════════════════════════════════════════════════════════

@router.get("/notifications")
async def get_notifications(
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Notification)
        .where(Notification.user_id == user.id)
        .order_by(desc(Notification.created_at))
        .limit(50))
    notifs = result.scalars().all()
    unread = sum(1 for n in notifs if not n.is_read)
    return {
        "unread": unread,
        "notifications": [{
            "id": n.id, "type": n.type, "title": n.title,
            "message": n.message, "link": n.link,
            "is_read": n.is_read,
            "created_at": n.created_at.isoformat() if n.created_at else None,
        } for n in notifs],
    }


@router.post("/notifications/{notif_id}/read")
async def mark_notification_read(
        notif_id: int,
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    notif = await db.get(Notification, notif_id)
    if not notif or notif.user_id != user.id:
        raise HTTPException(404, "Notification not found")
    notif.is_read = True
    await db.commit()
    return {"success": True}


@router.post("/notifications/read-all")
async def mark_all_read(
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Notification).where(
            Notification.user_id == user.id,
            Notification.is_read == False))
    for n in result.scalars().all():
        n.is_read = True
    await db.commit()
    return {"success": True}


# ══════════════════════════════════════════════════════════════════════════
# ADMIN — System management endpoints
# ══════════════════════════════════════════════════════════════════════════

@router.get("/admin/users")
async def admin_get_users(
        role: Optional[str] = None,
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    """List all users with optional role filter."""
    q = select(User)
    if role:
        q = q.where(User.role == role)
    result = await db.execute(q.order_by(desc(User.created_at)))
    users = result.scalars().all()
    out = []
    for u in users:
        sess_res = await db.execute(
            select(func.count(Session.id)).where(Session.user_id == u.id))
        sess_count = sess_res.scalar() or 0
        out.append({
            "id": u.id, "name": u.name, "email": u.email,
            "role": u.role, "is_active": u.is_active,
            "sessions": sess_count,
            "created_at": u.created_at.isoformat() if u.created_at else None,
        })
    return {"users": out, "total": len(out)}


class RoleUpdate(BaseModel):
    role: str  # student, teacher, admin


@router.put("/admin/users/{user_id}/role")
async def admin_change_role(
        user_id: int, data: RoleUpdate,
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    if data.role not in ("student", "teacher", "admin"):
        raise HTTPException(400, "Invalid role")
    user.role = data.role
    await db.commit()
    return {"success": True, "message": f"{user.name} is now {data.role}"}


@router.put("/admin/users/{user_id}/status")
async def admin_toggle_status(
        user_id: int,
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    user.is_active = not user.is_active
    await db.commit()
    return {"success": True, "is_active": user.is_active,
            "message": f"{user.name} {'activated' if user.is_active else 'deactivated'}"}


@router.get("/admin/stats")
async def admin_platform_stats(
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    """Platform-wide analytics."""
    total_users = (await db.execute(select(func.count(User.id)))).scalar() or 0
    total_students = (await db.execute(
        select(func.count(User.id)).where(User.role == "student"))).scalar() or 0
    total_teachers = (await db.execute(
        select(func.count(User.id)).where(User.role == "teacher"))).scalar() or 0
    total_courses = (await db.execute(select(func.count(Course.id)))).scalar() or 0
    total_sessions = (await db.execute(select(func.count(Session.id)))).scalar() or 0
    total_exams = (await db.execute(select(func.count(Exam.id)))).scalar() or 0
    total_submissions = (await db.execute(
        select(func.count(ExamSubmission.id)))).scalar() or 0
    total_enrollments = (await db.execute(
        select(func.count(Enrollment.id)))).scalar() or 0

    # Avg engagement across all sessions
    avg_eng = (await db.execute(
        select(func.avg(Session.avg_engagement)))).scalar() or 0

    # Avg exam score + focus
    avg_exam_score = (await db.execute(
        select(func.avg(ExamSubmission.score)))).scalar() or 0
    avg_focus = (await db.execute(
        select(func.avg(ExamSubmission.focus_score)))).scalar() or 0

    # Recent submissions
    recent_subs = await db.execute(
        select(ExamSubmission, User.name.label("sname"), Exam.title.label("etitle"))
        .join(User, ExamSubmission.student_id == User.id)
        .join(Exam, ExamSubmission.exam_id == Exam.id)
        .order_by(desc(ExamSubmission.submitted_at)).limit(10))

    return {
        "users": {"total": total_users, "students": total_students,
                  "teachers": total_teachers},
        "courses": total_courses,
        "enrollments": total_enrollments,
        "sessions": total_sessions,
        "avg_engagement": round(avg_eng * 100, 1) if avg_eng and avg_eng < 1 else round(avg_eng or 0, 1),
        "exams": {"total": total_exams, "submissions": total_submissions,
                  "avg_score": round(avg_exam_score or 0, 1),
                  "avg_focus": round(avg_focus or 0, 1)},
        "recent_submissions": [{
            "student": r.sname, "exam": r.etitle,
            "score": r.ExamSubmission.score,
            "focus": r.ExamSubmission.focus_score,
            "submitted": r.ExamSubmission.submitted_at.isoformat()
                         if r.ExamSubmission.submitted_at else None,
        } for r in recent_subs.all()],
    }


@router.get("/admin/health")
async def admin_system_health(admin: User = Depends(get_current_admin)):
    """Check health of all external services."""
    import httpx
    health = {}
    hf_token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    models = {
        "EfficientNet-B2": None,  # local model, always OK if server is up
        "Whisper (HF)": "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3",
        "RoBERTa (HF)": "https://router.huggingface.co/hf-inference/models/j-hartmann/emotion-english-distilroberta-base",
        "wav2vec2 (HF)": "https://router.huggingface.co/hf-inference/models/r-f/wav2vec-english-speech-emotion-recognition",
        "VIT Face (HF)": "https://router.huggingface.co/hf-inference/models/trpakov/vit-face-expression",
    }

    health["EfficientNet-B2"] = {"status": "online", "type": "local"}

    async with httpx.AsyncClient(timeout=10) as client:
        for name, url in models.items():
            if url is None:
                continue
            try:
                resp = await client.get(url, headers=headers)
                health[name] = {
                    "status": "online" if resp.status_code < 500 else "error",
                    "code": resp.status_code, "type": "huggingface",
                }
            except Exception as e:
                health[name] = {"status": "offline", "error": str(e), "type": "huggingface"}

    # Database check
    try:
        await db_check()
        health["PostgreSQL"] = {"status": "online", "type": "database"}
    except Exception:
        health["PostgreSQL"] = {"status": "error", "type": "database"}

    all_online = all(v["status"] == "online" for v in health.values())
    return {"overall": "healthy" if all_online else "degraded", "services": health}


async def db_check():
    from models_db import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        await db.execute(select(func.count(User.id)))


@router.get("/admin/exam-integrity")
async def admin_exam_integrity(
        threshold: float = 50.0,
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    """Flag exam submissions with focus score below threshold."""
    result = await db.execute(
        select(ExamSubmission, User.name.label("sname"),
               User.email.label("semail"), Exam.title.label("etitle"))
        .join(User, ExamSubmission.student_id == User.id)
        .join(Exam, ExamSubmission.exam_id == Exam.id)
        .where(ExamSubmission.focus_score < threshold)
        .order_by(ExamSubmission.focus_score))

    flagged = []
    for r in result.all():
        flagged.append({
            "submission_id": r.ExamSubmission.id,
            "student": r.sname, "email": r.semail,
            "exam": r.etitle,
            "score": r.ExamSubmission.score,
            "focus_score": r.ExamSubmission.focus_score,
            "alerts_count": len(r.ExamSubmission.alerts or []),
            "gap_warnings": len(r.ExamSubmission.gap_warnings or []),
            "submitted": r.ExamSubmission.submitted_at.isoformat()
                         if r.ExamSubmission.submitted_at else None,
        })
    return {"threshold": threshold, "flagged_count": len(flagged),
            "flagged": flagged}


# ══════════════════════════════════════════════════════════════════════════
# STUDENT — Compare with Class (anonymized)
# ══════════════════════════════════════════════════════════════════════════

@router.get("/students/compare-class")
async def student_compare_class(
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    """
    Returns student's engagement & emotion stats alongside
    the anonymized class average for comparison.
    """
    # ── 1. Student's own stats ────────────────────────────────────────
    my_sess_res = await db.execute(
        select(Session).where(Session.user_id == current_user.id))
    my_sessions = my_sess_res.scalars().all()

    my_total = len(my_sessions)
    my_avg_eng = 0.0
    my_detections = 0
    my_dist = {}

    if my_sessions:
        my_avg_eng = sum(s.avg_engagement or 0 for s in my_sessions) / my_total
        my_detections = sum(s.total_detections or 0 for s in my_sessions)

        combined = {}
        for s in my_sessions:
            if s.distribution:
                for emo, pct in s.distribution.items():
                    combined[emo] = combined.get(emo, 0) + float(pct)
        if combined:
            t = sum(combined.values())
            if t > 0:
                my_dist = {e: round(v / t * 100, 1) for e, v in combined.items()}

    # ── 2. Class-wide stats (anonymized) ──────────────────────────────
    class_res = await db.execute(
        select(
            func.count(Session.id).label("total_sessions"),
            func.avg(Session.avg_engagement).label("avg_eng"),
            func.sum(Session.total_detections).label("total_det"),
        ))
    cr = class_res.one()

    all_sess_res = await db.execute(select(Session))
    all_sessions = all_sess_res.scalars().all()
    class_dist = {}
    combined_all = {}
    for s in all_sessions:
        if s.distribution:
            for emo, pct in s.distribution.items():
                combined_all[emo] = combined_all.get(emo, 0) + float(pct)
    if combined_all:
        t = sum(combined_all.values())
        if t > 0:
            class_dist = {e: round(v / t * 100, 1) for e, v in combined_all.items()}

    # ── 3. Count unique students ──────────────────────────────────────
    unique_res = await db.execute(
        select(func.count(func.distinct(Session.user_id))))
    total_students = unique_res.scalar() or 0

    # ── 4. Student rank by engagement ─────────────────────────────────
    rank = 1
    if my_sessions:
        stu_res = await db.execute(
            select(
                Session.user_id,
                func.avg(Session.avg_engagement).label("avg_eng"),
            ).group_by(Session.user_id))
        sorted_students = sorted(
            stu_res.all(), key=lambda x: x.avg_eng or 0, reverse=True)
        for idx, row in enumerate(sorted_students):
            if row.user_id == current_user.id:
                rank = idx + 1
                break

    # ── 5. Engagement trend (last 10 sessions) ───────────────────────
    my_trend = []
    for s in sorted(my_sessions,
                    key=lambda x: x.started_at or datetime.min)[-10:]:
        my_trend.append({
            "date": s.started_at.isoformat() if s.started_at else "",
            "engagement": round((s.avg_engagement or 0) * 100, 1),
        })

    return {
        "student": {
            "name": current_user.name,
            "total_sessions": my_total,
            "avg_engagement": round(my_avg_eng * 100, 1),
            "total_detections": my_detections,
            "distribution": my_dist,
            "rank": rank,
            "trend": my_trend,
        },
        "class_average": {
            "total_students": total_students,
            "total_sessions": cr.total_sessions or 0,
            "avg_engagement": round((cr.avg_eng or 0) * 100, 1),
            "total_detections": cr.total_det or 0,
            "distribution": class_dist,
        },
    }


# ══════════════════════════════════════════════════════════════════════════
# TEACHER — Live Class Engagement Heatmap
# ══════════════════════════════════════════════════════════════════════════

@router.get("/teacher/heatmap")
async def teacher_engagement_heatmap(
        course_id: Optional[int] = None,
        teacher: User = Depends(get_current_teacher),
        db: AsyncSession = Depends(get_db)):
    """
    Grid of students with latest emotion + engagement for a heatmap view.
    Optionally filter by course.
    """
    if course_id:
        enroll_res = await db.execute(
            select(User)
            .join(Enrollment, Enrollment.student_id == User.id)
            .where(Enrollment.course_id == course_id,
                   User.role == "student"))
    else:
        enroll_res = await db.execute(
            select(User).where(User.role == "student"))
    students = enroll_res.scalars().all()

    grid = []
    for s in students:
        sess_res = await db.execute(
            select(Session)
            .where(Session.user_id == s.id)
            .order_by(desc(Session.started_at))
            .limit(1))
        latest = sess_res.scalar_one_or_none()

        last_emotion = None
        last_engagement = 0
        last_active = None
        if latest:
            log_res = await db.execute(
                select(EmotionLog)
                .where(EmotionLog.session_id == latest.id)
                .order_by(desc(EmotionLog.timestamp))
                .limit(1))
            last_log = log_res.scalar_one_or_none()
            if last_log:
                last_emotion = last_log.emotion
                last_engagement = round(
                    (last_log.engagement_score or 0) * 100, 1)
            else:
                last_emotion = latest.dominant_emotion
                last_engagement = round(
                    (latest.avg_engagement or 0) * 100, 1)
            last_active = (latest.started_at.isoformat()
                           if latest.started_at else None)

        from datetime import timedelta
        status = "inactive"
        if latest and latest.started_at:
            if datetime.utcnow() - latest.started_at < timedelta(hours=1):
                status = "active"
            elif datetime.utcnow() - latest.started_at < timedelta(days=1):
                status = "recent"

        cnt_res = await db.execute(
            select(func.count(Session.id))
            .where(Session.user_id == s.id))

        grid.append({
            "id": s.id,
            "name": s.name,
            "email": s.email,
            "emotion": last_emotion,
            "engagement": last_engagement,
            "status": status,
            "last_active": last_active,
            "total_sessions": cnt_res.scalar() or 0,
        })

    active_count = len([g for g in grid if g["status"] == "active"])
    avg_eng = round(
        sum(g["engagement"] for g in grid) / max(1, len(grid)), 1
    ) if grid else 0

    emotion_counts = {}
    for g in grid:
        if g["emotion"]:
            emotion_counts[g["emotion"]] = \
                emotion_counts.get(g["emotion"], 0) + 1

    confused_count = sum(
        1 for g in grid
        if g["emotion"] in ("fear", "sadness", "disgust"))
    engaged_count = sum(1 for g in grid if g["engagement"] >= 65)

    return {
        "students": grid,
        "summary": {
            "total": len(grid),
            "active": active_count,
            "avg_engagement": avg_eng,
            "engaged": engaged_count,
            "confused": confused_count,
            "emotion_breakdown": emotion_counts,
        },
    }


# ══════════════════════════════════════════════════════════════════════════
# TEACHER — Attendance + Engagement per course
# ══════════════════════════════════════════════════════════════════════════

@router.get("/teacher/attendance/{course_id}")
async def teacher_attendance_engagement(
        course_id: int,
        teacher: User = Depends(get_current_teacher),
        db: AsyncSession = Depends(get_db)):
    """
    Per-course: each enrolled student's sessions, avg engagement,
    classified as engaged / passive / disengaged / absent.
    """
    course = await db.get(Course, course_id)
    if not course or course.teacher_id != teacher.id:
        raise HTTPException(403, "Not your course")

    enroll_res = await db.execute(
        select(User, Enrollment.enrolled_at)
        .join(Enrollment, Enrollment.student_id == User.id)
        .where(Enrollment.course_id == course_id))
    rows = enroll_res.all()

    attendance = []
    for user, enrolled_at in rows:
        sess_res = await db.execute(
            select(Session)
            .where(Session.user_id == user.id)
            .order_by(desc(Session.started_at)))
        sessions = sess_res.scalars().all()

        total_sessions = len(sessions)
        avg_eng = 0
        if sessions:
            avg_eng = round(
                sum(s.avg_engagement or 0 for s in sessions)
                / total_sessions * 100, 1)

        if total_sessions == 0:
            level = "absent"
        elif avg_eng >= 65:
            level = "engaged"
        elif avg_eng >= 40:
            level = "passive"
        else:
            level = "disengaged"

        last_session = sessions[0] if sessions else None
        attendance.append({
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "enrolled_at": (enrolled_at.isoformat()
                            if enrolled_at else None),
            "total_sessions": total_sessions,
            "avg_engagement": avg_eng,
            "level": level,
            "last_emotion": (last_session.dominant_emotion
                             if last_session else None),
            "last_session": (last_session.started_at.isoformat()
                             if last_session and last_session.started_at
                             else None),
        })

    level_order = {"engaged": 0, "passive": 1,
                   "disengaged": 2, "absent": 3}
    attendance.sort(key=lambda x: (
        level_order.get(x["level"], 4), -x["avg_engagement"]))

    summary = {
        "course_name": course.name,
        "total_enrolled": len(attendance),
        "engaged": len([a for a in attendance
                        if a["level"] == "engaged"]),
        "passive": len([a for a in attendance
                        if a["level"] == "passive"]),
        "disengaged": len([a for a in attendance
                           if a["level"] == "disengaged"]),
        "absent": len([a for a in attendance
                       if a["level"] == "absent"]),
        "class_avg_engagement": round(
            sum(a["avg_engagement"] for a in attendance)
            / max(1, len(attendance)), 1),
    }

    return {"attendance": attendance, "summary": summary}


# ══════════════════════════════════════════════════════════════════════════
# TEACHER — Student Progress Tracking (semester-long trend)
# ══════════════════════════════════════════════════════════════════════════

@router.get("/teacher/progress/{course_id}")
async def teacher_student_progress(
        course_id: int,
        teacher: User = Depends(get_current_teacher),
        db: AsyncSession = Depends(get_db)):
    """
    Engagement trend for ALL enrolled students in a course.
    Each student gets a list of (date, engagement) data-points
    plus an improvement score comparing first-half vs second-half.
    """
    course = await db.get(Course, course_id)
    if not course or course.teacher_id != teacher.id:
        raise HTTPException(403, "Not your course")

    enroll_res = await db.execute(
        select(User)
        .join(Enrollment, Enrollment.student_id == User.id)
        .where(Enrollment.course_id == course_id))
    students = enroll_res.scalars().all()

    progress = []
    for s in students:
        sess_res = await db.execute(
            select(Session)
            .where(Session.user_id == s.id)
            .order_by(Session.started_at))
        sessions = sess_res.scalars().all()

        trend = []
        for sess in sessions[-20:]:
            trend.append({
                "date": (sess.started_at.isoformat()
                         if sess.started_at else ""),
                "engagement": round(
                    (sess.avg_engagement or 0) * 100, 1),
                "emotion": sess.dominant_emotion,
            })

        improvement = 0
        if len(trend) >= 2:
            half = len(trend) // 2
            first_half = trend[:half]
            second_half = trend[half:]
            avg_first = sum(
                t["engagement"] for t in first_half) / len(first_half)
            avg_second = sum(
                t["engagement"] for t in second_half) / len(second_half)
            improvement = round(avg_second - avg_first, 1)

        avg_eng = round(
            sum(t["engagement"] for t in trend) / max(1, len(trend)), 1
        ) if trend else 0

        progress.append({
            "id": s.id,
            "name": s.name,
            "email": s.email,
            "total_sessions": len(sessions),
            "avg_engagement": avg_eng,
            "improvement": improvement,
            "trend": trend,
        })

    progress.sort(key=lambda x: -x["improvement"])

    return {
        "course_name": course.name,
        "students": progress,
    }


# ══════════════════════════════════════════════════════════════════════════
# ADMIN — 1. Consent Management
# ══════════════════════════════════════════════════════════════════════════

@router.get("/admin/consents")
async def admin_get_consents(
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ConsentRecord, User.name.label("sname"), User.email.label("semail"))
        .join(User, ConsentRecord.student_id == User.id)
        .order_by(desc(ConsentRecord.created_at)))
    return {"consents": [{
        "id": r.ConsentRecord.id,
        "student_id": r.ConsentRecord.student_id,
        "student_name": r.sname,
        "student_email": r.semail,
        "consent_type": r.ConsentRecord.consent_type,
        "granted": r.ConsentRecord.granted,
        "granted_at": r.ConsentRecord.granted_at.isoformat() if r.ConsentRecord.granted_at else None,
        "revoked_at": r.ConsentRecord.revoked_at.isoformat() if r.ConsentRecord.revoked_at else None,
    } for r in result.all()]}


@router.get("/consent/my")
async def get_my_consent(
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ConsentRecord).where(ConsentRecord.student_id == user.id))
    records = result.scalars().all()
    return {"consents": [{
        "id": r.id, "type": r.consent_type,
        "granted": r.granted,
        "granted_at": r.granted_at.isoformat() if r.granted_at else None,
    } for r in records]}


class ConsentUpdate(BaseModel):
    consent_type: str   # emotion_tracking, data_storage, video_recording
    granted: bool


@router.post("/consent/update")
async def update_consent(
        data: ConsentUpdate, request: Request,
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ConsentRecord).where(
            ConsentRecord.student_id == user.id,
            ConsentRecord.consent_type == data.consent_type))
    record = result.scalar_one_or_none()
    ip = request.client.host if request.client else None
    if record:
        record.granted = data.granted
        if data.granted:
            record.granted_at = datetime.utcnow()
            record.revoked_at = None
        else:
            record.revoked_at = datetime.utcnow()
        record.ip_address = ip
    else:
        record = ConsentRecord(
            student_id=user.id, consent_type=data.consent_type,
            granted=data.granted, ip_address=ip,
            granted_at=datetime.utcnow() if data.granted else None)
        db.add(record)
    await db.commit()
    return {"success": True, "granted": data.granted}


# ══════════════════════════════════════════════════════════════════════════
# ADMIN — 2. Data Retention Policies
# ══════════════════════════════════════════════════════════════════════════

@router.post("/admin/retention/archive")
async def admin_archive_old_data(
        days_old: int = 90,
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    """Archive (delete) emotion logs older than X days. Sessions kept."""
    cutoff = datetime.utcnow() - timedelta(days=days_old)
    result = await db.execute(
        select(func.count(EmotionLog.id))
        .where(EmotionLog.timestamp < cutoff))
    count = result.scalar() or 0
    if count > 0:
        await db.execute(
            EmotionLog.__table__.delete().where(EmotionLog.timestamp < cutoff))
        await db.commit()
    db.add(AuditLog(user_id=admin.id, action="data_archive",
                    details={"days_old": days_old, "logs_deleted": count}))
    await db.commit()
    return {"success": True, "logs_archived": count, "cutoff_date": cutoff.isoformat()}


@router.post("/admin/retention/delete-student")
async def admin_delete_student_data(
        student_id: int,
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    """Delete ALL emotion data for a student (right to be forgotten)."""
    student = await db.get(User, student_id)
    if not student:
        raise HTTPException(404, "Student not found")
    # Delete emotion logs
    sessions_res = await db.execute(
        select(Session.id).where(Session.user_id == student_id))
    session_ids = [r[0] for r in sessions_res.all()]
    log_count = 0
    if session_ids:
        cnt = await db.execute(
            select(func.count(EmotionLog.id))
            .where(EmotionLog.session_id.in_(session_ids)))
        log_count = cnt.scalar() or 0
        await db.execute(
            EmotionLog.__table__.delete()
            .where(EmotionLog.session_id.in_(session_ids)))
    # Delete sessions
    sess_count = len(session_ids)
    if session_ids:
        await db.execute(
            Session.__table__.delete()
            .where(Session.user_id == student_id))
    # Delete consent records
    await db.execute(
        ConsentRecord.__table__.delete()
        .where(ConsentRecord.student_id == student_id))
    await db.commit()
    db.add(AuditLog(user_id=admin.id, action="student_data_delete",
                    target=f"user:{student_id}",
                    details={"sessions": sess_count, "logs": log_count}))
    await db.commit()
    return {"success": True, "student": student.name,
            "sessions_deleted": sess_count, "logs_deleted": log_count}


@router.get("/admin/retention/stats")
async def admin_retention_stats(
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    total_logs = (await db.execute(select(func.count(EmotionLog.id)))).scalar() or 0
    total_sessions = (await db.execute(select(func.count(Session.id)))).scalar() or 0
    old_logs_30 = (await db.execute(
        select(func.count(EmotionLog.id))
        .where(EmotionLog.timestamp < datetime.utcnow() - timedelta(days=30))
    )).scalar() or 0
    old_logs_90 = (await db.execute(
        select(func.count(EmotionLog.id))
        .where(EmotionLog.timestamp < datetime.utcnow() - timedelta(days=90))
    )).scalar() or 0
    return {
        "total_emotion_logs": total_logs,
        "total_sessions": total_sessions,
        "logs_older_30_days": old_logs_30,
        "logs_older_90_days": old_logs_90,
    }


# ══════════════════════════════════════════════════════════════════════════
# ADMIN — 3. Anonymization Controls (handled via consent — no extra table)
# ══════════════════════════════════════════════════════════════════════════

@router.get("/admin/anonymization/status")
async def admin_anonymization_status(
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    total = (await db.execute(
        select(func.count(User.id)).where(User.role == "student"))).scalar() or 0
    consented = (await db.execute(
        select(func.count(func.distinct(ConsentRecord.student_id)))
        .where(ConsentRecord.consent_type == "emotion_tracking",
               ConsentRecord.granted == True)
    )).scalar() or 0
    return {
        "total_students": total,
        "consented": consented,
        "not_consented": total - consented,
        "consent_rate": round(consented / max(1, total) * 100, 1),
    }


# ══════════════════════════════════════════════════════════════════════════
# ADMIN — 4. Camera/Stream Management (reads from main.py StreamManager)
# ══════════════════════════════════════════════════════════════════════════

@router.get("/admin/streams")
async def admin_get_streams(admin: User = Depends(get_current_admin)):
    """Proxy to the stream manager in main.py — admin-only access."""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("http://localhost:8000/api/streams")
            return r.json()
    except Exception:
        return {"streams": {}}


# ══════════════════════════════════════════════════════════════════════════
# ADMIN — 5. Model Performance Monitoring
# ══════════════════════════════════════════════════════════════════════════

@router.get("/admin/model-performance")
async def admin_model_performance(
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    """Aggregate stats about emotion detection quality."""
    total_logs = (await db.execute(select(func.count(EmotionLog.id)))).scalar() or 0
    avg_conf = (await db.execute(select(func.avg(EmotionLog.confidence)))).scalar() or 0
    # Emotion distribution across all logs
    emo_counts = {}
    result = await db.execute(
        select(EmotionLog.emotion, func.count(EmotionLog.id))
        .group_by(EmotionLog.emotion))
    for emo, cnt in result.all():
        emo_counts[emo] = cnt
    # Source breakdown
    source_res = await db.execute(
        select(EmotionLog.source, func.count(EmotionLog.id))
        .group_by(EmotionLog.source))
    sources = {s: c for s, c in source_res.all()}
    # High-confidence vs low-confidence
    high = (await db.execute(
        select(func.count(EmotionLog.id))
        .where(EmotionLog.confidence >= 0.7))).scalar() or 0
    low = (await db.execute(
        select(func.count(EmotionLog.id))
        .where(EmotionLog.confidence < 0.3))).scalar() or 0
    return {
        "total_detections": total_logs,
        "avg_confidence": round(avg_conf * 100, 1) if avg_conf < 1 else round(avg_conf, 1),
        "emotion_distribution": emo_counts,
        "source_breakdown": sources,
        "high_confidence_pct": round(high / max(1, total_logs) * 100, 1),
        "low_confidence_pct": round(low / max(1, total_logs) * 100, 1),
    }


# ══════════════════════════════════════════════════════════════════════════
# ADMIN — 6. System Health (enhanced — original kept)
# ══════════════════════════════════════════════════════════════════════════

@router.get("/admin/system-health")
async def admin_system_health_enhanced(
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    """Enhanced system health with DB stats."""
    import psutil, os
    health = {}
    # Database
    try:
        cnt = (await db.execute(select(func.count(User.id)))).scalar()
        health["database"] = {"status": "online", "users": cnt}
    except Exception as e:
        health["database"] = {"status": "error", "error": str(e)}
    # System resources
    try:
        health["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
        }
    except Exception:
        health["system"] = {"cpu_percent": 0, "memory_percent": 0, "disk_percent": 0}
    # Uptime
    try:
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.readline().split()[0])
            health["uptime_hours"] = round(uptime_seconds / 3600, 1)
    except Exception:
        health["uptime_hours"] = 0
    # Active sessions (last hour)
    one_hour = datetime.utcnow() - timedelta(hours=1)
    active = (await db.execute(
        select(func.count(Session.id))
        .where(Session.started_at >= one_hour))).scalar() or 0
    health["active_sessions_1h"] = active
    return health


# ══════════════════════════════════════════════════════════════════════════
# ADMIN — 7. User Activity / Audit Logs
# ══════════════════════════════════════════════════════════════════════════

@router.get("/admin/audit-logs")
async def admin_get_audit_logs(
        limit: int = 50,
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(AuditLog, User.name.label("uname"))
        .join(User, AuditLog.user_id == User.id, isouter=True)
        .order_by(desc(AuditLog.created_at))
        .limit(limit))
    return {"logs": [{
        "id": r.AuditLog.id,
        "user": r.uname or "System",
        "action": r.AuditLog.action,
        "target": r.AuditLog.target,
        "details": r.AuditLog.details,
        "ip": r.AuditLog.ip_address,
        "created_at": r.AuditLog.created_at.isoformat() if r.AuditLog.created_at else None,
    } for r in result.all()]}


# Helper to log audit events from other routes
async def log_audit(db, user_id, action, target=None, details=None, ip=None):
    db.add(AuditLog(user_id=user_id, action=action,
                    target=target, details=details or {}, ip_address=ip))
    await db.commit()


# ══════════════════════════════════════════════════════════════════════════
# ADMIN — 8. LMS Integration (stub — ready for future connection)
# ══════════════════════════════════════════════════════════════════════════

@router.get("/admin/integrations")
async def admin_get_integrations(admin: User = Depends(get_current_admin)):
    """List available LMS integrations and their status."""
    return {"integrations": [
        {"name": "Moodle", "status": "not_connected", "type": "lms",
         "description": "Sync courses, grades, and attendance with Moodle"},
        {"name": "Canvas", "status": "not_connected", "type": "lms",
         "description": "Integration with Canvas LMS"},
        {"name": "Google Classroom", "status": "not_connected", "type": "lms",
         "description": "Sync with Google Classroom"},
        {"name": "SMTP Email", "status": "not_connected", "type": "notification",
         "description": "Send email notifications to students"},
    ]}


# ══════════════════════════════════════════════════════════════════════════
# ADMIN — 9. Bulk Operations
# ══════════════════════════════════════════════════════════════════════════

class BulkStudentCreate(BaseModel):
    students: list  # [{name, email, password}, ...]


@router.post("/admin/bulk/students")
async def admin_bulk_create_students(
        data: BulkStudentCreate,
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    """Bulk create student accounts."""
    created = 0
    skipped = 0
    errors = []
    for s in data.students:
        try:
            existing = await db.execute(
                select(User).where(User.email == s.get("email")))
            if existing.scalar_one_or_none():
                skipped += 1
                continue
            user = User(
                name=s.get("name", ""),
                email=s.get("email", ""),
                password_hash=hash_password(s.get("password", "student123")),
                role="student")
            db.add(user)
            created += 1
        except Exception as e:
            errors.append({"email": s.get("email"), "error": str(e)})
    await db.commit()
    db.add(AuditLog(user_id=admin.id, action="bulk_student_create",
                    details={"created": created, "skipped": skipped}))
    await db.commit()
    return {"created": created, "skipped": skipped, "errors": errors}


class BulkEnroll(BaseModel):
    course_id: int
    student_emails: list  # [email1, email2, ...]


@router.post("/admin/bulk/enroll")
async def admin_bulk_enroll(
        data: BulkEnroll,
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    """Bulk enroll students in a course by email."""
    course = await db.get(Course, data.course_id)
    if not course:
        raise HTTPException(404, "Course not found")
    enrolled = 0
    skipped = 0
    for email in data.student_emails:
        u_res = await db.execute(select(User).where(User.email == email.strip()))
        user = u_res.scalar_one_or_none()
        if not user:
            skipped += 1
            continue
        existing = await db.execute(
            select(Enrollment).where(
                Enrollment.student_id == user.id,
                Enrollment.course_id == data.course_id))
        if existing.scalar_one_or_none():
            skipped += 1
            continue
        db.add(Enrollment(student_id=user.id, course_id=data.course_id))
        enrolled += 1
    await db.commit()
    return {"enrolled": enrolled, "skipped": skipped, "course": course.name}


@router.get("/admin/export/users")
async def admin_export_users_csv(
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    """Export all users as CSV."""
    result = await db.execute(select(User).order_by(User.id))
    users = result.scalars().all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "name", "email", "role", "is_active", "created_at"])
    for u in users:
        writer.writerow([u.id, u.name, u.email, u.role, u.is_active,
                         u.created_at.isoformat() if u.created_at else ""])
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=users_export.csv"})


# ══════════════════════════════════════════════════════════════════════════
# ADMIN — 10. Institution-Wide Analytics
# ══════════════════════════════════════════════════════════════════════════

@router.get("/admin/analytics/institution")
async def admin_institution_analytics(
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    """Campus-wide engagement analytics."""
    # Per-course engagement
    courses_res = await db.execute(select(Course))
    courses = courses_res.scalars().all()
    course_stats = []
    for c in courses:
        enrolled = (await db.execute(
            select(func.count(Enrollment.id))
            .where(Enrollment.course_id == c.id))).scalar() or 0
        # Get sessions from enrolled students
        stu_ids_res = await db.execute(
            select(Enrollment.student_id)
            .where(Enrollment.course_id == c.id))
        stu_ids = [r[0] for r in stu_ids_res.all()]
        avg_eng = 0
        total_sess = 0
        if stu_ids:
            eng_res = await db.execute(
                select(func.avg(Session.avg_engagement),
                       func.count(Session.id))
                .where(Session.user_id.in_(stu_ids)))
            row = eng_res.one()
            avg_eng = round((row[0] or 0) * 100, 1)
            total_sess = row[1] or 0
        teacher = await db.get(User, c.teacher_id)
        course_stats.append({
            "id": c.id, "name": c.name,
            "teacher": teacher.name if teacher else "Unknown",
            "enrolled": enrolled,
            "total_sessions": total_sess,
            "avg_engagement": avg_eng,
        })
    course_stats.sort(key=lambda x: -x["avg_engagement"])

    # Time-of-day analysis
    tod_res = await db.execute(select(Session.started_at).where(Session.started_at != None))
    hour_counts = {}
    for (ts,) in tod_res.all():
        if ts:
            h = ts.hour
            hour_counts[h] = hour_counts.get(h, 0) + 1

    # Weekly trend (sessions per day, last 30 days)
    thirty_days = datetime.utcnow() - timedelta(days=30)
    daily_res = await db.execute(
        select(Session.started_at, Session.avg_engagement)
        .where(Session.started_at >= thirty_days)
        .order_by(Session.started_at))
    daily = {}
    for ts, eng in daily_res.all():
        if ts:
            day = ts.strftime("%Y-%m-%d")
            if day not in daily:
                daily[day] = {"sessions": 0, "total_eng": 0}
            daily[day]["sessions"] += 1
            daily[day]["total_eng"] += (eng or 0)
    daily_trend = [{
        "date": d,
        "sessions": v["sessions"],
        "avg_engagement": round(v["total_eng"] / v["sessions"] * 100, 1) if v["sessions"] > 0 else 0,
    } for d, v in sorted(daily.items())]

    return {
        "courses": course_stats,
        "time_of_day": hour_counts,
        "daily_trend": daily_trend,
    }


# ══════════════════════════════════════════════════════════════════════════
# ADMIN — System Announcements
# ══════════════════════════════════════════════════════════════════════════

class AnnouncementCreate(BaseModel):
    title: str
    content: str
    priority: str = "normal"
    target_role: Optional[str] = None
    expires_at: Optional[str] = None


@router.post("/admin/announcements")
async def admin_create_announcement(
        data: AnnouncementCreate,
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    expires = None
    if data.expires_at:
        try:
            expires = datetime.fromisoformat(
                data.expires_at.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            pass
    ann = SystemAnnouncement(
        admin_id=admin.id, title=data.title, content=data.content,
        priority=data.priority, target_role=data.target_role,
        expires_at=expires)
    db.add(ann)
    await db.commit()
    return {"success": True, "id": ann.id}


@router.get("/admin/announcements")
async def admin_get_announcements(
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(SystemAnnouncement)
        .order_by(desc(SystemAnnouncement.created_at)))
    return {"announcements": [{
        "id": a.id, "title": a.title, "content": a.content,
        "priority": a.priority, "target_role": a.target_role,
        "is_active": a.is_active,
        "expires_at": a.expires_at.isoformat() if a.expires_at else None,
        "created_at": a.created_at.isoformat() if a.created_at else None,
    } for a in result.scalars().all()]}


@router.delete("/admin/announcements/{ann_id}")
async def admin_delete_announcement(
        ann_id: int,
        admin: User = Depends(get_current_admin),
        db: AsyncSession = Depends(get_db)):
    ann = await db.get(SystemAnnouncement, ann_id)
    if not ann:
        raise HTTPException(404, "Announcement not found")
    await db.delete(ann)
    await db.commit()
    return {"success": True}


@router.get("/announcements/active")
async def get_active_announcements(
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    """Get announcements visible to the current user."""
    now = datetime.utcnow()
    result = await db.execute(
        select(SystemAnnouncement)
        .where(
            SystemAnnouncement.is_active == True,
            or_(SystemAnnouncement.expires_at == None,
                SystemAnnouncement.expires_at > now),
            or_(SystemAnnouncement.target_role == None,
                SystemAnnouncement.target_role == user.role))
        .order_by(desc(SystemAnnouncement.created_at))
        .limit(10))
    return {"announcements": [{
        "id": a.id, "title": a.title, "content": a.content,
        "priority": a.priority,
        "created_at": a.created_at.isoformat() if a.created_at else None,
    } for a in result.scalars().all()]}


# ══════════════════════════════════════════════════════════════════════════
# CHAT SYSTEM — Direct messages (Student↔Teacher, Teacher↔Admin)
# ══════════════════════════════════════════════════════════════════════════

@router.get("/chat/conversations")
async def get_conversations(
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    """List all conversations for the current user."""
    result = await db.execute(
        select(Conversation)
        .where(or_(
            Conversation.user1_id == user.id,
            Conversation.user2_id == user.id))
        .order_by(desc(Conversation.last_at)))
    convos = result.scalars().all()
    out = []
    for c in convos:
        other_id = c.user2_id if c.user1_id == user.id else c.user1_id
        other = await db.get(User, other_id)
        # Count unread
        unread = (await db.execute(
            select(func.count(Message.id))
            .where(Message.conversation_id == c.id,
                   Message.sender_id != user.id,
                   Message.is_read == False))).scalar() or 0
        out.append({
            "id": c.id,
            "other_user": {
                "id": other.id, "name": other.name,
                "email": other.email, "role": other.role,
            } if other else None,
            "last_message": c.last_message,
            "last_at": c.last_at.isoformat() if c.last_at else None,
            "unread": unread,
        })
    return {"conversations": out}


class StartConversation(BaseModel):
    other_user_id: int
    message: str


@router.post("/chat/conversations")
async def start_conversation(
        data: StartConversation,
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    """Start a new conversation or send to existing one."""
    other = await db.get(User, data.other_user_id)
    if not other:
        raise HTTPException(404, "User not found")
    # Check role permissions (student↔teacher, teacher↔admin)
    allowed = False
    if user.role == "student" and other.role == "teacher":
        allowed = True
    elif user.role == "teacher" and other.role in ("student", "admin"):
        allowed = True
    elif user.role == "admin":
        allowed = True
    if not allowed:
        raise HTTPException(403, "You can only message your instructors or admins")
    # Find existing conversation
    result = await db.execute(
        select(Conversation).where(
            or_(
                and_(Conversation.user1_id == user.id,
                     Conversation.user2_id == data.other_user_id),
                and_(Conversation.user1_id == data.other_user_id,
                     Conversation.user2_id == user.id))))
    convo = result.scalar_one_or_none()
    if not convo:
        convo = Conversation(
            user1_id=user.id, user2_id=data.other_user_id,
            last_message=data.message, last_at=datetime.utcnow())
        db.add(convo)
        await db.flush()
    # Add message
    msg = Message(conversation_id=convo.id,
                  sender_id=user.id, content=data.message)
    convo.last_message = data.message
    convo.last_at = datetime.utcnow()
    db.add(msg)
    await db.commit()
    return {"success": True, "conversation_id": convo.id, "message_id": msg.id}


@router.get("/chat/conversations/{convo_id}/messages")
async def get_messages(
        convo_id: int,
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    """Get all messages in a conversation."""
    convo = await db.get(Conversation, convo_id)
    if not convo:
        raise HTTPException(404, "Conversation not found")
    if convo.user1_id != user.id and convo.user2_id != user.id:
        raise HTTPException(403, "Not your conversation")
    # Mark messages as read
    unread = await db.execute(
        select(Message).where(
            Message.conversation_id == convo_id,
            Message.sender_id != user.id,
            Message.is_read == False))
    for m in unread.scalars().all():
        m.is_read = True
    await db.commit()
    # Get all messages
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == convo_id)
        .order_by(Message.created_at))
    msgs = result.scalars().all()
    return {"messages": [{
        "id": m.id,
        "sender_id": m.sender_id,
        "is_mine": m.sender_id == user.id,
        "content": m.content,
        "is_read": m.is_read,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    } for m in msgs]}


class SendMessage(BaseModel):
    content: str


@router.post("/chat/conversations/{convo_id}/messages")
async def send_message(
        convo_id: int, data: SendMessage,
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    """Send a message in an existing conversation."""
    convo = await db.get(Conversation, convo_id)
    if not convo:
        raise HTTPException(404, "Conversation not found")
    if convo.user1_id != user.id and convo.user2_id != user.id:
        raise HTTPException(403, "Not your conversation")
    msg = Message(conversation_id=convo_id,
                  sender_id=user.id, content=data.content)
    convo.last_message = data.content
    convo.last_at = datetime.utcnow()
    db.add(msg)
    await db.commit()
    return {"success": True, "message_id": msg.id}


@router.get("/chat/unread-count")
async def chat_unread_count(
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    """Total unread messages across all conversations."""
    convos = await db.execute(
        select(Conversation.id).where(
            or_(Conversation.user1_id == user.id,
                Conversation.user2_id == user.id)))
    convo_ids = [r[0] for r in convos.all()]
    if not convo_ids:
        return {"unread": 0}
    count = (await db.execute(
        select(func.count(Message.id))
        .where(Message.conversation_id.in_(convo_ids),
               Message.sender_id != user.id,
               Message.is_read == False))).scalar() or 0
    return {"unread": count}


@router.get("/chat/contacts")
async def get_chat_contacts(
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):
    """List users the current user can message."""
    contacts = []
    if user.role == "student":
        # Students can message teachers of their enrolled courses
        enrolled = await db.execute(
            select(Course.teacher_id).distinct()
            .join(Enrollment, Enrollment.course_id == Course.id)
            .where(Enrollment.student_id == user.id))
        teacher_ids = [r[0] for r in enrolled.all()]
        if teacher_ids:
            teachers = await db.execute(
                select(User).where(User.id.in_(teacher_ids)))
            contacts = [{"id": t.id, "name": t.name, "email": t.email,
                         "role": t.role} for t in teachers.scalars().all()]
    elif user.role == "teacher":
        # Teachers can message their students + admins
        course_ids = await db.execute(
            select(Course.id).where(Course.teacher_id == user.id))
        cids = [r[0] for r in course_ids.all()]
        if cids:
            stu_res = await db.execute(
                select(User).distinct()
                .join(Enrollment, Enrollment.student_id == User.id)
                .where(Enrollment.course_id.in_(cids)))
            contacts.extend([{"id": s.id, "name": s.name, "email": s.email,
                              "role": s.role} for s in stu_res.scalars().all()])
        admin_res = await db.execute(
            select(User).where(User.role == "admin"))
        contacts.extend([{"id": a.id, "name": a.name, "email": a.email,
                          "role": a.role} for a in admin_res.scalars().all()])
    elif user.role == "admin":
        # Admins can message everyone
        all_res = await db.execute(
            select(User).where(User.id != user.id))
        contacts = [{"id": u.id, "name": u.name, "email": u.email,
                     "role": u.role} for u in all_res.scalars().all()]
    return {"contacts": contacts}
