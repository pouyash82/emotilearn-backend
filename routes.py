import io, csv
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
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
                        datetime.utcnow().isoformat())),
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
