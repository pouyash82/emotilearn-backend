import io, csv
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from models_db import (
    User, Course, Lecture, Session, EmotionLog,
    Enrollment, Exam, ExamSubmission, Notification, get_db)
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
