from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models_db import User, get_db
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

SECRET_KEY = os.getenv("SECRET_KEY", "fallback_secret")
ALGORITHM  = os.getenv("ALGORITHM", "HS256")
EXPIRE_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security    = HTTPBearer()

def hash_password(password: str) -> str:
    # bcrypt has a 72 byte limit — truncate to be safe
    return pwd_context.hash(password[:72])

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain[:72], hashed)

def create_token(data: dict,
                 expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire    = datetime.utcnow() + (
        expires_delta or timedelta(minutes=EXPIRE_MIN))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    try:
        return jwt.decode(
            token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None

def oauth2_scheme(
        credentials: HTTPAuthorizationCredentials = Depends(security)):
    return credentials.credentials

async def get_current_user(
        token: str = Depends(oauth2_scheme),
        db   : AsyncSession = Depends(get_db)) -> User:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"})
    payload = decode_token(token)
    if not payload:
        raise exc
    user_id = payload.get("sub")
    if not user_id:
        raise exc
    result = await db.execute(
        select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise exc
    return user

async def get_current_teacher(
        current_user: User = Depends(get_current_user)) -> User:
    if current_user.role not in ["teacher", "admin"]:
        raise HTTPException(status_code=403,
                            detail="Teacher access required")
    return current_user

async def get_current_admin(
        current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(status_code=403,
                            detail="Admin access required")
    return current_user
