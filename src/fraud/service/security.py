"""Passwords, JWTs and role checks."""

import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from fraud.service.db import User, get_db
from fraud.service.settings import get_service_settings

ROLES = ("analyst", "admin")
PBKDF2_ROUNDS = 200_000
bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), PBKDF2_ROUNDS).hex()
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, rounds, salt, digest = stored.split("$")
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(rounds)).hex()
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(candidate, digest)


def create_token(username: str, role: str) -> str:
    settings = get_service_settings()
    payload = {"sub": username, "role": role, "exp": datetime.now(UTC) + timedelta(minutes=settings.token_minutes)}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    settings = get_service_settings()
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token",
                            headers={"WWW-Authenticate": "Bearer"}) from exc


def user_from_token(token: str, db: Session) -> User:
    claims = decode_token(token)
    user = db.scalar(select(User).where(User.username == claims.get("sub")))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unknown user")
    return user


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
                 db: Session = Depends(get_db)) -> User:
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    return user_from_token(credentials.credentials, db)


def require_role(*roles: str):
    """Dependency: the caller must be signed in with one of ``roles``."""
    def check(user: User = Depends(current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Requires role: {' or '.join(roles)}")
        return user
    return check


any_user = require_role(*ROLES)
admin_only = require_role("admin")


def ensure_default_users(db: Session):
    """Create the demo analyst and admin accounts if no users exist yet."""
    if db.scalar(select(User).limit(1)) is not None:
        return
    settings = get_service_settings()
    db.add_all([
        User(username=settings.admin_username, password_hash=hash_password(settings.admin_password), role="admin"),
        User(username=settings.analyst_username, password_hash=hash_password(settings.analyst_password), role="analyst"),
    ])
    db.commit()
