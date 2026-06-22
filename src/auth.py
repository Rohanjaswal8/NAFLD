"""Local user authentication with JSON storage and remember-me tokens."""

import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

from src.config import ROOT_DIR

DATA_DIR = ROOT_DIR / "data"
USERS_FILE = DATA_DIR / "users.json"
TOKENS_FILE = DATA_DIR / "auth_tokens.json"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists():
        USERS_FILE.write_text("{}", encoding="utf-8")
    if not TOKENS_FILE.exists():
        TOKENS_FILE.write_text("{}", encoding="utf-8")


def _load_json(path: Path) -> dict:
    _ensure_data_dir()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_json(path: Path, data: dict) -> None:
    _ensure_data_dir()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def signup(name: str, email: str, password: str) -> tuple[bool, str]:
    name = name.strip()
    email = email.strip().lower()
    password = password.strip()

    if not name or not email or not password:
        return False, "All fields are required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if "@" not in email:
        return False, "Please enter a valid email address."

    users = _load_json(USERS_FILE)
    if email in users:
        return False, "An account with this email already exists."

    salt = secrets.token_hex(16)
    users[email] = {
        "name": name,
        "email": email,
        "password_hash": _hash_password(password, salt),
        "salt": salt,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_json(USERS_FILE, users)
    return True, "Account created successfully. Please log in."


def login(email: str, password: str) -> tuple[bool, str, dict | None, str | None]:
    email = email.strip().lower()
    password = password.strip()

    users = _load_json(USERS_FILE)
    user = users.get(email)
    if not user:
        return False, "Invalid email or password.", None, None

    if _hash_password(password, user["salt"]) != user["password_hash"]:
        return False, "Invalid email or password.", None, None

    token = secrets.token_urlsafe(32)
    tokens = _load_json(TOKENS_FILE)
    tokens[token] = {
        "email": email,
        "name": user["name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_json(TOKENS_FILE, tokens)

    profile = {"name": user["name"], "email": email}
    return True, "Login successful.", profile, token


def validate_token(token: str | None) -> dict | None:
    if not token or token == "null":
        return None
    tokens = _load_json(TOKENS_FILE)
    session = tokens.get(token)
    if not session:
        return None
    return {"name": session["name"], "email": session["email"], "token": token}


def logout(token: str | None) -> None:
    if not token:
        return
    tokens = _load_json(TOKENS_FILE)
    if token in tokens:
        del tokens[token]
        _save_json(TOKENS_FILE, tokens)
