"""Authentication and user management backed by SQLite."""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from werkzeug.security import check_password_hash, generate_password_hash


def is_valid_email(email: str) -> bool:
    """Validate email with a simple check: one '@' and a dot after it."""
    if not isinstance(email, str):
        return False
    if email.count("@") != 1:
        return False
    local, domain = email.split("@")
    if not local or not domain:
        return False
    if "." not in domain:
        return False
    dot_idx = domain.find(".")
    if dot_idx == 0 or dot_idx == len(domain) - 1:
        return False
    return True


def init_db(db_path: str | Path) -> None:
    """Ensure the users table exists in the SQLite database."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def create_user(db_path: str | Path, email: str, password: str) -> int:
    """Create a new user with hashed password.

    Raises:
        ValueError: If email is invalid, password is < 8 chars, or email already exists.
    """
    if not is_valid_email(email):
        raise ValueError(f"Invalid email: '{email}'")

    if not isinstance(password, str) or len(password) < 8:
        raise ValueError("Password must be at least 8 characters long")

    init_db(db_path)
    email_norm = email.strip().lower()
    pw_hash = generate_password_hash(password, method="pbkdf2:sha256")
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (email, password_hash, created_at)
                VALUES (?, ?, ?)
                """,
                (email_norm, pw_hash, now_iso),
            )
            return int(cursor.lastrowid)
    except sqlite3.IntegrityError as e:
        raise ValueError(f"User with email '{email_norm}' already exists") from e


def verify_user(db_path: str | Path, email: str, password: str) -> bool:
    """Verify user credentials in constant time.

    Returns False for unknown users or invalid credentials.
    """
    if not email or not password or not isinstance(email, str) or not isinstance(password, str):
        return False

    init_db(db_path)
    email_norm = email.strip().lower()

    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.execute(
            "SELECT password_hash FROM users WHERE email = ?",
            (email_norm,),
        )
        row = cursor.fetchone()

    dummy_hash = (
        "pbkdf2:sha256:600000$0000000000000000$"
        "0000000000000000000000000000000000000000000000000000000000000000"
    )
    if row is None:
        check_password_hash(dummy_hash, password)
        return False

    return bool(check_password_hash(row[0], password))
