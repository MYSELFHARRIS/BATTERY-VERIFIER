"""Verification log with hash chaining backed by SQLite."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

INITIAL_PREV_HASH = "0" * 64


def compute_hash(
    step_id: int,
    result: str,
    confidence: float,
    timestamp: str,
    sop_version: int,
    prev_hash: str,
) -> str:
    """Compute SHA-256 hash for a verification log entry."""
    payload = [step_id, result, confidence, timestamp, sop_version, prev_hash]
    serialized = json.dumps(payload)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class VerificationLog:
    """Tamper-evident verification log using hash chaining."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.conn: sqlite3.Connection | None = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self) -> None:
        if self.conn is None:
            raise RuntimeError("Database connection is closed.")
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    step_id INTEGER,
                    result TEXT,
                    confidence REAL,
                    timestamp TEXT,
                    sop_version INTEGER,
                    prev_hash TEXT,
                    hash TEXT
                )
                """
            )

    def append(
        self,
        step_id: int,
        result: str,
        confidence: float,
        sop_version: int,
        timestamp: str | None = None,
    ) -> int:
        """Append a verification record to the hash-chained log."""
        if self.conn is None:
            raise RuntimeError("Database connection is closed.")

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        cursor = self.conn.cursor()
        cursor.execute("SELECT hash FROM log ORDER BY id DESC LIMIT 1")
        last_row = cursor.fetchone()
        prev_hash = last_row[0] if last_row is not None else INITIAL_PREV_HASH

        current_hash = compute_hash(
            step_id=step_id,
            result=result,
            confidence=confidence,
            timestamp=timestamp,
            sop_version=sop_version,
            prev_hash=prev_hash,
        )

        with self.conn:
            cursor.execute(
                """
                INSERT INTO log (step_id, result, confidence, timestamp, sop_version, prev_hash, hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (step_id, result, confidence, timestamp, sop_version, prev_hash, current_hash),
            )
        return int(cursor.lastrowid)

    def verify_chain(self) -> tuple[bool, int | None]:
        """Verify the integrity of the entire hash chain."""
        if self.conn is None:
            raise RuntimeError("Database connection is closed.")

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, step_id, result, confidence, timestamp, sop_version, prev_hash, hash
            FROM log
            ORDER BY id ASC
            """
        )
        rows = cursor.fetchall()
        if not rows:
            return True, None

        expected_prev_hash = INITIAL_PREV_HASH
        for row in rows:
            row_id, step_id, result, confidence, timestamp, sop_version, prev_hash, stored_hash = row
            if prev_hash != expected_prev_hash:
                return False, row_id

            calculated_hash = compute_hash(
                step_id=step_id,
                result=result,
                confidence=confidence,
                timestamp=timestamp,
                sop_version=sop_version,
                prev_hash=prev_hash,
            )
            if stored_hash != calculated_hash:
                return False, row_id

            expected_prev_hash = stored_hash

        return True, None

    def close(self) -> None:
        """Close the SQLite database connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "VerificationLog":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()
