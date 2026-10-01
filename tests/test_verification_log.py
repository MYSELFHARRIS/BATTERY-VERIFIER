"""Tests for VerificationLog with hash chain verification."""

from pathlib import Path
import sqlite3
from src.logic.verification_log import VerificationLog


def test_three_appends_valid_chain(tmp_path: Path) -> None:
    """Test that three appended records produce a valid hash chain."""
    db_file = tmp_path / "test_chain.db"
    with VerificationLog(db_file) as log:
        log.append(1, "pass", 0.98, 1, timestamp="2026-10-01T10:00:00Z")
        log.append(2, "pass", 0.95, 1, timestamp="2026-10-01T10:01:00Z")
        log.append(3, "pass", 0.99, 1, timestamp="2026-10-01T10:02:00Z")

        valid, bad_id = log.verify_chain()
        assert valid is True
        assert bad_id is None


def test_edit_confidence_tamper_detected(tmp_path: Path) -> None:
    """Test that directly tampering with confidence is detected."""
    db_file = tmp_path / "test_tamper.db"
    with VerificationLog(db_file) as log:
        log.append(1, "pass", 0.98, 1, timestamp="2026-10-01T10:00:00Z")
        row2_id = log.append(2, "pass", 0.95, 1, timestamp="2026-10-01T10:01:00Z")
        log.append(3, "pass", 0.99, 1, timestamp="2026-10-01T10:02:00Z")

    # Tamper with row 2 directly via sqlite3
    conn = sqlite3.connect(str(db_file))
    try:
        with conn:
            conn.execute("UPDATE log SET confidence = 0.50 WHERE id = ?", (row2_id,))
    finally:
        conn.close()

    # Reopen and verify tampering detection
    with VerificationLog(db_file) as log:
        valid, bad_id = log.verify_chain()
        assert valid is False
        assert bad_id == row2_id


def test_delete_middle_row_detected(tmp_path: Path) -> None:
    """Test that deleting a row breaks the chain and is detected."""
    db_file = tmp_path / "test_delete.db"
    with VerificationLog(db_file) as log:
        log.append(1, "pass", 0.98, 1, timestamp="2026-10-01T10:00:00Z")
        row2_id = log.append(2, "pass", 0.95, 1, timestamp="2026-10-01T10:01:00Z")
        log.append(3, "pass", 0.99, 1, timestamp="2026-10-01T10:02:00Z")

    # Delete the middle row directly via sqlite3
    conn = sqlite3.connect(str(db_file))
    try:
        with conn:
            conn.execute("DELETE FROM log WHERE id = ?", (row2_id,))
    finally:
        conn.close()

    # Reopen and verify chain failure
    with VerificationLog(db_file) as log:
        valid, bad_id = log.verify_chain()
        assert valid is False
        assert bad_id is not None


def test_empty_log_is_valid(tmp_path: Path) -> None:
    """Test that an empty log is considered valid."""
    db_file = tmp_path / "test_empty.db"
    with VerificationLog(db_file) as log:
        valid, bad_id = log.verify_chain()
        assert valid is True
        assert bad_id is None


def test_reopen_and_append_chain_valid(tmp_path: Path) -> None:
    """Test that reopening an existing DB and appending preserves chain validity."""
    db_file = tmp_path / "test_reopen.db"
    with VerificationLog(db_file) as log1:
        log1.append(1, "pass", 0.98, 1, timestamp="2026-10-01T10:00:00Z")
        log1.append(2, "pass", 0.95, 1, timestamp="2026-10-01T10:01:00Z")

    # Reopen in a new instance and append
    with VerificationLog(db_file) as log2:
        log2.append(3, "pass", 0.99, 1, timestamp="2026-10-01T10:02:00Z")
        valid, bad_id = log2.verify_chain()
        assert valid is True
        assert bad_id is None
