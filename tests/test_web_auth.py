import sqlite3
import pytest
from src.web.app import create_app
from src.web.auth import create_user


@pytest.fixture
def app(tmp_path):
    db_path = str(tmp_path / "test_users.db")
    test_app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test_secret_key",
            "USERS_DB": db_path,
        }
    )
    return test_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def db_path(app):
    return app.config["USERS_DB"]


def test_get_app_without_sign_in_redirects_to_login(client):
    response = client.get("/app")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/login")


def test_valid_credentials_sign_in_and_app_returns_200(client, db_path):
    create_user(db_path, "operator@example.com", "Password123!")

    response = client.post(
        "/login",
        data={"email": "operator@example.com", "password": "Password123!"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"operator@example.com" in response.data
    assert b"Battery Verifier" in response.data

    # Direct subsequent GET /app also returns 200
    app_response = client.get("/app")
    assert app_response.status_code == 200
    assert b"Dashboard" in app_response.data


def test_wrong_password_and_unknown_email_return_same_message(client, db_path):
    create_user(db_path, "user@example.com", "SecretPass123")

    # Unknown email
    resp_unknown = client.post(
        "/login",
        data={"email": "nobody@example.com", "password": "SecretPass123"},
    )
    assert resp_unknown.status_code == 200
    assert b"Invalid email or password" in resp_unknown.data

    # Known email with wrong password
    resp_wrong = client.post(
        "/login",
        data={"email": "user@example.com", "password": "WrongPassword!"},
    )
    assert resp_wrong.status_code == 200
    assert b"Invalid email or password" in resp_wrong.data

    # Ensure message is identical
    assert b"Invalid email or password" in resp_unknown.data
    assert b"Invalid email or password" in resp_wrong.data


def test_logout_clears_session_and_app_redirects_to_login(client, db_path):
    create_user(db_path, "logout_user@example.com", "Password123!")
    client.post(
        "/login",
        data={"email": "logout_user@example.com", "password": "Password123!"},
    )

    # Confirm authenticated
    app_response = client.get("/app")
    assert app_response.status_code == 200

    # Logout
    logout_resp = client.post("/logout")
    assert logout_resp.status_code == 302
    assert logout_resp.headers["Location"].endswith("/signed-out")

    # Subsequent /app redirects to /login
    follow_up = client.get("/app")
    assert follow_up.status_code == 302
    assert follow_up.headers["Location"].endswith("/login")


def test_stored_password_hash_is_not_plaintext(db_path):
    plain_pass = "MySecretPassword123"
    create_user(db_path, "secure@example.com", plain_pass)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE email = 'secure@example.com'")
    row = cur.fetchone()
    conn.close()

    assert row is not None
    stored_hash = row[0]
    assert stored_hash != plain_pass
    assert plain_pass not in stored_hash
    assert stored_hash.startswith("pbkdf2:sha256")


def test_create_user_rejects_invalid_inputs(db_path):
    # Short password (< 8 chars)
    with pytest.raises(ValueError, match="at least 8 characters"):
        create_user(db_path, "valid@example.com", "short")

    # Bad emails
    with pytest.raises(ValueError, match="Invalid email"):
        create_user(db_path, "no_at_sign.com", "password123")

    with pytest.raises(ValueError, match="Invalid email"):
        create_user(db_path, "two@@example.com", "password123")

    with pytest.raises(ValueError, match="Invalid email"):
        create_user(db_path, "user@nodot", "password123")

    with pytest.raises(ValueError, match="Invalid email"):
        create_user(db_path, "@nodomain.com", "password123")

    # Duplicate email (case-insensitive test)
    create_user(db_path, "dupe@example.com", "password123")
    with pytest.raises(ValueError, match="already exists"):
        create_user(db_path, "DUPE@example.com", "newpassword123")


def test_login_page_contains_email_and_password_fields(client):
    response = client.get("/login")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Email" in html
    assert 'name="email"' in html
    assert "Password" in html
    assert 'name="password"' in html
    assert 'type="password"' in html
    assert "Battery Verifier" in html
