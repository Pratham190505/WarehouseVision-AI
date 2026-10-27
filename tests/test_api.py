"""
Comprehensive Tests for WarehouseVision AI API & Authentication System.

Validates:
- Health check endpoints
- User registration and validation
- Duplicate account prevention
- Password hashing & verification
- Login, session tokens, and cookie handling
- Protected /dashboard redirection
- /api/auth/me endpoint
- Logout flow
- User-scoped analytics & privacy enforcement
"""

import io
import sys
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app, follow_redirects=False)


class TestHealthEndpoint:
    """Tests for health endpoints."""

    def test_health_returns_200(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "WarehouseVision AI"
        assert data["status"] == "healthy"


class TestAuthenticationFlow:
    """Tests for registration, login, logout, and protected routes."""

    test_user_email = "tester_warehouse@example.com"
    test_user_pwd = "SecurePassword123!"

    def test_unauthenticated_dashboard_redirects_to_login(self):
        """Unauthenticated user accessing /dashboard must be redirected to /login."""
        response = client.get("/dashboard")
        assert response.status_code == 303
        assert response.headers["location"] == "/login"

    def test_unauthenticated_root_redirects_to_login(self):
        """Unauthenticated user accessing / must be redirected to /login."""
        response = client.get("/")
        assert response.status_code == 303
        assert response.headers["location"] == "/login"

    def test_login_and_signup_pages_render(self):
        """Verify login and signup HTML templates load."""
        login_resp = client.get("/login")
        assert login_resp.status_code == 200
        assert "Welcome Back" in login_resp.text

        signup_resp = client.get("/signup")
        assert signup_resp.status_code == 200
        assert "Create Account" in signup_resp.text

    def test_signup_validation_mismatched_passwords(self):
        """Password confirmation mismatch should be rejected."""
        response = client.post("/api/auth/signup", json={
            "name": "Warehouse Tester",
            "email": "mismatch@example.com",
            "password": "Password123",
            "confirm_password": "DifferentPassword456",
        })
        assert response.status_code == 400
        assert "passwords do not match" in response.json()["detail"].lower()

    def test_signup_validation_short_password(self):
        """Passwords shorter than 6 characters should be rejected."""
        response = client.post("/api/auth/signup", json={
            "name": "Warehouse Tester",
            "email": "short@example.com",
            "password": "123",
            "confirm_password": "123",
        })
        assert response.status_code == 400

    def test_signup_success(self):
        """Valid registration creates account and establishes session."""
        response = client.post("/api/auth/signup", json={
            "name": "Pratham Operator",
            "email": self.test_user_email,
            "password": self.test_user_pwd,
            "confirm_password": self.test_user_pwd,
        })
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["user"]["email"] == self.test_user_email
        assert "password_hash" not in data["user"]
        assert "wv_session" in response.cookies

    def test_signup_duplicate_email_rejected(self):
        """Attempting to register with duplicate email should return 400."""
        response = client.post("/api/auth/signup", json={
            "name": "Duplicate Tester",
            "email": self.test_user_email,
            "password": self.test_user_pwd,
            "confirm_password": self.test_user_pwd,
        })
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"].lower()

    def test_login_invalid_credentials(self):
        """Wrong password returns 401."""
        response = client.post("/api/auth/login", json={
            "email": self.test_user_email,
            "password": "WrongPassword999!",
        })
        assert response.status_code == 401
        assert "invalid email or password" in response.json()["detail"].lower()

    def test_login_success_and_me(self):
        """Valid login sets session cookie and unlocks /api/auth/me and /dashboard."""
        auth_client = TestClient(app, follow_redirects=False)

        login_resp = auth_client.post("/api/auth/login", json={
            "email": self.test_user_email,
            "password": self.test_user_pwd,
        })
        assert login_resp.status_code == 200
        assert "wv_session" in login_resp.cookies

        # Test /api/auth/me
        me_resp = auth_client.get("/api/auth/me")
        assert me_resp.status_code == 200
        me_data = me_resp.json()
        assert me_data["email"] == self.test_user_email
        assert me_data["name"] == "Pratham Operator"

        # Test authenticated /dashboard view
        dash_resp = auth_client.get("/dashboard")
        assert dash_resp.status_code == 200
        assert "WarehouseVision AI" in dash_resp.text

        # Test logout
        logout_resp = auth_client.post("/api/auth/logout")
        assert logout_resp.status_code == 200

        # After logout, /api/auth/me should return 401
        after_logout = auth_client.get("/api/auth/me")
        assert after_logout.status_code == 401


class TestUserScopedAnalytics:
    """Test that users only see their own analyses and statistics."""

    def test_privacy_isolation_between_users(self):
        client_a = TestClient(app, follow_redirects=False)
        client_b = TestClient(app, follow_redirects=False)

        # Register User A
        client_a.post("/api/auth/signup", json={
            "name": "User Alpha",
            "email": "alpha@example.com",
            "password": "Password123!",
            "confirm_password": "Password123!",
        })

        # Register User B
        client_b.post("/api/auth/signup", json={
            "name": "User Beta",
            "email": "beta@example.com",
            "password": "Password123!",
            "confirm_password": "Password123!",
        })

        # User A uploads an image
        fake_jpg = b"\xff\xd8\xff\xe0" + b"\x00" * 200
        up_resp = client_a.post(
            "/api/upload",
            files={"file": ("warehouse_a.jpg", io.BytesIO(fake_jpg), "image/jpeg")},
        )
        assert up_resp.status_code == 200
        analysis_id = up_resp.json()["analysis_id"]

        # User A can access the analysis
        resp_a = client_a.get(f"/api/analysis/{analysis_id}")
        assert resp_a.status_code == 200

        # User B cannot access User A's analysis (must return 404 access denied)
        resp_b = client_b.get(f"/api/analysis/{analysis_id}")
        assert resp_b.status_code == 404

        # User B history should not include User A's upload
        b_history = client_b.get("/api/history").json()["history"]
        assert all(item["analysis_id"] != analysis_id for item in b_history)

    def test_video_upload_and_retrieval(self):
        auth_client = TestClient(app, follow_redirects=False)
        auth_client.post("/api/auth/signup", json={
            "name": "Video Operator",
            "email": "videotrack@example.com",
            "password": "Password123!",
            "confirm_password": "Password123!",
        })

        fake_mp4 = b"\x00\x00\x00\x1c" + b"ftypisom" + b"\x00" * 300
        up_resp = auth_client.post(
            "/api/upload",
            files={"file": ("warehouse_aisle.mp4", io.BytesIO(fake_mp4), "video/mp4")},
        )
        assert up_resp.status_code == 200
        data = up_resp.json()
        assert data["media_type"] == "video"
        analysis_id = data["analysis_id"]

        # Retrieve analysis
        get_resp = auth_client.get(f"/api/analysis/{analysis_id}")
        assert get_resp.status_code == 200
        analysis_data = get_resp.json()
        assert analysis_data["media_type"] == "video"
        assert analysis_data["original_filename"] == "warehouse_aisle.mp4"


class TestUserProfileSettings:
    """Tests for user profile updates via Settings."""

    def test_unauthenticated_profile_update_rejected(self):
        """Unauthenticated profile update must return 401."""
        unauth_client = TestClient(app, follow_redirects=False)
        resp = unauth_client.post("/api/auth/profile", json={"name": "Hacker"})
        assert resp.status_code == 401

    def test_profile_update_validation_short_name(self):
        """Profile update with empty or single char name must return 400."""
        auth_client = TestClient(app, follow_redirects=False)
        auth_client.post("/api/auth/signup", json={
            "name": "Original Name",
            "email": "settings_val@example.com",
            "password": "Password123!",
            "confirm_password": "Password123!",
        })
        resp = auth_client.post("/api/auth/profile", json={"name": "A"})
        assert resp.status_code == 400
        assert "at least 2 characters" in resp.json()["detail"]


    def test_profile_update_success_and_reflected_on_dashboard(self):
        """Profile update successfully modifies name, organization, department and is reflected on /dashboard."""
        auth_client = TestClient(app, follow_redirects=False)
        auth_client.post("/api/auth/signup", json={
            "name": "Initial Name",
            "email": "settings_update@example.com",
            "password": "Password123!",
            "confirm_password": "Password123!",
        })

        # Update profile
        update_resp = auth_client.post("/api/auth/profile", json={
            "name": "Pratham Vision Lead",
            "organization": "Global Warehouse Hub Alpha",
            "department": "AI Operations Lead",
        })
        assert update_resp.status_code == 200
        data = update_resp.json()
        assert data["success"] is True
        assert data["user"]["name"] == "Pratham Vision Lead"
        assert data["user"]["organization"] == "Global Warehouse Hub Alpha"
        assert data["user"]["department"] == "AI Operations Lead"

        # Check /api/auth/me
        me_resp = auth_client.get("/api/auth/me")
        assert me_resp.status_code == 200
        assert me_resp.json()["name"] == "Pratham Vision Lead"

        # Check /dashboard renders the new name
        dash_resp = auth_client.get("/dashboard")
        assert dash_resp.status_code == 200
        assert "Pratham Vision Lead" in dash_resp.text

        settings_resp = auth_client.get("/settings")
        assert settings_resp.status_code == 200
        assert "Pratham Vision Lead" in settings_resp.text
        assert "Global Warehouse Hub Alpha" in settings_resp.text
