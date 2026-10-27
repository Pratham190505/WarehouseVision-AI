"""
Authentication and Session Service for WarehouseVision AI.

Handles:
- Secure password hashing and verification via bcrypt
- User registration and validation
- Session creation, validation, and destruction
- In-memory session fallback for testing/offline mode
"""

import re
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from bson import ObjectId

import bcrypt
from fastapi import Request, HTTPException, status
from app.database.mongodb import get_collection

logger = logging.getLogger(__name__)

# Session cookie name
SESSION_COOKIE_NAME = "wv_session"
SESSION_MAX_AGE_DAYS = 7

# Email validation regex
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

# In-memory session & user fallback cache (used when MongoDB is unavailable/in testing)
_in_memory_users: Dict[str, Dict[str, Any]] = {}
_in_memory_sessions: Dict[str, Dict[str, Any]] = {}


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against the stored bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def register_user(name: str, email: str, password: str, confirm_password: str) -> Dict[str, Any]:
    """
    Validate and register a new user.
    
    Raises HTTPException for validation or duplicate errors.
    Returns the created user record.
    """
    clean_name = (name or "").strip()
    clean_email = (email or "").strip().lower()

    if not clean_name or len(clean_name) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Name must be at least 2 characters.",
        )

    if not clean_email or not EMAIL_REGEX.match(clean_email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide a valid email address.",
        )

    if not password or len(password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters long.",
        )

    if password != confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match.",
        )

    users_col = get_collection("users")

    # Check if duplicate email exists in MongoDB
    if users_col is not None:
        try:
            existing = users_col.find_one({"email": clean_email})
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="An account with this email already exists.",
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error querying users collection: {e}")
    else:
        # Fallback check
        if clean_email in _in_memory_users:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="An account with this email already exists.",
            )

    pwd_hash = hash_password(password)
    user_doc = {
        "name": clean_name,
        "email": clean_email,
        "password_hash": pwd_hash,
        "role": "user",
        "created_at": datetime.utcnow(),
    }

    if users_col is not None:
        try:
            result = users_col.insert_one(user_doc)
            user_doc["_id"] = result.inserted_id
        except Exception as e:
            logger.error(f"Failed to insert user: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to register user in database.",
            )
    else:
        mock_id = str(ObjectId())
        user_doc["_id"] = mock_id
        _in_memory_users[clean_email] = user_doc

    return sanitize_user(user_doc)


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate user by email and password.
    Returns user dict on success, None on failure.
    """
    clean_email = (email or "").strip().lower()
    if not clean_email or not password:
        return None

    users_col = get_collection("users")
    user_doc = None

    if users_col is not None:
        try:
            user_doc = users_col.find_one({"email": clean_email})
        except Exception as e:
            logger.error(f"Failed to lookup user {clean_email}: {e}")
    else:
        user_doc = _in_memory_users.get(clean_email)

    if not user_doc:
        return None

    pwd_hash = user_doc.get("password_hash", "")
    if not verify_password(password, pwd_hash):
        return None

    return sanitize_user(user_doc)


def create_session(user_id: str) -> str:
    """Create a new session token for the user."""
    session_id = secrets.token_urlsafe(32)
    now = datetime.utcnow()
    expires_at = now + timedelta(days=SESSION_MAX_AGE_DAYS)

    session_doc = {
        "session_id": session_id,
        "user_id": str(user_id),
        "created_at": now,
        "expires_at": expires_at,
    }

    sessions_col = get_collection("sessions")
    if sessions_col is not None:
        try:
            sessions_col.insert_one(session_doc)
        except Exception as e:
            logger.error(f"Failed to store session: {e}")

    # Also keep in in-memory session cache
    _in_memory_sessions[session_id] = session_doc
    return session_id


def get_user_from_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve and validate the active user for a session token."""
    if not session_id:
        return None

    sessions_col = get_collection("sessions")
    session_data = None

    if sessions_col is not None:
        try:
            session_data = sessions_col.find_one({"session_id": session_id})
        except Exception as e:
            logger.error(f"Error querying session: {e}")

    if not session_data:
        session_data = _in_memory_sessions.get(session_id)

    if not session_data:
        return None

    # Check expiry
    expires_at = session_data.get("expires_at")
    if isinstance(expires_at, datetime) and expires_at < datetime.utcnow():
        destroy_session(session_id)
        return None

    user_id = session_data.get("user_id")
    if not user_id:
        return None

    # Fetch user
    users_col = get_collection("users")
    user_doc = None
    if users_col is not None:
        try:
            # Try ObjectId first, fallback to str
            try:
                user_doc = users_col.find_one({"_id": ObjectId(user_id)})
            except Exception:
                user_doc = users_col.find_one({"_id": user_id})
        except Exception as e:
            logger.error(f"Error finding user {user_id}: {e}")

    if not user_doc:
        for u in _in_memory_users.values():
            if str(u.get("_id")) == str(user_id):
                user_doc = u
                break

    if not user_doc:
        return None

    return sanitize_user(user_doc)


def destroy_session(session_id: str) -> bool:
    """Delete a session from database and cache."""
    if not session_id:
        return False

    sessions_col = get_collection("sessions")
    if sessions_col is not None:
        try:
            sessions_col.delete_one({"session_id": session_id})
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")

    _in_memory_sessions.pop(session_id, None)
    return True


def sanitize_user(user_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize user document for API output."""
    raw_id = user_doc.get("_id", "")
    created_at = user_doc.get("created_at")
    if isinstance(created_at, datetime):
        created_str = created_at.isoformat()
    else:
        created_str = str(created_at or "")

    return {
        "id": str(raw_id),
        "name": user_doc.get("name", "User"),
        "email": user_doc.get("email", ""),
        "role": user_doc.get("role", "user"),
        "organization": user_doc.get("organization", ""),
        "department": user_doc.get("department", ""),
        "created_at": created_str,
    }


def update_user_profile(user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update user profile fields (name, organization, department) in MongoDB and memory fallback.
    """
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID is required.",
        )
    if not isinstance(updates, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Profile updates must be an object.")

    clean_updates: Dict[str, Any] = {}
    if "name" in updates:
        raw_name = updates["name"]
        if not isinstance(raw_name, str):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Name must be text.")
        name = raw_name.strip()
        if not name or len(name) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Name must be at least 2 characters.",
            )
        if len(name) > 100:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Name must be 100 characters or fewer.")
        clean_updates["name"] = name

    if "organization" in updates:
        organization = updates["organization"] or ""
        if not isinstance(organization, str) or len(organization.strip()) > 150:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Organization must be 150 characters or fewer.")
        clean_updates["organization"] = organization.strip()

    if "department" in updates:
        department = updates["department"] or ""
        if not isinstance(department, str) or len(department.strip()) > 100:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Department must be 100 characters or fewer.")
        clean_updates["department"] = department.strip()

    if not clean_updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid fields provided for update.",
        )

    clean_updates["updated_at"] = datetime.utcnow()

    users_col = get_collection("users")
    user_doc = None

    if users_col is not None:
        try:
            try:
                query = {"_id": ObjectId(user_id)}
            except Exception:
                query = {"_id": user_id}

            users_col.update_one(query, {"$set": clean_updates})
            user_doc = users_col.find_one(query)
        except Exception as e:
            logger.error(f"Error updating user profile {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update user profile in database.",
            )

    # In-memory fallback
    for u in _in_memory_users.values():
        if str(u.get("_id")) == str(user_id):
            u.update(clean_updates)
            if not user_doc:
                user_doc = u
            break

    if not user_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    logger.info(f"User profile updated: {user_id} -> {clean_updates.get('name')}")
    return sanitize_user(user_doc)


def get_current_user_optional(request: Request) -> Optional[Dict[str, Any]]:
    """Retrieve current user from session cookie or Authorization header, or None."""
    session_token = request.cookies.get(SESSION_COOKIE_NAME)

    # Fallback to Authorization: Bearer <session_id>
    if not session_token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            session_token = auth_header.split(" ", 1)[1].strip()

    if not session_token:
        return None

    return get_user_from_session(session_token)


def get_current_user(request: Request) -> Dict[str, Any]:
    """FastAPI dependency: require authenticated user or raise 401."""
    user = get_current_user_optional(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
        )
    return user
