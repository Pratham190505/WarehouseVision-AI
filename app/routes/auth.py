"""
Authentication routes for WarehouseVision AI.

Includes:
- GET /login, GET /signup (HTML views)
- POST /api/auth/signup
- POST /api/auth/login
- POST /api/auth/logout
- GET /api/auth/me
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, Response, HTTPException, status, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.database.models import UserCreate, UserLogin
from app.services.auth_service import (
    register_user,
    authenticate_user,
    create_session,
    destroy_session,
    update_user_profile,
    get_current_user_optional,
    get_current_user,
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE_DAYS,
)

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _set_auth_cookie(response: Response, session_token: str):
    """Set secure HTTP-only session cookie."""
    max_age_seconds = SESSION_MAX_AGE_DAYS * 24 * 3600
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        max_age=max_age_seconds,
        httponly=True,
        samesite="lax",
        secure=False,  # Set to True in HTTPS production environments
        path="/",
    )


def _clear_auth_cookie(response: Response):
    """Delete the session cookie."""
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
    )


# --------------------------------------------------------------------------
# HTML Page Routes
# --------------------------------------------------------------------------

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Render login page. Redirect to dashboard if already logged in."""
    user = get_current_user_optional(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"user": None},
    )


@router.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    """Render signup page. Redirect to dashboard if already logged in."""
    user = get_current_user_optional(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(
        request=request,
        name="signup.html",
        context={"user": None},
    )


# --------------------------------------------------------------------------
# Authentication API Endpoints
# --------------------------------------------------------------------------

@router.post("/api/auth/signup")
async def api_signup(request: Request, response: Response):
    """
    Register a new user account.
    Supports both JSON body and multipart form data.
    """
    # Parse payload
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON payload.")
        name = data.get("name")
        email = data.get("email")
        password = data.get("password")
        confirm_password = data.get("confirm_password")
    else:
        form = await request.form()
        name = form.get("name")
        email = form.get("email")
        password = form.get("password")
        confirm_password = form.get("confirm_password")

    user = register_user(
        name=name,
        email=email,
        password=password,
        confirm_password=confirm_password,
    )

    # Establish session
    session_token = create_session(user["id"])

    res = JSONResponse(
        content={
            "success": True,
            "message": "Account created successfully.",
            "user": user,
            "redirect_url": "/dashboard",
        },
        status_code=status.HTTP_201_CREATED,
    )
    _set_auth_cookie(res, session_token)
    return res


@router.post("/api/auth/login")
async def api_login(request: Request, response: Response):
    """
    Authenticate user and create session cookie.
    Supports both JSON body and multipart form data.
    """
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON payload.")
        email = data.get("email")
        password = data.get("password")
    else:
        form = await request.form()
        email = form.get("email")
        password = form.get("password")

    if not email or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email and password are required.",
        )

    user = authenticate_user(email=email, password=password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    session_token = create_session(user["id"])

    res = JSONResponse(content={
        "success": True,
        "message": "Login successful.",
        "user": user,
        "redirect_url": "/dashboard",
    })
    _set_auth_cookie(res, session_token)
    return res


@router.post("/api/auth/logout")
async def api_logout(request: Request):
    """
    Log out user, destroy active session, and clear cookie.
    """
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    if session_token:
        destroy_session(session_token)

    res = JSONResponse(content={
        "success": True,
        "message": "Logged out successfully.",
        "redirect_url": "/login",
    })
    _clear_auth_cookie(res)
    return res


@router.get("/api/auth/me")
async def api_me(request: Request):
    """
    Return currently authenticated user information.
    """
    user = get_current_user(request)
    return JSONResponse(content=user)


@router.put("/api/auth/profile")
@router.post("/api/auth/profile")
async def api_update_profile(request: Request):
    """
    Update profile for the currently logged in user.
    Accepts JSON body or Form data.
    """
    current_user = get_current_user(request)
    user_id = current_user.get("id")

    body = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body.")
    else:
        form = await request.form()
        body = dict(form)

    updated_user = update_user_profile(user_id, body)
    return JSONResponse(content={
        "success": True,
        "message": "Profile updated successfully.",
        "user": updated_user,
    })
