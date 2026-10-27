"""
WarehouseVision AI - Main FastAPI Application

AI-Powered Warehouse Vision and Analytics Platform.
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.database.mongodb import connect_db, close_db
from app.routes.upload import router as upload_router
from app.routes.analytics import router as analytics_router
from app.routes.auth import router as auth_router
from app.routes.safety import router as safety_router
from app.routes.box_tracking import router as box_tracking_router
from app.services.auth_service import get_current_user_optional

# Load environment variables
load_dotenv(override=True)

# Ensure directories exist
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
RESULT_DIR = BASE_DIR / os.getenv("RESULT_DIR", "results")

for d in [
    UPLOAD_DIR / "images",
    UPLOAD_DIR / "videos",
    RESULT_DIR / "images",
    RESULT_DIR / "videos",
]:
    d.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: connect to DB on startup, close on shutdown."""
    connect_db()
    yield
    close_db()


app = FastAPI(
    title="WarehouseVision AI",
    description="AI-Powered Warehouse Vision and Analytics Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & templates
STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/results", StaticFiles(directory=str(RESULT_DIR)), name="results")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Include routers
app.include_router(auth_router)
app.include_router(upload_router, prefix="/api")
app.include_router(analytics_router, prefix="/api")
app.include_router(safety_router, prefix="/api")
app.include_router(box_tracking_router, prefix="/api")


@app.get("/")
async def root_redirect(request: Request):
    """
    Root redirect: if logged in -> /dashboard, else -> /login.
    """
    user = get_current_user_optional(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=303)
    return RedirectResponse(url="/login", status_code=303)


@app.get("/dashboard")
@app.get("/analysis")
@app.get("/safety")
@app.get("/box-counting")
async def dashboard_view(request: Request):
    """Serve the shared, authenticated application shell for each main view."""
    user = get_current_user_optional(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"user": user},
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    from app.database.mongodb import get_db

    db_status = "connected"
    try:
        db = get_db()
        if db is not None:
            db.command("ping")
        else:
            db_status = "not connected"
    except Exception:
        db_status = "error"

    return {
        "status": "healthy",
        "service": "WarehouseVision AI",
        "version": "1.0.0",
        "database": db_status,
    }
