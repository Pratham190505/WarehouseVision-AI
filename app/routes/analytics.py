"""
Analytics and history API routes for WarehouseVision AI.
Protected by user authentication: users only see their own analyses and statistics.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse

from app.services.processing_service import get_processing_service, RESULT_DIR
from app.services.auth_service import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/analytics/health")
async def analytics_health():
    """Health check for analytics subservice."""
    return {"status": "ok", "service": "analytics"}


@router.get("/analysis/{analysis_id}")
async def get_analysis(
    analysis_id: str,
    user: dict = Depends(get_current_user),
):
    """
    Get a specific analysis result by ID.
    Guarantees user cannot view another user's analysis.
    """
    service = get_processing_service()
    user_id = str(user.get("id"))
    result = service.get_analysis(analysis_id, user_id=user_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Analysis not found or access denied.")

    return JSONResponse(content=jsonable_encoder(result))


@router.delete("/analysis/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    user: dict = Depends(get_current_user),
):
    """
    Delete a specific analysis result by ID and associated safety events.
    Guarantees user cannot delete another user's analysis.
    """
    service = get_processing_service()
    user_id = str(user.get("id"))
    success = service.delete_analysis(analysis_id, user_id=user_id)

    if not success:
        raise HTTPException(status_code=404, detail="Analysis not found or access denied.")

    return JSONResponse(content={"success": True, "message": "Analysis deleted successfully."})


@router.get("/analytics")
async def get_analytics(
    limit: int = Query(default=50, ge=1, le=200),
    user: dict = Depends(get_current_user),
):
    """
    Get analytics data: dashboard stats + analysis history scoped to current user.
    """
    service = get_processing_service()
    user_id = str(user.get("id"))

    stats = service.get_dashboard_stats(user_id=user_id)
    history = service.get_history(user_id=user_id, limit=limit)

    return JSONResponse(content=jsonable_encoder({
        "stats": stats,
        "history": history,
    }))


@router.get("/history")
async def get_history(
    limit: int = Query(default=50, ge=1, le=200),
    user: dict = Depends(get_current_user),
):
    """
    Get analysis history records scoped to current user.
    """
    service = get_processing_service()
    user_id = str(user.get("id"))
    history = service.get_history(user_id=user_id, limit=limit)
    return JSONResponse(content=jsonable_encoder({"history": history}))

@router.get("/analysis/{analysis_id}/video")
async def get_analysis_video(
    analysis_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """
    Serve processed video securely with HTTP byte-range support for browser playback.
    Strictly scoped to the authenticated user owning the analysis.
    """
    service = get_processing_service()
    user_id = str(user.get("id"))
    record = service.get_analysis(analysis_id, user_id=user_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found or access denied.")

    videos_dir = (RESULT_DIR / "videos").resolve()
    target_path = None

    # Check video_filename
    video_filename = record.get("video_filename")
    if video_filename:
        candidate = (videos_dir / video_filename).resolve()
        if candidate.is_file() and candidate.is_relative_to(videos_dir):
            target_path = candidate

    # Check result_path or processed_video_url
    if not target_path:
        for key in ("result_path", "processed_video_url"):
            val = record.get(key, "")
            if val and ("/videos/" in val or val.endswith(".mp4")):
                fname = Path(val).name
                candidate = (videos_dir / fname).resolve()
                if candidate.is_file() and candidate.is_relative_to(videos_dir):
                    target_path = candidate
                    break

    # Standard naming conventions
    if not target_path:
        for candidate_name in [f"{analysis_id}_processed.mp4", f"{analysis_id}_web.mp4"]:
            candidate = (videos_dir / candidate_name).resolve()
            if candidate.is_file() and candidate.is_relative_to(videos_dir):
                target_path = candidate
                break

    if not target_path or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Processed video file not found.")

    return FileResponse(
        path=str(target_path),
        media_type="video/mp4",
        filename=target_path.name,
        content_disposition_type="inline",
    )


@router.get("/analysis/{analysis_id}/thumbnail")
async def get_analysis_thumbnail(
    analysis_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """
    Serve processed thumbnail securely scoped to authenticated owner.
    """
    service = get_processing_service()
    user_id = str(user.get("id"))
    record = service.get_analysis(analysis_id, user_id=user_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found or access denied.")

    videos_dir = (RESULT_DIR / "videos").resolve()
    target_path = None
    thumb_path = record.get("thumbnail_path", "")
    if thumb_path and ("/videos/" in thumb_path or thumb_path.endswith(".jpg")):
        candidate = (videos_dir / Path(thumb_path).name).resolve()
        if candidate.is_file() and candidate.is_relative_to(videos_dir):
            target_path = candidate

    if not target_path:
        candidate = (videos_dir / f"{analysis_id}_thumb.jpg").resolve()
        if candidate.is_file() and candidate.is_relative_to(videos_dir):
            target_path = candidate

    if not target_path or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Thumbnail not found.")

    return FileResponse(
        path=str(target_path),
        media_type="image/jpeg",
        filename=target_path.name,
        content_disposition_type="inline",
    )

