"""Authenticated recorded-video unique box tracking endpoints."""

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.services.auth_service import get_current_user
from app.services.processing_service import BOX_TRACKING_RESULT_DIR, get_processing_service

router = APIRouter(prefix="/box-tracking", tags=["Box Tracking"])


def _owned_output_path(analysis_id: str, user: dict, key: str):
    record = get_processing_service().get_analysis(analysis_id, user_id=str(user.get("id")))
    if not record or not record.get("box_tracking"):
        raise HTTPException(status_code=404, detail="Box tracking result not found.")
    filename = record["box_tracking"].get(key)
    if not filename or Path(filename).name != filename:
        raise HTTPException(status_code=404, detail="Requested output is unavailable.")
    output_path = (BOX_TRACKING_RESULT_DIR / filename).resolve()
    if output_path.parent != BOX_TRACKING_RESULT_DIR.resolve() or not output_path.is_file():
        raise HTTPException(status_code=404, detail="Requested output is unavailable.")
    return output_path


@router.post("/{analysis_id}/analyze")
def analyze_box_tracking(analysis_id: str, user: dict = Depends(get_current_user)):
    """Start a fresh, independent unique-track session for an owned uploaded video."""
    service = get_processing_service()
    user_id = str(user.get("id"))
    record = service.get_analysis(analysis_id, user_id=user_id)
    if not record or record.get("media_type") != "video":
        raise HTTPException(status_code=404, detail="Uploaded video not found.")
    media_path = record.get("media_path", "")
    if not media_path or not os.path.isfile(media_path):
        raise HTTPException(status_code=404, detail="Uploaded video file is unavailable.")

    session_id = service.create_analysis(
        filename=record.get("filename", Path(media_path).name),
        original_filename=record.get("original_filename", Path(media_path).name),
        media_type="video",
        file_size=record.get("file_size", 0),
        media_path=media_path,
        user_id=user_id,
    )
    result = service.process_media(
        session_id,
        media_path,
        "video",
        box_tracking_config={},
    )
    if result.get("status") != "completed":
        raise HTTPException(status_code=422, detail=result.get("error_message", "Box tracking failed."))
    return result


@router.get("/{analysis_id}/result")
def get_processed_video(analysis_id: str, user: dict = Depends(get_current_user)):
    path = _owned_output_path(analysis_id, user, "output_filename")
    return FileResponse(path, media_type="video/mp4", filename=path.name, content_disposition_type="inline")


@router.get("/{analysis_id}/thumbnail")
def get_processed_thumbnail(analysis_id: str, user: dict = Depends(get_current_user)):
    path = _owned_output_path(analysis_id, user, "thumbnail_filename")
    return FileResponse(path, media_type="image/jpeg", filename=path.name, content_disposition_type="inline")
