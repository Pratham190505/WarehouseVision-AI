"""
Upload and analysis API routes for WarehouseVision AI.
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.services.processing_service import get_processing_service, get_media_type
from app.services.auth_service import get_current_user

load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter()

# File size limit (bytes)
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
MAX_UPLOAD_SIZE = MAX_UPLOAD_SIZE_MB * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")


@router.post("/upload")
async def upload_media(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """
    Upload a warehouse image or video file.
    Scopes the upload to the authenticated user.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # Check media type
    media_type = get_media_type(file.filename)
    if media_type is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: JPG, JPEG, PNG, MP4, AVI, MOV, MKV",
        )

    # Generate unique filename
    ext = Path(file.filename).suffix.lower()
    unique_filename = f"{uuid.uuid4()}{ext}"
    subdir = "images" if media_type == "image" else "videos"
    save_dir = UPLOAD_DIR / subdir
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / unique_filename

    # Read and save file
    try:
        contents = await file.read()
        file_size = len(contents)

        if file_size > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE_MB}MB",
            )

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file.")

        with open(save_path, "wb") as f:
            f.write(contents)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed.")

    # Create analysis record associated with user
    service = get_processing_service()
    user_id = str(user.get("id"))
    analysis_id = service.create_analysis(
        filename=unique_filename,
        original_filename=file.filename,
        media_type=media_type,
        file_size=file_size,
        media_path=str(save_path),
        user_id=user_id,
    )

    return JSONResponse(content={
        "analysis_id": analysis_id,
        "filename": file.filename,
        "media_type": media_type,
        "file_size": file_size,
        "status": "uploaded",
        "message": f"File uploaded successfully. Use /api/analyze to start analysis.",
    })


@router.post("/analyze")
async def analyze_media(
    request: Request,
    analysis_id: str = None,
    file: UploadFile = File(None),
    background_tasks: BackgroundTasks = None,
    user: dict = Depends(get_current_user),
):
    """
    Analyze uploaded warehouse media.
    Scoped to the authenticated user.
    """
    service = get_processing_service()
    user_id = str(user.get("id"))

    # If file is provided directly, upload and analyze in one step
    if file and file.filename:
        media_type = get_media_type(file.filename)
        if media_type is None:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type.",
            )

        ext = Path(file.filename).suffix.lower()
        unique_filename = f"{uuid.uuid4()}{ext}"
        subdir = "images" if media_type == "image" else "videos"
        save_dir = UPLOAD_DIR / subdir
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / unique_filename

        try:
            contents = await file.read()
            file_size = len(contents)

            if file_size > MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum: {MAX_UPLOAD_SIZE_MB}MB",
                )

            with open(save_path, "wb") as f:
                f.write(contents)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise HTTPException(status_code=500, detail="Upload failed.")

        analysis_id = service.create_analysis(
            filename=unique_filename,
            original_filename=file.filename,
            media_type=media_type,
            file_size=file_size,
            media_path=str(save_path),
            user_id=user_id,
        )

        # Process synchronously
        result = service.process_media(analysis_id, str(save_path), media_type)
        return JSONResponse(content=jsonable_encoder(result))

    # Process previously uploaded file
    if not analysis_id:
        raise HTTPException(
            status_code=400,
            detail="Provide either an analysis_id or upload a file.",
        )

    # Look up the analysis record scoped to current user
    record = service.get_analysis(analysis_id, user_id=user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found or access denied.")

    if record.get("status") == "completed":
        return JSONResponse(content=jsonable_encoder(record))

    # Get file path from record
    media_path = record.get("media_path", "")
    media_type = record.get("media_type", "image")

    if not media_path or not os.path.exists(media_path):
        raise HTTPException(status_code=404, detail="Media file not found.")

    # Process
    result = service.process_media(analysis_id, media_path, media_type)
    return JSONResponse(content=jsonable_encoder(result))
