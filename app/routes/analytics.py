"""
Analytics and history API routes for WarehouseVision AI.
Protected by user authentication: users only see their own analyses and statistics.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.services.processing_service import get_processing_service
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
