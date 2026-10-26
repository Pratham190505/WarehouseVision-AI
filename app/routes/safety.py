"""
Safety API Routes for WarehouseVision AI.

Provides endpoints for:
- Retrieving active safety proximity alerts
- Listing safety events history
- Querying a single safety event by ID
- Acknowledging a safety event
- Inspecting active safety zone configuration

All endpoints are protected by user authentication to ensure strict privacy isolation.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.services.processing_service import get_processing_service
from app.services.auth_service import get_current_user
from app.vision.safety_config import get_safety_config

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/safety/health")
async def safety_health():
    """Health check for safety subservice."""
    return {"status": "ok", "service": "safety_monitoring"}


@router.get("/safety/config")
async def get_config(user: dict = Depends(get_current_user)):
    """
    Get current machine safety configuration (thresholds, classes, calibration status).
    """
    config = get_safety_config()
    return JSONResponse(content=jsonable_encoder(config.to_dict()))


@router.get("/safety/summary")
async def get_safety_summary(
    user: dict = Depends(get_current_user),
):
    """
    Get safety analytics summary (active, total, resolved counts) scoped strictly to authenticated user.
    """
    service = get_processing_service()
    user_id = str(user.get("id"))
    summary = service.get_safety_summary(user_id=user_id)
    return JSONResponse(content=jsonable_encoder(summary))


@router.get("/safety/alerts")
async def get_active_alerts(
    user: dict = Depends(get_current_user),
):
    """
    Get active safety alerts for the authenticated user.
    Alerts remain ACTIVE as long as a worker is inside a machine's danger zone.
    """
    service = get_processing_service()
    user_id = str(user.get("id"))
    alerts = service.get_active_safety_alerts(user_id=user_id)
    return JSONResponse(content=jsonable_encoder({"alerts": alerts, "count": len(alerts)}))


@router.get("/safety/events")
async def get_safety_events(
    limit: int = Query(default=50, ge=1, le=200),
    status: Optional[str] = Query(default=None),
    analysis_id: Optional[str] = Query(default=None),
    user: dict = Depends(get_current_user),
):
    """
    Get safety events history scoped to current user.
    Supports filtering by status (ACTIVE, RESOLVED) and analysis_id.
    """
    service = get_processing_service()
    user_id = str(user.get("id"))
    events = service.get_safety_events(
        user_id=user_id,
        status=status,
        analysis_id=analysis_id,
        limit=limit,
    )
    return JSONResponse(content=jsonable_encoder({"events": events, "count": len(events)}))


@router.get("/safety/events/{event_id}")
async def get_safety_event(
    event_id: str,
    user: dict = Depends(get_current_user),
):
    """
    Get details of a specific safety event by ID.
    Enforces user isolation: users cannot access events belonging to another account.
    """
    service = get_processing_service()
    user_id = str(user.get("id"))
    event = service.get_safety_event(event_id, user_id=user_id)

    if event is None:
        raise HTTPException(status_code=404, detail="Safety event not found or access denied.")

    return JSONResponse(content=jsonable_encoder(event))


@router.post("/safety/events/{event_id}/acknowledge")
async def acknowledge_safety_event(
    event_id: str,
    user: dict = Depends(get_current_user),
):
    """
    Acknowledge a safety alert.
    CRITICAL SAFETY REQUIREMENT:
    Acknowledging records that a safety supervisor has seen the notification,
    but does NOT mark an ongoing physical hazard as safe. The status remains ACTIVE
    until the worker leaves the machine's safety zone.
    """
    service = get_processing_service()
    user_id = str(user.get("id"))
    user_name = user.get("name", "User")

    updated = service.acknowledge_safety_event(
        event_id=event_id,
        user_id=user_id,
        acknowledged_by=user_name,
    )

    if updated is None:
        raise HTTPException(status_code=404, detail="Safety event not found or access denied.")

    return JSONResponse(content=jsonable_encoder({
        "success": True,
        "message": "Safety alert acknowledged.",
        "event": updated,
    }))
