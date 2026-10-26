"""
MongoDB connection and database management for WarehouseVision AI.
"""

import os
import logging
from typing import Optional

from pymongo import MongoClient
from pymongo.database import Database
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def connect_db() -> Optional[Database]:
    """Connect to MongoDB. Returns the database instance or None if connection fails."""
    global _client, _db

    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB_NAME", "warehousevision")

    try:
        _client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # Verify connection
        _client.server_info()
        _db = _client[db_name]
        logger.info(f"Connected to MongoDB: {db_name}")

        # Create indexes
        _db.users.create_index("email", unique=True)
        _db.analyses.create_index("analysis_id", unique=True)
        _db.analyses.create_index("user_id")
        _db.analyses.create_index("created_at")
        _db.analyses.create_index("timestamp")
        _db.analyses.create_index("status")
        _db.sessions.create_index("session_id", unique=True)
        _db.sessions.create_index("expires_at", expireAfterSeconds=0)
        _db.safety_events.create_index("event_id", unique=True)
        _db.safety_events.create_index("user_id")
        _db.safety_events.create_index("analysis_id")
        _db.safety_events.create_index("status")
        _db.safety_events.create_index("created_at")

        return _db
    except Exception as e:
        logger.warning(f"Could not connect to MongoDB: {e}. Running without database.")
        _client = None
        _db = None
        return None


def get_db() -> Optional[Database]:
    """Get the current database instance."""
    return _db


def get_collection(name: str):
    """Get a collection from the database. Returns None if not connected."""
    if _db is None:
        return None
    return _db[name]


def close_db():
    """Close the MongoDB connection."""
    global _client, _db
    if _client:
        _client.close()
        logger.info("MongoDB connection closed.")
    _client = None
    _db = None
