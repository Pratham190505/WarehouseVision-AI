# WarehouseVision AI

**AI-Powered Warehouse Vision and Analytics Platform**

WarehouseVision AI is a computer vision platform for analyzing warehouse environments. Upload images or videos to detect objects, track movements, identify restricted zone violations, and view analytics through a modern dashboard.

---

## Features

- **Machine Proximity Safety Alerts** — Detect workers entering configured hazardous zones around forklifts and industrial machinery with prominent dashboard alerts
- **Object Detection** — Detect workers, forklifts, boxes, and pallets using YOLO
- **Video Tracking** — Track objects across video frames with ByteTrack maintaining persistent object IDs
- **Restricted Zone Detection** — Configurable polygon zones with violation alerts
- **Occupancy Analysis** — Approximate space utilization estimation
- **Analytics Dashboard** — Modern SaaS-style dashboard with KPI cards, safety banners, charts, and insights
- **Light/Dark Theme** — Full theme system with smooth transitions and localStorage persistence
- **Analysis History & Safety Logs** — MongoDB-backed history and dedicated `safety_events` collection with searchable records
- **User Authentication & Isolation** — Secure session management with strict user data privacy
- **Responsive Design** — Works on desktop, tablet, and mobile

---

## Architecture

```
Browser (Dashboard)
    ↓
FastAPI + Jinja2 (Auth & Routes)
    ↓
Vision Engine (YOLO + ByteTrack + Safety Zone Evaluator)
    ↓
Analytics & Safety Event Processing Service
    ↓
MongoDB (users, analyses, sessions, safety_events)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python, FastAPI, Uvicorn, Pydantic |
| **Frontend** | Jinja2, HTML5, CSS3, Vanilla JavaScript, Chart.js |
| **Computer Vision** | OpenCV, Ultralytics YOLO, NumPy |
| **Tracking** | YOLO built-in tracking (ByteTrack) |
| **Database** | MongoDB, PyMongo |
| **Configuration** | python-dotenv, Dataclasses |
| **Deployment** | Docker, Docker Compose |

---

## Folder Structure

```
WarehouseVision/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   ├── routes/
│   │   ├── auth.py                # Authentication endpoints & session management
│   │   ├── upload.py              # Upload & analyze endpoints
│   │   ├── analytics.py           # Analytics & history endpoints
│   │   └── safety.py              # Safety alerts & incident management endpoints
│   ├── vision/
│   │   ├── detector.py            # YOLO object detector
│   │   ├── tracker.py             # Multi-object tracker (ByteTrack)
│   │   ├── safety_config.py       # Machine safety thresholds & calibration configuration
│   │   ├── safety_zone.py         # Safety zone evaluator, ground-contact logic & annotations
│   │   ├── zone_detection.py      # Restricted polygon zone detection
│   │   ├── inventory.py           # Inventory analysis
│   │   ├── occupancy.py           # Space utilization estimation
│   │   └── analytics.py           # Analytics engine
│   ├── services/
│   │   ├── auth_service.py        # User authentication, password hashing & sessions
│   │   ├── image_service.py       # Image processing & safety evaluation pipeline
│   │   ├── video_service.py       # Video processing, tracking & stateful safety lifecycle
│   │   └── processing_service.py  # Orchestration, safety event persistence & MongoDB
│   ├── database/
│   │   ├── mongodb.py             # MongoDB connection & index initialization
│   │   └── models.py              # Pydantic data models (SafetyEventRecord, AnalysisRecord)
│   ├── templates/
│   │   ├── login.html             # User login template
│   │   ├── signup.html            # User registration template
│   │   └── dashboard.html         # Dashboard template with Machine Safety section
│   └── static/
│       ├── css/
│       │   ├── themes.css         # Light/Dark theme tokens & safety alert variables
│       │   ├── auth.css           # Authentication styling
│       │   └── dashboard.css      # Dashboard styles & safety alert banners
│       └── js/
│           ├── theme.js           # Theme manager
│           ├── charts.js          # Chart.js manager
│           ├── auth.js            # Auth forms controller
│           └── dashboard.js       # Dashboard controller & safety alerts polling/rendering
├── models/                        # YOLO model files
├── uploads/                       # Uploaded media (images/videos)
├── results/                       # Processed results & annotated media
├── tests/
│   ├── test_api.py                # API, authentication & isolation tests
│   └── test_safety.py             # Machine proximity safety alert test suite
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- MongoDB 6+ (local or remote)
- Git

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd WarehouseVision

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### MongoDB Setup

Install and start MongoDB locally:

```bash
# macOS (Homebrew)
brew install mongodb-community
brew services start mongodb-community

# Ubuntu
sudo apt install mongodb
sudo systemctl start mongod

# Windows
# Download from https://www.mongodb.com/try/download/community
# Run as a service or start mongod.exe
```

Or use Docker:

```bash
docker run -d --name warehousevision-mongo -p 27017:27017 mongo:7
```

### Environment Variables

```bash
# Copy the example and edit as needed
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB_NAME` | `warehousevision` | Database name |
| `MODEL_PATH` | `models/warehouse_model.pt` | Path to YOLO model |
| `UPLOAD_DIR` | `uploads` | Upload storage directory |
| `RESULT_DIR` | `results` | Result storage directory |
| `MAX_UPLOAD_SIZE_MB` | `500` | Maximum upload size in MB |

### Model Setup

By default, the application uses a pretrained YOLOv8n model (COCO classes mapped to warehouse categories). For production use:

1. Train a custom YOLO model with warehouse-specific classes
2. Place the `.pt` file in the `models/` directory
3. Update `MODEL_PATH` in `.env`
4. Update class mappings in `app/vision/detector.py`

---

## Running Locally

```bash
# Start the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open the dashboard at: **http://localhost:8000**

---

---

## AI-Powered Machine Proximity Safety Alerts

### 1. Overview & Objective
Warehouses are high-risk industrial environments where forklifts and heavy machinery operate in close proximity to floor personnel. The **Machine Proximity Safety Alerts** feature provides automated computer vision assistance to detect when a worker enters the configured danger zone around a hazardous machine (e.g., forklift, reach truck, industrial loader) and alerts operators via high-visibility red dashboard banners and annotated media.

> **CRITICAL INDUSTRIAL SAFETY DISCLAIMER**
> This feature is an **AI assistance and monitoring prototype tool**. It does **NOT** constitute a certified industrial safety system (e.g., ISO 13849 / IEC 61508 / OSHA certified safety interlock). It must never replace certified safety procedures, physical barriers, pedestrian walkways, spotters, or personal protective equipment (PPE). The system does not claim to prevent workplace accidents.

---

### 2. Implementation Architecture & Tech Stack

The feature was implemented as a non-breaking, native extension to the existing WarehouseVision architecture:

```
Media Upload (Image / Video)
         │
         ▼
YOLOv8 Object Detector (Classes: worker/person, forklift, truck, machinery)
         │
         ▼
ByteTrack Multi-Object Tracker (Stable object IDs across video frames)
         │
         ▼
SafetyZoneEvaluator (app/vision/safety_zone.py)
   ├── Ground Contact Point Estimation (bottom-center anchor)
   ├── Proximity Assessment:
   │     ├─ Calibrated: Ground-plane homography / Pixels-per-meter (m)
   │     └─ Uncalibrated: Ground ellipse footprint / Configurable polygons
   └── Stateful Incident Lifecycle (persistence, cooldown, anti-flapping)
         │
         ├──► OpenCV Visual Annotation (highlighted red boxes, zones, lines)
         ├──► MongoDB `safety_events` Collection (user-isolated event logs)
         └──► REST API & Dashboard (prominent red banners & incident table)
```

**Technologies Used:**
- **OpenCV (`cv2`)**: Ground-plane ellipse drawing, polygon containment testing, transparency blending, and media annotation.
- **YOLOv8 & ByteTrack (`ultralytics`)**: Worker and machine detection with persistent multi-object tracking IDs.
- **FastAPI**: Authenticated REST API endpoints scoped to user sessions.
- **Pydantic V2**: Schema models (`SafetyEventRecord`, `SafetyIncident`).
- **MongoDB**: Event persistence with indexed queries in `safety_events` collection.
- **Vanilla JavaScript & CSS**: Dynamic dashboard polling, active alert banners, acknowledge actions, and theme styling (light and dark modes).

---

### 3. Proximity Estimation Methods & Metric Integrity

A core safety requirement is that **image pixels cannot be claimed to represent physical metres without camera calibration**. The system strictly adheres to this principle:

#### A. Uncalibrated Mode (Default)
When no camera calibration parameters are supplied:
- The system evaluates proximity using an **image-space ground ellipse** around the base of the machine (expanded by `image_space_expansion_ratio`, default: 2.2x).
- Alternatively, operators can define custom **polygon danger zones** (e.g. blind spots, loading dock bays).
- **Metric Integrity**: The system **refuses to fabricate or display numeric distances in metres**. Media overlays and API outputs clearly state: `Image-space proximity zone` or `Worker inside machine perimeter`.

#### B. Calibrated Camera Mode (Optional)
When calibrated camera parameters are configured in `SafetyConfig`:
- **Homography Matrix**: Maps image coordinates `(u, v)` to world ground coordinates `(X, Y)` in metres using `cv2.perspectiveTransform`.
- **Pixels-per-Meter (PPM)**: For fixed, top-down or ortho-rectified overhead cameras, converts Euclidean ground distance: $\text{distance} = \frac{\Delta\text{pixels}}{\text{PPM}}$.
- Distance in metres is reported only when calibration is valid and verified.

---

### 4. Ground Contact Point Calculation
Traditional bounding box centers can cause severe false alarms in perspective views (e.g., worker's head overlapping a distant forklift's mast). WarehouseVision solves this by computing the **ground contact point**:
$$\text{Point}_{\text{ground}} = \left(\frac{x_1 + x_2}{2}, y_2\right)$$
This represents the physical contact point with the warehouse floor plane, ensuring reliable spatial evaluation.

---

### 5. Stateful Incident Lifecycle & Anti-Flapping

To avoid alert flapping, false triggers, and database spam across video frames, the `SafetyZoneEvaluator` maintains an incident state machine:

1. **Detection & Persistence Verification**: A breach must persist for `persistence_frames` (default: 2 consecutive frames) before triggering an active incident.
2. **Single Persistent Database Event**: For a continuous incident, a single MongoDB record is created and continuously updated with `last_detected_at`, `last_frame`, and `min_estimated_distance_m`. Duplicate records are **not** created per frame.
3. **Hazard Resolution**: When the worker exits the machine's safety zone, the incident transitions to `RESOLVED` with timestamp `resolved_at`.
4. **Alert Cooldown**: A cooldown period (`cooldown_seconds`, default: 3.0s) prevents immediate duplicate alerts if a worker briefly oscillates near the boundary.
5. **Physical Safety vs. Operator Acknowledgement**:
   - Acknowledging an alert records `is_acknowledged = true`, `acknowledged_at`, and `acknowledged_by`.
   - **Crucially, acknowledging does NOT mark an ongoing hazard as safe or resolved**. The event remains physically `ACTIVE` until the worker physically leaves the zone.

---

### 6. Processed Media Visual Annotations

When media is processed, visual indicators are overlaid frame-by-frame:
- **Green/Neutral Bounding Boxes**: Normal workers outside hazardous zones.
- **Cyan/Yellow Danger Zone**: Elliptical danger perimeter projected at the machine's ground contact base.
- **Bold Red Bounding Box**: Worker detected inside the machine's danger zone.
- **Red Hazard Connector Line**: Line drawn directly between the worker's ground contact point and the machine's base.
- **Prominent Machine Status Header**: Red status banner displaying machine ID, worker ID, proximity method, and warning text without fabricated numbers.

---

### 7. Configuration (`app/vision/safety_config.py`)

Safety settings are configured via `SafetyConfig` (with per-machine overrides and optional calibration):

```python
from app.vision.safety_config import SafetyConfig

config = SafetyConfig(
    default_threshold_m=2.0,                  # Prototype initial threshold
    machine_thresholds={
        "forklift": 2.5,                      # Forklift threshold override
        "truck": 4.0,                         # Truck threshold override
        "machinery": 2.0,                     # General machinery threshold
    },
    machine_id_overrides={"03": 3.0},         # Specific machine ID override
    persistence_frames=2,                     # Anti-flapping frame threshold
    cooldown_seconds=3.0,                     # Incident re-trigger cooldown
    image_space_expansion_ratio=2.2,          # Base ellipse expansion factor
    camera_calibration={"enabled": False},    # Set to True only with validated parameters
)
```

---

### 8. API & Web Endpoints

### Authentication & Pages
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/login` | User login view |
| `GET` | `/signup` | User registration view |
| `GET` | `/dashboard` | Protected warehouse analytics dashboard (requires login) |
| `GET` | `/` | Root redirect (redirects to `/dashboard` or `/login`) |
| `POST` | `/api/auth/signup` | Register new user account (bcrypt password hashing) |
| `POST` | `/api/auth/login` | Authenticate user & set secure session cookie |
| `POST` | `/api/auth/logout` | Terminate session & clear cookie |
| `GET` | `/api/auth/me` | Retrieve currently authenticated user profile |
| `PUT/POST` | `/api/auth/profile` | Update user name, department, or organization |

### Machine Safety Alerts (User-Scoped)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/safety/alerts` | Get currently active safety alerts for the authenticated user |
| `GET` | `/api/safety/events` | List historical safety events with status and machine filters |
| `GET` | `/api/safety/events/{event_id}` | Retrieve details for a specific safety event (verifies ownership) |
| `POST` | `/api/safety/events/{event_id}/acknowledge` | Acknowledge safety alert (does not clear physical active hazard) |
| `GET` | `/api/safety/config` | View configured machine thresholds, classes, and calibration status |

### Warehouse Vision & Media Processing (User-Scoped)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Service & database health check |
| `POST` | `/api/upload` | Upload warehouse image or video (scoped to current user) |
| `POST` | `/api/analyze` | Direct media analysis or analyze by ID (integrates safety evaluation) |
| `GET` | `/api/analysis/{id}` | Retrieve specific analysis result and safety alerts (user-scoped) |
| `GET` | `/api/analytics` | Aggregated KPIs, safety counters, and history for current user |
| `GET` | `/api/history` | Analysis history records for current user |

---

## Testing

The project includes unit and integration tests covering API flows, user isolation, and all 11 machine proximity safety requirements:

```bash
# Run all tests (API, Auth, Analytics, Safety)
E:\python.exe -m pytest -v

# Run the dedicated safety alert test suite
E:\python.exe -m pytest tests/test_safety.py -v
```

**Verified Test Scenarios (`tests/test_safety.py`):**
1. `test_worker_outside_safety_zone` — Worker far from forklift generates zero alerts.
2. `test_worker_entering_safety_zone` — Worker entering machine perimeter triggers active alert.
3. `test_worker_leaving_safety_zone_in_video` — Incident state transitions to `RESOLVED` upon exit.
4. `test_repeated_frames_without_duplicate_events` — Consecutive breach frames maintain single incident.
5. `test_different_machine_specific_thresholds` — Machine-specific overrides and machine-type thresholds work.
6. `test_missing_camera_calibration_does_not_fabricate_meters` — Uncalibrated mode does NOT fabricate metres.
7. `test_calibrated_camera_computes_metric_distance` — Calibrated camera correctly computes distance in metres.
8. `test_invalid_or_unsupported_detections_handled_gracefully` — Engine tolerates malformed data gracefully.
9. `test_safety_config_endpoint` — Configuration endpoint returns thresholds and classes.
10. `test_mongodb_event_persistence_and_alerts_retrieval` — Incidents correctly persist in `safety_events`.
11. `test_user_isolation_on_safety_events` — User B receives 404 Access Denied on User A's safety events.
12. `test_alert_acknowledgement_while_hazard_remains_active` — Acknowledged event remains physically `ACTIVE`.
13. `test_end_to_end_image_analysis_pipeline_with_safety` — Upload and analysis flow runs end-to-end.
14. `test_theme_css_tokens_present` — CSS tokens present for light and dark themes.
15. `test_dashboard_css_safety_classes_present` — Dashboard CSS classes exist.

---

## Known Limitations

1. **Uncalibrated Cameras** — Without extrinsic/intrinsic camera calibration, distances in metres cannot be physically measured. Image-space zones are prototype geometric approximations.
2. **Pretrained Model Extensibility** — The default configuration maps standard COCO classes (person $\rightarrow$ worker, truck/bus/car $\rightarrow$ forklift/machinery). For maximum industrial accuracy, deploy a custom YOLO model trained on target warehouse equipment and register new classes in `SafetyConfig`.
3. **Occlusions** — Dense warehouse racking can cause visual occlusion of worker feet. Ground contact estimation assumes the bottom of the bounding box touches the ground plane.
4. **Prototype Status** — Intended for computer vision assistance and situational awareness; not certified for life-critical automation.

---

## Future Improvements

- [ ] Custom-trained warehouse YOLO model
- [ ] User authentication and role-based access
- [ ] Real-time video stream processing (RTSP)
- [ ] WebSocket-based live analysis updates
- [ ] Multi-camera support
- [ ] Heatmap generation for worker/forklift paths
- [ ] Automated alert system (email/Slack)
- [ ] Report generation (PDF/CSV export)
- [ ] GPU-accelerated inference
- [ ] Horizontal scaling with Celery workers
- [ ] Advanced inventory management integration

---

## License

This project is for internal/demonstration use. See LICENSE file for details.
