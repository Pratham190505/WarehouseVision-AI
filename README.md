# WarehouseVision AI

**AI-Powered Warehouse Vision and Analytics Platform**

WarehouseVision AI is a computer vision platform for analyzing warehouse environments. Upload images or videos to detect objects, track movements, identify restricted zone violations, and view analytics through a modern dashboard.

---

## Features

- **Object Detection** — Detect workers, forklifts, boxes, and pallets using YOLO
- **Video Tracking** — Track objects across video frames with ByteTrack
- **Restricted Zone Detection** — Configurable polygon zones with violation alerts
- **Occupancy Analysis** — Approximate space utilization estimation
- **Analytics Dashboard** — Modern SaaS-style dashboard with KPI cards, charts, and insights
- **Light/Dark Theme** — Full theme system with smooth transitions and localStorage persistence
- **Analysis History** — MongoDB-backed history with searchable records
- **Responsive Design** — Works on desktop, tablet, and mobile

---

## Architecture

```
Browser (Dashboard)
    ↓
FastAPI + Jinja2
    ↓
Vision Engine (YOLO + OpenCV)
    ↓
Analytics Engine
    ↓
MongoDB
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
| **Configuration** | python-dotenv |
| **Deployment** | Docker, Docker Compose |

---

## Folder Structure

```
WarehouseVision/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   ├── routes/
│   │   ├── upload.py              # Upload & analyze endpoints
│   │   └── analytics.py           # Analytics & history endpoints
│   ├── vision/
│   │   ├── detector.py            # YOLO object detector
│   │   ├── tracker.py             # Multi-object tracker
│   │   ├── zone_detection.py      # Restricted zone detection
│   │   ├── inventory.py           # Inventory analysis
│   │   ├── occupancy.py           # Space utilization estimation
│   │   └── analytics.py           # Analytics engine
│   ├── services/
│   │   ├── image_service.py       # Image processing pipeline
│   │   ├── video_service.py       # Video processing pipeline
│   │   └── processing_service.py  # Orchestration + MongoDB
│   ├── database/
│   │   ├── mongodb.py             # MongoDB connection
│   │   └── models.py              # Pydantic data models
│   ├── templates/
│   │   └── dashboard.html         # Dashboard template
│   └── static/
│       ├── css/
│       │   ├── themes.css         # Light/Dark theme tokens
│       │   └── dashboard.css      # Dashboard styles
│       └── js/
│           ├── theme.js           # Theme manager
│           ├── charts.js          # Chart.js manager
│           └── dashboard.js       # Dashboard controller
├── models/                        # YOLO model files
├── uploads/                       # Uploaded media
├── results/                       # Processed results
├── tests/
│   └── test_api.py
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

## API & Web Endpoints

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

### Protected Warehouse Vision & Analytics (User-Scoped)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Service & database health check |
| `POST` | `/api/upload` | Upload warehouse image or video (scoped to current user) |
| `POST` | `/api/analyze` | Direct media analysis or analyze by ID |
| `GET` | `/api/analysis/{id}` | Retrieve specific analysis result (user-scoped) |
| `GET` | `/api/analytics` | Aggregated KPIs and history for current user |
| `GET` | `/api/history` | Analysis history records for current user |

---

## Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up --build

# Run in background
docker-compose up -d --build

# Stop
docker-compose down
```

The application will be available at **http://localhost:8000**.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --tb=short
```

---

## Known Limitations

1. **Occupancy estimation** — Based on 2D bounding box coverage, not physical measurements. Requires camera calibration for accurate square-meter calculations.
2. **Default model** — Uses COCO-pretrained YOLOv8n with proxy class mappings. A custom warehouse-trained model will significantly improve accuracy.
3. **Video processing** — Processes frames sequentially; large videos may take considerable time.
4. **No authentication** — The MVP does not implement user authentication.
5. **Single-server** — Not configured for horizontal scaling or distributed processing.

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
