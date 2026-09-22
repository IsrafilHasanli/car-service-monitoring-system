# Car Service Monitoring System

FastAPI based ANPR service-lane monitor for a car service camera. The app reads an RTSP stream, runs vehicle tracking, plate detection, OCR, and exposes a live operations dashboard.

## Project Structure

```text
app/
  config.py              Environment-driven settings
  storage.py             JSON storage and image resolution helpers
  web.py                 FastAPI routes and dashboard
  worker.py              Background inference entrypoint
  services/inference.py  Worker process lifecycle
  vision/ocr.py          Plate OCR normalization
  vision/pipeline.py     RTSP, detection, tracking, OCR pipeline
config/
  roi.json               Camera ROI configuration
data/
  images/                Runtime captures, ignored by git
models/
  yolov8n.pt             Vehicle detector
  license_plate_detector.pt
main.py                  ASGI app entrypoint
```

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and set `RTSP_URL`.

3. Place model files in `models/`:

```text
models/yolov8n.pt
models/license_plate_detector.pt
```

4. Start the web app.

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The dashboard starts inference automatically when `RTSP_URL` is configured.

## Runtime Data

Runtime files are written under `data/` and are intentionally ignored by git:

- `data/queue_garage.json`
- `data/entered_plates.txt`
- `data/anpr_inference.log`
- `data/live_frame.jpg`
- `data/images/*.jpg`

Records are only created when the pipeline has a valid plate read and a usable vehicle crop. This prevents empty `Reading...` rows from appearing in the dashboard.

## API

- `GET /` - Dashboard
- `GET /api/queue` - Service lane records
- `GET /api/inference/status` - Worker status
- `POST /api/inference/start` - Start RTSP inference
- `POST /api/inference/stop` - Stop RTSP inference
- `GET /live-frame` - Latest live frame JPEG
- `GET /image-file?name=...` - Runtime capture file

## Configuration

All runtime configuration is loaded from `.env`. Important variables:

- `RTSP_URL`
- `ANPR_ROI_FILE`
- `ANPR_VEHICLE_MODEL`
- `ANPR_PLATE_MODEL`
- `ANPR_FRAME_SKIP`
- `ANPR_TARGET_WIDTH`
- `ANPR_OCR_EVERY_N_FRAMES`
- `ANPR_MIN_VEHICLE_WIDTH`
- `ANPR_MIN_VEHICLE_HEIGHT`
- `ANPR_MIN_VEHICLE_AREA`

See `.env.example` for the full list.
