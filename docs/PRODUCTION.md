# Production Notes

## Deployment Checklist

- Keep `.env` outside version control.
- Deploy model weights into `models/` on the target machine.
- Confirm `config/roi.json` matches the camera angle.
- Run the app with the project virtual environment.
- Put a process manager in front of Uvicorn for long-running deployments.
- Back up `data/queue_garage.json` if service history must be retained.

## Recommended Run Command

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

For LAN-only service screens, bind to the service machine IP and restrict firewall access to trusted devices.

## Model Files

Required local files:

```text
models/yolov8n.pt
models/license_plate_detector.pt
```

Other experimental weights and demo videos are not part of production and should stay out of the repository.

## Operations

The FastAPI process controls the inference worker through API endpoints. If a stale worker remains after a crash, stop matching `python -m app.worker` processes and start inference again from the dashboard.

The TensorRT warning in the log is non-fatal when CUDA or CPU fallback is available. Install matching TensorRT libraries only if TensorRT acceleration is required.
