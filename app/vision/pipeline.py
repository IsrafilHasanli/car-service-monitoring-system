from __future__ import annotations

import logging
import os
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from app.config import Settings
from app.storage import build_id_index, load_records, save_records
from app.vision.ocr import PlateOCR, is_valid_plate

logger = logging.getLogger(__name__)


def iou_ltrb(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def nms_tlwh(dets, iou_thr=0.5):
    if not dets:
        return dets

    boxes = []
    scores = []
    for tlwh, conf, _ in dets:
        x, y, w, h = tlwh
        boxes.append([x, y, w, h])
        scores.append(float(conf))

    idxs = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_thr)
    if len(idxs) == 0:
        return []

    return [dets[i] for i in idxs.flatten().tolist()]


class ANPRPipeline:
    def __init__(self, settings: Settings):
        if not settings.rtsp_url:
            raise RuntimeError("RTSP_URL is empty. Add the camera URL to .env.")

        self.settings = settings
        self.settings.image_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vehicle_model = YOLO(str(settings.vehicle_model_path)).to(self.device)
        self.plate_model = YOLO(str(settings.plate_model_path)).to(self.device)
        self.tracker = DeepSort(
            max_age=40,
            n_init=5,
            max_iou_distance=0.5,
            nms_max_overlap=0.5,
        )
        self.plate_ocr = PlateOCR(settings.ocr_model_name)

        self.track_status: dict[int, str] = {}
        self.track_entry_ts: dict[int, str] = {}
        self.json_index_by_id: dict[int, int] = {}
        self.plate_candidates = defaultdict(list)
        self.plate_memory: dict[int, str] = {}
        self.vehicle_snapshots: dict[int, tuple[int, object]] = {}
        self.record_status_written: dict[int, str] = {}
        self.last_live_frame_write = 0.0
        self.frame_index = 0

        logger.info("Using device: %s", self.device)
        logger.info("RTSP source configured")
        logger.info(
            "Performance config: frame_skip=%s, target_width=%s, vehicle_imgsz=%s, plate_imgsz=%s, ocr_every=%s",
            settings.frame_skip,
            settings.target_width,
            settings.vehicle_img_size,
            settings.plate_img_size,
            settings.ocr_every_n_frames,
        )

    def run_forever(self) -> None:
        while True:
            cap = self._open_capture()
            if cap is None:
                time.sleep(self.settings.reconnect_delay_sec)
                continue

            try:
                self._run_capture(cap)
            finally:
                cap.release()

            logger.warning("RTSP stream ended. Reconnecting in %.1fs", self.settings.reconnect_delay_sec)
            time.sleep(self.settings.reconnect_delay_sec)

    def _open_capture(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = self.settings.rtsp_transport
        cap = cv2.VideoCapture(self.settings.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            logger.error("Could not open RTSP stream. Check RTSP_URL in .env.")
            cap.release()
            return None
        logger.info("Connected to RTSP stream")
        return cap

    def _run_capture(self, cap) -> None:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = self._resize_frame(frame)
            self._process_frame(frame)
            self._publish_live_frame(frame)
            self.frame_index += 1
            self._drop_stale_frames(cap)

    def _resize_frame(self, frame):
        h, w = frame.shape[:2]
        if self.settings.target_width and w != self.settings.target_width:
            scale = self.settings.target_width / w
            return cv2.resize(frame, (int(w * scale), int(h * scale)))
        return frame

    def _drop_stale_frames(self, cap) -> None:
        for _ in range(max(0, self.settings.frame_skip - 1)):
            if not cap.grab():
                break

    def _load_roi(self):
        if not self.settings.roi_file.exists():
            return None
        try:
            import json

            data = json.loads(self.settings.roi_file.read_text(encoding="utf-8"))
            roi = data.get("roi")
            if isinstance(roi, list) and len(roi) == 4:
                return [int(x) for x in roi]
        except Exception as e:
            logger.warning("Could not load ROI: %s", e)
        return None

    def _process_frame(self, frame) -> None:
        roi = self._load_roi()
        if not roi:
            cv2.putText(frame, "ROI is not configured", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return

        rx1, ry1, rx2, ry2 = roi
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)
        if rx2 - rx1 < 10 or ry2 - ry1 < 10:
            cv2.putText(frame, "INVALID ROI", (rx1, max(30, ry1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return

        roi_crop = frame[ry1:ry2, rx1:rx2]
        detections = self._detect_vehicles(roi_crop, rx1, ry1)
        tracks = self.tracker.update_tracks(nms_tlwh(detections), frame=frame)
        self._handle_tracks(frame, tracks, roi)

    def _detect_vehicles(self, roi_crop, rx1: int, ry1: int):
        detections = []
        results = self.vehicle_model(
            roi_crop,
            classes=[2, 3, 5, 7],
            conf=self.settings.vehicle_confidence,
            imgsz=self.settings.vehicle_img_size,
            verbose=False,
        )[0]

        for box in results.boxes:
            if float(box.conf[0]) < self.settings.vehicle_confidence:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1 + rx1, y1 + ry1, x2 - x1, y2 - y1], float(box.conf[0]), "car"))
        return detections

    def _handle_tracks(self, frame, tracks, roi) -> None:
        rx1, ry1, rx2, ry2 = roi
        roi_top_y = ry1 + int((ry2 - ry1) * self.settings.top_entry_ratio)
        roi_right_x = rx2 - self.settings.right_margin_px
        cv2.line(frame, (rx1, roi_top_y), (rx2, roi_top_y), (255, 255, 0), 2)
        cv2.line(frame, (roi_right_x, ry1), (roi_right_x, ry2), (255, 255, 0), 2)

        drawn = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            if hasattr(track, "time_since_update") and track.time_since_update > 0:
                continue

            tid = int(track.track_id)
            l, t, r, b = map(int, track.to_ltrb())
            cur_box = (l, t, r, b)
            if any(iou_ltrb(cur_box, prev) > 0.7 for prev in drawn):
                continue
            drawn.append(cur_box)
            self._remember_vehicle_snapshot(frame, tid, cur_box, roi)

            if tid not in self.track_status:
                if t > roi_top_y:
                    continue
                ts = self._now()
                self.track_status[tid] = "In Queue"
                self.track_entry_ts[tid] = ts
            elif self.plate_memory.get(tid):
                self._commit_track_if_ready(tid, self.track_status[tid])

            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 255), 2)
            cv2.putText(frame, f"ID {tid}", (l, max(20, t - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            self._detect_plate_for_track(frame, tid, cur_box)

            if self.track_status.get(tid) == "In Queue" and r >= roi_right_x:
                now_ts = self._now()
                self.track_status[tid] = "In Garage"
                self._commit_track_if_ready(tid, "In Garage", exit_ts=now_ts)

    def _detect_plate_for_track(self, frame, tid: int, box) -> None:
        l, t, r, b = box
        car_crop = frame[max(0, t):max(0, b), max(0, l):max(0, r)]
        if car_crop.size == 0:
            return

        results = self.plate_model(
            car_crop,
            conf=self.settings.plate_confidence,
            imgsz=self.settings.plate_img_size,
            verbose=False,
        )[0]

        for pbox in results.boxes:
            if float(pbox.conf[0]) < self.settings.plate_confidence:
                continue

            px1, py1, px2, py2 = map(int, pbox.xyxy[0])
            plate_crop = car_crop[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue

            text = ""
            if tid not in self.plate_memory and self.frame_index % self.settings.ocr_every_n_frames == 0:
                text = self.plate_ocr.read(plate_crop)

            if text and is_valid_plate(text):
                self.plate_candidates[tid].append(text)
                self.plate_candidates[tid] = self.plate_candidates[tid][-12:]

            self._maybe_confirm_plate(frame, plate_crop, tid, box)

            gx1, gy1 = l + px1, t + py1
            gx2, gy2 = l + px2, t + py2
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
            label = self.plate_memory.get(tid, text)
            if label:
                cv2.putText(frame, label, (gx1, max(20, gy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def _maybe_confirm_plate(self, frame, plate_crop, tid: int, box) -> None:
        cands = self.plate_candidates[tid]
        if len(cands) < 5 or tid in self.plate_memory:
            return

        counts = Counter(cands)
        best, best_n = counts.most_common(1)[0]
        second_n = counts.most_common(2)[1][1] if len(counts) > 1 else 0
        if best_n < 3 or best_n - second_n < 2:
            return

        self.plate_memory[tid] = best
        plate_path = self._save_plate_image(plate_crop, tid, best)
        self._append_plate_text(best, tid)

        if tid in self.track_status:
            self._commit_track_if_ready(
                tid,
                self.track_status.get(tid, "In Queue"),
                plate_image_path=plate_path,
            )

    def _commit_track_if_ready(
        self,
        tid: int,
        status: str,
        exit_ts: str = "",
        plate_image_path: str = "",
    ) -> None:
        plate = self.plate_memory.get(tid, "")
        snapshot = self.vehicle_snapshots.get(tid)
        if not plate or not snapshot:
            return
        if self.record_status_written.get(tid) == status and not plate_image_path:
            return

        vehicle_path = self._save_vehicle_crop(snapshot[1], tid, status)
        if not vehicle_path:
            return

        self._upsert_record(
            tid,
            self.track_entry_ts.get(tid, self._now()),
            plate,
            status,
            image_path=vehicle_path,
            exit_ts=exit_ts,
            plate_image_path=plate_image_path,
        )
        self.record_status_written[tid] = status

    def _remember_vehicle_snapshot(self, frame, track_id: int, box, roi) -> None:
        if not self._vehicle_crop_is_usable(frame, box, roi):
            return

        crop = self._crop_vehicle(frame, box)
        if crop.size == 0:
            return

        score = crop.shape[0] * crop.shape[1]
        current = self.vehicle_snapshots.get(track_id)
        if current and current[0] >= score:
            return
        self.vehicle_snapshots[track_id] = (score, crop.copy())

    def _vehicle_crop_is_usable(self, frame, box, roi) -> bool:
        l, t, r, b = box
        h, w = frame.shape[:2]
        rx1, ry1, rx2, ry2 = roi
        margin = self.settings.vehicle_edge_margin
        width = r - l
        height = b - t
        if width < self.settings.min_vehicle_width or height < self.settings.min_vehicle_height:
            return False
        if width * height < self.settings.min_vehicle_area:
            return False
        if l <= margin or t <= margin or r >= w - margin or b >= h - margin:
            return False
        if l <= rx1 + margin or t <= ry1 + margin or r >= rx2 - margin or b >= ry2 - margin:
            return False
        return True

    def _upsert_record(
        self,
        tid: int,
        ts: str,
        plate: str,
        status: str,
        image_path: str = "",
        exit_ts: str = "",
        plate_image_path: str = "",
    ) -> None:
        records = load_records()
        if not self.json_index_by_id:
            self.json_index_by_id = build_id_index(records)

        now_ts = self._now()
        if tid in self.json_index_by_id:
            rec = records[self.json_index_by_id[tid]]
            rec["timestamp"] = rec.get("timestamp") or ts
            rec["entry_time"] = rec.get("entry_time") or rec.get("timestamp") or ts
            rec["platenumber"] = plate or rec.get("platenumber", "")
            rec["status"] = status
            rec["source_type"] = "rtsp"
            rec["updated_at"] = now_ts
            if status == "In Garage":
                rec["exit_time"] = rec.get("exit_time") or exit_ts or now_ts
            else:
                rec["exit_time"] = rec.get("exit_time", "")
            if image_path:
                rec["vehicle_image_path"] = image_path
                rec["image_path"] = rec.get("image_path") or image_path
            if plate_image_path:
                rec["plate_image_path"] = plate_image_path
                rec["image_path"] = rec.get("image_path") or plate_image_path
        else:
            rec = {
                "timestamp": ts,
                "entry_time": ts,
                "exit_time": exit_ts if status == "In Garage" else "",
                "id": int(tid),
                "platenumber": plate or "",
                "status": status,
                "image_path": image_path or plate_image_path or "",
                "vehicle_image_path": image_path or "",
                "plate_image_path": plate_image_path or "",
                "source_type": "rtsp",
                "updated_at": now_ts,
            }
            records.append(rec)
            self.json_index_by_id[tid] = len(records) - 1

        save_records(records)

    def _save_plate_image(self, img, track_id: int, text: str) -> str:
        path = self.settings.image_dir / f"{self._file_ts()}_ID{track_id}_{text}.jpg"
        cv2.imwrite(str(path), img)
        return str(path.relative_to(self.settings.base_dir))

    def _save_vehicle_image(self, frame, track_id: int, status: str, box) -> str:
        crop = self._crop_vehicle(frame, box)
        if crop.size == 0:
            return ""
        return self._save_vehicle_crop(crop, track_id, status)

    def _crop_vehicle(self, frame, box):
        l, t, r, b = box
        h, w = frame.shape[:2]
        l, t = max(0, l), max(0, t)
        r, b = min(w, r), min(h, b)
        return frame[t:b, l:r]

    def _save_vehicle_crop(self, crop, track_id: int, status: str) -> str:
        if crop.size == 0:
            return ""
        safe_status = status.upper().replace(" ", "_")
        path = self.settings.image_dir / f"{self._file_ts()}_ID{track_id}_{safe_status}.jpg"
        cv2.imwrite(str(path), crop)
        return str(path.relative_to(self.settings.base_dir))

    def _append_plate_text(self, text: str, track_id: int) -> None:
        with self.settings.entered_plates_file.open("a", encoding="utf-8") as f:
            f.write(f"{text} | ID:{track_id} | {self._now()}\n")
        logger.info("Saved plate: %s (ID %s)", text, track_id)

    def _publish_live_frame(self, frame) -> None:
        now = time.monotonic()
        if now - self.last_live_frame_write < self.settings.live_frame_interval_sec:
            return

        display = frame
        h, w = display.shape[:2]
        if w > 1280:
            scale = 1280 / w
            display = cv2.resize(display, (1280, int(h * scale)))

        ok, encoded = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return

        tmp_path = Path(str(self.settings.live_frame_file) + f".{os.getpid()}.tmp")
        try:
            tmp_path.write_bytes(encoded.tobytes())
            for _ in range(3):
                try:
                    tmp_path.replace(self.settings.live_frame_file)
                    self.last_live_frame_write = now
                    return
                except PermissionError:
                    time.sleep(0.02)
        except OSError as exc:
            logger.debug("Live frame publish skipped: %s", exc)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _file_ts() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
