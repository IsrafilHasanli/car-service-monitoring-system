from __future__ import annotations

import re

import cv2
from fast_plate_ocr import ONNXPlateRecognizer

PLATE_REGEX = re.compile(r"^\d{2}[A-Z]{2}\d{3}$")


def is_valid_plate(text: str) -> bool:
    return bool(PLATE_REGEX.match(text or ""))


def clean_ocr_text(text: str) -> str:
    text = (text or "").upper()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^A-Z0-9]", "", text)


def normalize_plate_candidate(text: str) -> str:
    cleaned = clean_ocr_text(text)
    if len(cleaned) < 7:
        return ""

    digit_map = {
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "L": "1", "T": "7",
        "Z": "2", "S": "5", "B": "8", "G": "6",
    }
    letter_map = {
        "0": "O", "1": "I", "2": "Z", "3": "B", "4": "A",
        "5": "S", "6": "G", "7": "T", "8": "B", "9": "G",
    }

    candidates = [cleaned[i:i + 7] for i in range(0, len(cleaned) - 6)]
    if len(cleaned) == 7:
        candidates.insert(0, cleaned)

    for raw in candidates:
        chars = list(raw)
        for i in (0, 1, 4, 5, 6):
            chars[i] = digit_map.get(chars[i], chars[i])
        for i in (2, 3):
            chars[i] = letter_map.get(chars[i], chars[i])
        candidate = "".join(chars)
        if is_valid_plate(candidate):
            return candidate

    return ""


class PlateOCR:
    def __init__(self, model_name: str):
        self._recognizer = ONNXPlateRecognizer(model_name)

    def read(self, plate_crop) -> str:
        if plate_crop is None or plate_crop.size == 0:
            return ""

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        target_h = 96
        if h < target_h:
            scale = target_h / max(1, h)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        gray = cv2.bilateralFilter(gray, 7, 35, 35)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        pred = self._recognizer.run(gray[..., None])

        if isinstance(pred, (list, tuple)):
            if not pred:
                return ""
            pred = pred[0]

        return normalize_plate_candidate(str(pred))
