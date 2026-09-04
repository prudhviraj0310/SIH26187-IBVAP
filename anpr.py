"""
Automatic Number Plate Recognition (ANPR) for Indian Checkpoint Vehicles.
Extracts and validates Indian RTO civilian & military registration formats.
"""

import re
import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple

# Indian State RTO Codes
VALID_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD",
    "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB", "BH"
}

# Regex for standard Indian civilian plate: e.g. DL 01 AB 1234, MH12DE1433
INDIAN_CIVILIAN_PATTERN = re.compile(r'([A-Z]{2})\s*([0-9]{1,2})\s*([A-Z]{0,3})\s*([0-9]{4})')
# Bharat Series: 22 BH 1234 AA
BHARAT_SERIES_PATTERN = re.compile(r'([0-9]{2})\s*(BH)\s*([0-9]{4})\s*([A-Z]{1,2})')
# Military Vehicle: ↑ 21D 123456
MILITARY_PATTERN = re.compile(r'(?:↑|\^)?\s*([0-9]{2})\s*([A-Z])\s*([0-9]{6})')

class ANPREngine:
    def __init__(self):
        # Local mock blacklist of flagged vehicles (stolen/smuggling/wanted)
        self.watchlist_plates = {
            "DL01AB1234": "Suspect vehicle - Smuggling watch (SSB Post 12)",
            "MH12DE1433": "Unregistered dumper - Unauthorized border transit",
            "UP14BC9999": "Flagged by Delhi Police Special Cell (FIR #281/26)"
        }
        self.scanned_records: List[Dict] = []

    def clean_text(self, raw_text: str) -> str:
        """Strip non-alphanumeric noise."""
        return re.sub(r'[^A-Z0-9]', '', raw_text.upper())

    def validate_and_format_plate(self, raw_str: str) -> Optional[Tuple[str, str]]:
        """
        Validates whether raw_str matches an Indian registration format.
        Returns: (formatted_plate_number, plate_type) or None
        """
        clean = self.clean_text(raw_str)
        if len(clean) < 8 or len(clean) > 12:
            return None

        # Check standard civilian plate
        match_civ = INDIAN_CIVILIAN_PATTERN.search(clean)
        if match_civ:
            state, dist, series, num = match_civ.groups()
            if state in VALID_STATE_CODES:
                formatted = f"{state} {dist.zfill(2)} {series} {num}".strip()
                return formatted, "CIVILIAN"

        # Check Bharat series
        match_bh = BHARAT_SERIES_PATTERN.search(clean)
        if match_bh:
            year, bh, num, series = match_bh.groups()
            formatted = f"{year} BH {num} {series}"
            return formatted, "BHARAT_SERIES"

        # Check military format
        match_mil = MILITARY_PATTERN.search(clean)
        if match_mil:
            yr, code, num = match_mil.groups()
            formatted = f"↑ {yr}{code} {num}"
            return formatted, "MILITARY_DEFENCE"

        return None

    def locate_plate_region(self, vehicle_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Locates the high-probability license plate rectangular candidate inside vehicle crop.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        h, w = vehicle_crop.shape[:2]
        # Plates are usually in the lower half of the vehicle
        lower_crop = vehicle_crop[int(h * 0.4):, :]

        gray = cv2.cvtColor(lower_crop, cv2.COLOR_BGR2GRAY)
        # Morphological gradient to highlight high contrast text/plate borders
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # Sobel X gradient
        grad_x = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        min_val, max_val = np.min(grad_x), np.max(grad_x)
        if max_val > min_val:
            grad_x = 255 * ((grad_x - min_val) / (max_val - min_val))
        grad_x = grad_x.astype("uint8")

        # Threshold and morphology
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = float(cw) / max(1.0, ch)
            # Indian HSRP plates typically have aspect ratio 2.5 to 5.2
            if 2.2 <= aspect <= 5.5 and cw > 60 and ch > 15:
                # Add margin and crop
                return lower_crop[max(0, y-5):min(lower_crop.shape[0], y+ch+5), max(0, x-5):min(lower_crop.shape[1], x+cw+5)]

        return None

    def process_vehicle(self, vehicle_crop: np.ndarray, track_id: int) -> Optional[Dict]:
        """
        Extracts license plate from a detected vehicle and matches against watchlist.
        """
        plate_crop = self.locate_plate_region(vehicle_crop)
        if plate_crop is None:
            return None

        # Here we simulate/extract plate characters or use lightweight OCR
        # For demonstration on sample data, return structured detection
        return {
            "vehicle_track_id": track_id,
            "plate_candidate_found": True,
            "plate_crop_shape": plate_crop.shape
        }

    def register_manual_scan(self, raw_plate: str, vehicle_class: str = "car") -> Dict:
        """Register a verified or OCR plate."""
        parsed = self.validate_and_format_plate(raw_plate)
        if parsed:
            formatted, p_type = parsed
        else:
            formatted = raw_plate.upper().strip()
            p_type = "STANDARD"

        clean_key = self.clean_text(formatted)
        is_wanted = clean_key in self.watchlist_plates
        reason = self.watchlist_plates.get(clean_key, "")

        record = {
            "plate": formatted,
            "type": p_type,
            "vehicle_class": vehicle_class,
            "is_flagged": is_wanted,
            "reason": reason if is_wanted else "Clear - Authorized Transit"
        }
        self.scanned_records.append(record)
        return record
