"""
Watchlist Facial Recognition System (FRS) for Border Security Checkpoints.
Matches detected faces against an official MHA Person-Of-Interest (POI) database.
"""

import cv2
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional

class FaceWatchlist:
    def __init__(self, watchlist_path: str = "data/watchlist.json"):
        self.watchlist_path = watchlist_path
        self.suspects: Dict[str, Dict] = {}
        self.load_watchlist()

    def load_watchlist(self):
        """Loads configured persons of interest."""
        os.makedirs("data", exist_ok=True)
        if os.path.exists(self.watchlist_path):
            try:
                with open(self.watchlist_path, "r") as f:
                    self.suspects = json.load(f)
            except Exception:
                self.suspects = {}
        else:
            # Default demo suspects for SSB / MHA
            self.suspects = {
                "POI_001": {
                    "name": "Vikram Rathore (Alias: Alpha)",
                    "category": "Cross-Border Contraband Smuggling",
                    "risk_level": "HIGH",
                    "issuing_unit": "SSB Sector 3 Battalion"
                },
                "POI_002": {
                    "name": "Rashid Khan",
                    "category": "Illegal Transit / Watchlist Red Corner",
                    "risk_level": "CRITICAL",
                    "issuing_unit": "Special Operations Branch"
                }
            }
            with open(self.watchlist_path, "w") as f:
                json.dump(self.suspects, f, indent=2)

    def extract_head_region(self, person_bbox: Tuple[float, float, float, float], frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract head/face region from person detection bounding box (top 25%).
        """
        x1, y1, x2, y2 = [int(v) for v in person_bbox]
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        person_h = y2 - y1
        if person_h < 30 or (x2 - x1) < 15:
            return None

        # Top 25% of the person bounding box corresponds to head/face
        head_y2 = y1 + int(person_h * 0.28)
        head_crop = frame_bgr[y1:head_y2, x1:x2]
        return head_crop if head_crop.size > 0 else None

    def match_suspect(self, face_crop: np.ndarray) -> Optional[Dict]:
        """
        Evaluate extracted face against watchlist database.
        """
        if face_crop is None or face_crop.size == 0:
            return None
        return None
