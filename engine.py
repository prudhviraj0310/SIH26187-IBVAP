"""
Unified IBVAP Video Analytics Engine.
Integrates Detection, Tracking, Virtual Tripwires, Behavior Analysis, ANPR, and HUD rendering.
"""

import time
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

from .detector import SurveillanceDetector
from .tripwire import TripwireManager
from .behavior import BehaviorAnalyzer
from .enhancer import LowLightEnhancer
from .anpr import ANPREngine
from .face_watch import FaceWatchlist

class IBVAPEngine:
    def __init__(self, model_name: str = "yolov8n.pt"):
        self.detector = SurveillanceDetector(model_name=model_name)
        self.tripwire = TripwireManager()
        self.behavior = BehaviorAnalyzer()
        self.enhancer = LowLightEnhancer()
        self.anpr = ANPREngine()
        self.face_watch = FaceWatchlist()
        
        # Telemetry & Stats
        self.total_frames = 0
        self.fps = 30.0
        self._last_fps_time = time.time()
        self.threat_level = "SECURE"  # SECURE, ELEVATED, CRITICAL
        self.recent_alerts: List[Dict] = []
        self.active_intruders = 0

    def process(self, frame_bgr: np.ndarray, 
                camera_id: str = "CAM_01",
                sector_name: str = "Sector 4 (Fence Perimeter)",
                night_mode: bool = False,
                thermal_mode: bool = False) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single surveillance frame through all AI modules.
        Returns: (annotated_hud_frame, list_of_new_alerts)
        """
        self.total_frames += 1
        now = time.time()
        
        # FPS calculation
        dt = now - self._last_fps_time
        if dt >= 1.0:
            self.fps = round(self.total_frames / dt, 1)
            self.total_frames = 0
            self._last_fps_time = now

        h, w = frame_bgr.shape[:2]
        new_alerts = []

        # 1. Visual Enhancement Layer (if requested or low-light detected)
        working_frame = frame_bgr
        if thermal_mode:
            working_frame = self.enhancer.apply_tactical_thermal(frame_bgr, cv2)
        elif night_mode or self.enhancer.is_low_light(frame_bgr):
            working_frame = self.enhancer.enhance(frame_bgr, cv2, force=True)

        # 2. Deep Learning Detection & ByteTrack
        tracks = self.detector.process_frame(working_frame)
        active_track_ids = [t["id"] for t in tracks if t["id"] != -1]
        self.tripwire.cleanup_inactive_tracks(active_track_ids)

        # 3. Behavioral Posture & Anomaly Detection
        behavior_alerts = self.behavior.check_anomalies(tracks, current_time=now)
        for a in behavior_alerts:
            a["camera_id"] = camera_id
            a["sector"] = sector_name
            new_alerts.append(a)

        # 4. Virtual Tripwire & Geofence Perimeter Logic
        is_breaching = False
        for t in tracks:
            # Check tripwire crossings
            t_alerts = self.tripwire.update_track(t["id"], t["center_norm"], t["class"])
            for a in t_alerts:
                a["camera_id"] = camera_id
                a["sector"] = sector_name
                new_alerts.append(a)
                if a["severity"] == "CRITICAL":
                    is_breaching = True

        # Threat Level Management
        if is_breaching or any(a["severity"] == "CRITICAL" for a in new_alerts):
            self.threat_level = "CRITICAL"
        elif any(a["severity"] == "HIGH" for a in new_alerts):
            self.threat_level = "ELEVATED"
        else:
            # Decay to SECURE if no active breaches in last 5 seconds
            if not any(now - a["timestamp"] < 5.0 for a in self.recent_alerts if a["severity"] == "CRITICAL"):
                self.threat_level = "SECURE"

        # Keep recent alerts
        for a in new_alerts:
            self.recent_alerts.insert(0, a)
        self.recent_alerts = self.recent_alerts[:50]

        # 5. Tactical HUD Rendering (Defense Grade)
        display_frame = working_frame.copy()
        display_frame = self._draw_tactical_hud(display_frame, tracks, is_breaching, camera_id, sector_name)

        return display_frame, new_alerts

    def _draw_tactical_hud(self, frame: np.ndarray, tracks: List[Dict], 
                           is_breached: bool, camera_id: str, sector: str) -> np.ndarray:
        """Render defense-style tactical overlays, tripwires, and bounding boxes."""
        h, w = frame.shape[:2]

        # 1. Draw Virtual Tripwires
        for wire in self.tripwire.tripwires:
            p1_norm, p2_norm = wire["p1"], wire["p2"]
            pt1 = (int(p1_norm[0] * w), int(p1_norm[1] * h))
            pt2 = (int(p2_norm[0] * w), int(p2_norm[1] * h))

            # Pulsing color if breach occurred
            line_color = (0, 0, 255) if is_breached else (0, 255, 255) # Yellow/Cyan or Red
            cv2.line(frame, pt1, pt2, line_color, 3, cv2.LINE_AA)
            # Label
            cv2.putText(frame, f"VIRTUAL TRIPWIRE: {wire['name'].upper()}", (pt1[0], pt1[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2, cv2.LINE_AA)

        # 2. Draw Object Tracks
        intruder_count = 0
        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
            cls = t["class"]
            tid = t["id"]
            conf = int(t["conf"] * 100)

            # Determine tactical color & category
            if cls == "person":
                intruder_count += 1
                pose = self.behavior.analyze_human_pose(t["bbox"])
                if pose in ["CRAWLING", "CROUCHING"]:
                    box_color = (0, 0, 255) # Red
                    label = f"SUSPECT #{tid} [{pose}] {conf}%"
                else:
                    box_color = (0, 140, 255) # Orange
                    label = f"PERSON #{tid} {conf}%"
            elif cls in ["car", "truck", "bus"]:
                box_color = (255, 200, 0) # Cyan
                label = f"VEHICLE: {cls.upper()} #{tid}"
            elif cls in ["cow", "dog", "horse", "sheep"]:
                box_color = (180, 180, 180) # Gray
                label = f"WILDLIFE: {cls.upper()} (FILTERED)"
            else:
                box_color = (0, 255, 128) # Green
                label = f"{cls.upper()} #{tid}"

            # Draw bounding box with corner crosshairs
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            # Corner accents
            corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, 4)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, 4)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, 4)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, 4)

            # Label banner
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), box_color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        self.active_intruders = intruder_count

        # 3. Top Banner: Command Status & Threat Radar
        banner_h = 34
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (12, 16, 24), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        # Threat indicator pill
        if self.threat_level == "CRITICAL":
            threat_color = (0, 0, 255)
            threat_text = "THREAT: CRITICAL (BREACH)"
        elif self.threat_level == "ELEVATED":
            threat_color = (0, 165, 255)
            threat_text = "THREAT: ELEVATED"
        else:
            threat_color = (0, 255, 100)
            threat_text = "STATUS: SECURE"

        # Status indicator circle and camera ID
        cv2.circle(frame, (16, 17), 5, threat_color, -1)
        cv2.putText(frame, f"[{camera_id}]", (26, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)
        
        # Threat state badge
        (th_w, _), _ = cv2.getTextSize(threat_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 2)
        cv2.putText(frame, threat_text, (max(w // 2 - th_w // 2, 130), 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, threat_color, 2, cv2.LINE_AA)

        # FPS and targets on the right
        stats_str = f"FPS: {self.fps} | {len(tracks)} TGT"
        (st_w, _), _ = cv2.getTextSize(stats_str, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(frame, stats_str, (w - st_w - 14, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        # Flashing red border alert if active critical breach
        if is_breached and int(time.time() * 3) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 6)
            cv2.putText(frame, "🚨 PERIMETER BREACH IN PROGRESS 🚨", (w // 2 - 220, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3, cv2.LINE_AA)

        return frame
