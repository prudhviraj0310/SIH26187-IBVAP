"""
Virtual Tripwire & Geofencing Engine for Perimeter Intrusion Detection.
Mathematically deterministic directional line crossing & polygon containment.
"""

from typing import Tuple, List, Dict, Optional
import time

def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    """Check if three points are listed in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A: Tuple[float, float], B: Tuple[float, float],
                       C: Tuple[float, float], D: Tuple[float, float]) -> bool:
    """Return True if line segment AB intersects line segment CD."""
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))

def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Ray casting algorithm to determine if a point is inside a polygon."""
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

class TripwireManager:
    def __init__(self):
        # Default border tripwire across frame: normalized (0-1) coordinates
        # (x1, y1) to (x2, y2)
        self.tripwires = [
            {
                "id": "border_fence_alpha",
                "name": "Border Fence - Sector 4",
                "p1": (0.1, 0.65),  # 10% from left, 65% down
                "p2": (0.9, 0.65),  # 90% across, 65% down
                "direction": "both", # 'inbound', 'outbound', 'both'
                "color": (0, 0, 255) # Red in BGR
            }
        ]
        
        # Restricted Exclusion Zones (Polygons in normalized coordinates)
        self.exclusion_zones = [
            {
                "id": "restricted_bunker_zone",
                "name": "Buffer Zone 14",
                "polygon": [(0.2, 0.7), (0.8, 0.7), (0.85, 0.95), (0.15, 0.95)],
                "max_dwell_sec": 10.0 # Alert if person stays longer than 10s
            }
        ]
        
        # Track history: track_id -> [(x, y, timestamp), ...]
        self.track_history: Dict[int, List[Tuple[float, float, float]]] = {}
        # Crossed records: (track_id, wire_id) -> timestamp
        self.crossed_cache: Dict[Tuple[int, str], float] = {}

    def set_tripwire(self, p1: Tuple[float, float], p2: Tuple[float, float], name: str = "Custom Perimeter Line"):
        """Update or set custom perimeter line (normalized 0-1)."""
        self.tripwires = [{
            "id": "custom_tripwire_1",
            "name": name,
            "p1": p1,
            "p2": p2,
            "direction": "both",
            "color": (0, 0, 255)
        }]
        self.crossed_cache.clear()

    def update_track(self, track_id: int, center_norm: Tuple[float, float], class_name: str) -> List[Dict]:
        """
        Update tracking position for an object.
        Returns list of triggered alerts (if any).
        """
        now = time.time()
        alerts = []
        
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        
        history = self.track_history[track_id]
        history.append((center_norm[0], center_norm[1], now))
        
        # Keep only recent 5 seconds of trajectory
        history = [pt for pt in history if now - pt[2] <= 5.0]
        self.track_history[track_id] = history
        
        # We need at least 2 points to evaluate motion across lines
        if len(history) < 2:
            return alerts
        
        prev_pt = (history[-2][0], history[-2][1])
        curr_pt = (history[-1][0], history[-1][1])
        
        # Check Tripwire crossings
        for wire in self.tripwires:
            w_id = wire["id"]
            cache_key = (track_id, w_id)
            
            # Don't trigger repeatedly within 10 seconds for the same track
            if cache_key in self.crossed_cache and (now - self.crossed_cache[cache_key] < 10.0):
                continue
            
            p1, p2 = wire["p1"], wire["p2"]
            if segments_intersect(prev_pt, curr_pt, p1, p2):
                self.crossed_cache[cache_key] = now
                
                # Animal suppression check
                is_animal = class_name.lower() in ["dog", "cat", "cow", "horse", "sheep", "bird"]
                severity = "LOW" if is_animal else "CRITICAL"
                alert_type = "WILDLIFE_CROSSING" if is_animal else "PERIMETER_BREACH"
                
                alerts.append({
                    "type": alert_type,
                    "severity": severity,
                    "track_id": track_id,
                    "class_name": class_name,
                    "zone": wire["name"],
                    "timestamp": now,
                    "position": curr_pt,
                    "suppressed": is_animal,
                    "message": f"🚨 {alert_type}: {class_name.upper()} (ID #{track_id}) crossed {wire['name']}!"
                })
        
        # Check Exclusion Zone loitering
        for zone in self.exclusion_zones:
            if point_in_polygon(curr_pt, zone["polygon"]):
                first_seen_in_zone = None
                for pt in history:
                    if point_in_polygon((pt[0], pt[1]), zone["polygon"]):
                        first_seen_in_zone = pt[2]
                        break
                
                if first_seen_in_zone and (now - first_seen_in_zone >= zone["max_dwell_sec"]):
                    loiter_key = (track_id, f"loiter_{zone['id']}")
                    if loiter_key not in self.crossed_cache or (now - self.crossed_cache[loiter_key] > 15.0):
                        self.crossed_cache[loiter_key] = now
                        is_animal = class_name.lower() in ["dog", "cat", "cow", "horse", "sheep"]
                        alerts.append({
                            "type": "SUSPICIOUS_LOITERING",
                            "severity": "MEDIUM" if is_animal else "HIGH",
                            "track_id": track_id,
                            "class_name": class_name,
                            "zone": zone["name"],
                            "timestamp": now,
                            "position": curr_pt,
                            "suppressed": is_animal,
                            "message": f"⚠️ LOITERING ALERT: {class_name.upper()} (ID #{track_id}) lingering in {zone['name']} for >{zone['max_dwell_sec']}s!"
                        })
        
        return alerts

    def cleanup_inactive_tracks(self, active_track_ids: List[int]):
        """Remove tracks that have exited the camera field of view."""
        current_tracks = set(self.track_history.keys())
        inactive = current_tracks - set(active_track_ids)
        for tid in inactive:
            self.track_history.pop(tid, None)
