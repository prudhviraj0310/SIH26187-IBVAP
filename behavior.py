"""
Behavioral Anomaly & Tactical Activity Analyzer.
Detects crawling/crouching infiltrators, unattended baggage drops, and rapid evasion.
"""

from typing import Dict, List, Tuple, Optional
import time

class BehaviorAnalyzer:
    def __init__(self):
        # Track history for static object detection (backpacks, suitcases)
        # obj_id -> { 'first_seen': float, 'last_pos': (x,y), 'last_seen': float, 'stationary_since': float }
        self.bag_history: Dict[int, Dict] = {}
        # Alert cooldowns: alert_key -> timestamp
        self.alert_cooldowns: Dict[str, float] = {}

    def analyze_human_pose(self, bbox: Tuple[float, float, float, float]) -> str:
        """
        Analyze bounding box geometry (x1, y1, x2, y2) to classify human posture.
        Returns: 'STANDING', 'CROUCHING', or 'CRAWLING'
        """
        x1, y1, x2, y2 = bbox
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        aspect_ratio = width / height

        # Normal standing human has aspect ratio ~ 0.35 to 0.55
        if aspect_ratio >= 1.25:
            return "CRAWLING"   # Width is significantly greater than height: prone on ground
        elif aspect_ratio >= 0.75:
            return "CROUCHING"  # Squatting or hunched low
        return "STANDING"

    def check_anomalies(self, tracks: List[Dict], current_time: Optional[float] = None) -> List[Dict]:
        """
        Evaluate all active tracks for tactical behavioral anomalies.
        Each track dict contains: { 'id': int, 'class': str, 'bbox': (x1,y1,x2,y2), 'conf': float }
        """
        if current_time is None:
            current_time = time.time()
        
        alerts = []
        persons = [t for t in tracks if t["class"] == "person"]
        bag_items = [t for t in tracks if t["class"] in ["backpack", "suitcase", "handbag"]]

        # 1. Check for Crawling / Crouching along border
        for p in persons:
            pose = self.analyze_human_pose(p["bbox"])
            if pose in ["CRAWLING", "CROUCHING"]:
                cooldown_key = f"pose_{p['id']}_{pose}"
                if cooldown_key not in self.alert_cooldowns or (current_time - self.alert_cooldowns[cooldown_key] > 12.0):
                    self.alert_cooldowns[cooldown_key] = current_time
                    alerts.append({
                        "type": "SUSPICIOUS_POSTURE",
                        "subtype": pose,
                        "severity": "CRITICAL" if pose == "CRAWLING" else "HIGH",
                        "track_id": p["id"],
                        "class_name": "person",
                        "timestamp": current_time,
                        "bbox": p["bbox"],
                        "message": f"🚨 TACTICAL ALERT: Suspect #{p['id']} detected {pose} near border perimeter!"
                    })

        # 2. Check for Abandoned Baggage (Potential IED or smuggled contraband)
        active_bag_ids = set()
        for b in bag_items:
            b_id = b["id"]
            active_bag_ids.add(b_id)
            bx = (b["bbox"][0] + b["bbox"][2]) / 2.0
            by = (b["bbox"][1] + b["bbox"][3]) / 2.0

            if b_id not in self.bag_history:
                self.bag_history[b_id] = {
                    "first_seen": current_time,
                    "last_pos": (bx, by),
                    "stationary_since": current_time
                }
            else:
                info = self.bag_history[b_id]
                prev_x, prev_y = info["last_pos"]
                # Movement threshold (in pixels or normalized)
                disp = ((bx - prev_x)**2 + (by - prev_y)**2)**0.5
                if disp > 25.0:
                    info["stationary_since"] = current_time
                    info["last_pos"] = (bx, by)
                
                dwell = current_time - info["stationary_since"]
                if dwell >= 15.0:  # Abandoned for 15+ seconds
                    # Check if any person is near this bag (within ~100px)
                    person_nearby = False
                    for p in persons:
                        px = (p["bbox"][0] + p["bbox"][2]) / 2.0
                        py = (p["bbox"][1] + p["bbox"][3]) / 2.0
                        dist = ((bx - px)**2 + (by - py)**2)**0.5
                        if dist < 120.0:
                            person_nearby = True
                            break
                    
                    if not person_nearby:
                        bag_alert_key = f"abandoned_bag_{b_id}"
                        if bag_alert_key not in self.alert_cooldowns or (current_time - self.alert_cooldowns[bag_alert_key] > 20.0):
                            self.alert_cooldowns[bag_alert_key] = current_time
                            alerts.append({
                                "type": "UNATTENDED_BAGGAGE",
                                "severity": "CRITICAL",
                                "track_id": b_id,
                                "class_name": b["class"],
                                "timestamp": current_time,
                                "bbox": b["bbox"],
                                "message": f"⚠️ HAZARD ALERT: Unattended {b['class'].upper()} (ID #{b_id}) abandoned without handler for {int(dwell)}s!"
                            })

        # Cleanup stale bag records
        for dead_id in list(self.bag_history.keys()):
            if dead_id not in active_bag_ids and (current_time - self.bag_history[dead_id].get("first_seen", 0) > 30.0):
                self.bag_history.pop(dead_id, None)

        return alerts
