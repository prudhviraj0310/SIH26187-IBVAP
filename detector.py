"""
Real-Time YOLOv11/v8 + ByteTrack Object Detection & Multi-Object Tracking.
Optimized for Apple Silicon MPS GPU acceleration or standard CPU.
"""

import os
from typing import List, Dict, Optional
import numpy as np
import torch
from ultralytics import YOLO

# Tactical surveillance classes to retain from standard pre-trained weights
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    24: "backpack",
    26: "handbag",
    28: "suitcase"
}

class SurveillanceDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf_thresh: float = 0.35):
        self.conf_thresh = conf_thresh
        
        # CPU is used for rock-solid multi-threaded streaming across 5 cameras on macOS
        self.device = "cpu"
            
        print(f"[IBVAP] Initializing YOLO ({model_name}) on device: {self.device} (Thread-Safe Multi-Camera Mode)...")
        self.model = YOLO(model_name)
        # Warmup model
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model(dummy, verbose=False, device=self.device)
        print("[IBVAP] YOLO Detection Core ready.")

    def process_frame(self, frame_bgr: np.ndarray) -> List[Dict]:
        """
        Run tracking on a single video frame.
        Returns list of active tracks with bounding boxes, classes, and IDs.
        """
        # Run ByteTrack multi-object tracking
        results = self.model.track(
            frame_bgr,
            persist=True,
            conf=self.conf_thresh,
            classes=list(TARGET_CLASSES.keys()),
            tracker="bytetrack.yaml",
            verbose=False,
            device=self.device
        )
        
        active_tracks = []
        if not results or len(results) == 0:
            return active_tracks
            
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return active_tracks
            
        h, w = frame_bgr.shape[:2]
        
        for box in boxes:
            cls_id = int(box.cls[0].item())
            class_name = TARGET_CLASSES.get(cls_id, "unknown")
            conf = float(box.conf[0].item())
            
            # Extract track ID if available from ByteTrack
            if box.id is not None:
                track_id = int(box.id[0].item())
            else:
                track_id = -1
                
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            
            active_tracks.append({
                "id": track_id,
                "class": class_name,
                "conf": round(conf, 2),
                "bbox": (round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)),
                "center": (round(cx, 1), round(cy, 1)),
                "center_norm": (round(cx / w, 4), round(cy / h, 4))
            })
            
        return active_tracks
