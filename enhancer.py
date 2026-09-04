"""
Night-Vision & Low-Light Contrast Enhancer for Border Surveillance Feeds.
Includes adaptive CLAHE luminance boost and Tactical Thermal IR simulation.
"""

import numpy as np

class LowLightEnhancer:
    def __init__(self, clip_limit: float = 3.0, tile_grid_size: tuple = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = None

    def _get_clahe(self, cv2):
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return self._clahe

    def is_low_light(self, frame_bgr: np.ndarray, threshold: float = 70.0) -> bool:
        """Check if frame average luminance is below threshold (0-255)."""
        # Fast luminance approximation
        gray = np.mean(frame_bgr, axis=2)
        return np.mean(gray) < threshold

    def enhance(self, frame_bgr: np.ndarray, cv2, force: bool = False) -> np.ndarray:
        """
        Enhance low-light video using LAB color space CLAHE.
        Preserves color fidelity while boosting shadow details.
        """
        if not force and not self.is_low_light(frame_bgr):
            return frame_bgr

        # Convert to LAB color space
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L (Luminance) channel only
        clahe = self._get_clahe(cv2)
        enhanced_l = clahe.apply(l_channel)

        # Merge back and convert to BGR
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Apply subtle gamma correction for midtones
        inv_gamma = 1.0 / 1.3
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(enhanced_bgr, table)

    def apply_tactical_thermal(self, frame_bgr: np.ndarray, cv2) -> np.ndarray:
        """
        Converts camera feed into a Tactical Thermal (Ironbow / FLIR) spectrum view.
        Useful for simulating multi-spectral border surveillance feeds.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Apply CLAHE to maximize thermal contrast
        clahe = self._get_clahe(cv2)
        norm_gray = clahe.apply(gray)
        # Apply Inferno/Ironbow colormap
        thermal = cv2.applyColorMap(norm_gray, cv2.COLORMAP_INFERNO)
        return thermal
