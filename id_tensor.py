"""
ID Tensor (Informational Deviation Tensor)
Author: Naoki S. Kiryu

Description:
    Maps macroscopic atomic properties (EN, IE, R) to microscopic 
    geometric deviations (bond angles) using a 10D tensor space 
    and Coulson's Theorem.
"""

import numpy as np
import math

class ID_Tensor:
    def __init__(self):
        # [EN(Pauling), IE(eV), Radius(Å), Target Angle(°)]
        self.training_data = {
            "N":  {"EN": 3.04, "IE": 14.53, "R": 0.71, "Angle": 107.8},
            "O":  {"EN": 3.44, "IE": 13.62, "R": 0.63, "Angle": 104.5},
            "P":  {"EN": 2.19, "IE": 10.48, "R": 1.11, "Angle": 93.5},
            "S":  {"EN": 2.58, "IE": 10.36, "R": 1.03, "Angle": 92.1},
            "As": {"EN": 2.18, "IE": 9.81,  "R": 1.21, "Angle": 91.8},
            "Se": {"EN": 2.55, "IE": 9.75,  "R": 1.16, "Angle": 91.0},
            "Sb": {"EN": 2.05, "IE": 8.60,  "R": 1.40, "Angle": 91.7},
            "Te": {"EN": 2.10, "IE": 9.01,  "R": 1.36, "Angle": 90.3},
            "Po": {"EN": 2.00, "IE": 8.42,  "R": 1.40, "Angle": 90.9}
        }
        self.weights = None

    def _target_deviation(self, angle_deg):
        """Converts bond angle to deviation state (S) via Coulson's Theorem."""
        rad = math.radians(angle_deg)
        cos_theta = math.cos(rad)
        return cos_theta / (cos_theta - 1.0)

    def _resolve_angle(self, deviation_state):
        """Resolves physical angle from deviation state (S)."""
        bounded_state = max(0.0, min(0.25, deviation_state))
        if bounded_state == 0:
            return 90.0
        
        cos_theta = bounded_state / (bounded_state - 1.0)
        return math.degrees(math.acos(cos_theta))

    def _build_features(self, EN, IE, R):
        """Expands 3D vector to 10D geometric tensor."""
        return np.array([
            1.0, 
            EN, IE, R, 
            EN**2, IE**2, R**2, 
            EN*IE, EN*R, IE*R
        ])

    def calibrate(self):
        """Fits the tensor weights to the spatial mapping."""
        X, Y = [], []
        for data in self.training_data.values():
            X.append(self._build_features(data["EN"], data["IE"], data["R"]))
            Y.append(self._target_deviation(data["Angle"]))

        self.weights, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        print("[System] Tensor calibrated. Mapping locked.")

    def verify(self):
        """Verifies the exact mapping of known elements."""
        print(f"{'Atom':<6} | {'Target Angle':<15} | {'Predicted':<15} | {'Error'}")
        print("-" * 55)
        for el, data in self.training_data.items():
            features = self._build_features(data["EN"], data["IE"], data["R"])
            pred_angle = self._resolve_angle(np.dot(features, self.weights))
            error = abs(data["Angle"] - pred_angle)
            print(f"{el:<6} | {data['Angle']:>12.4f}°   | {pred_angle:>12.4f}°  | {error:>8.4f}°")

    def predict(self, name, EN, IE, R):
        """Predicts the geometric deviation for any given parameters."""
        if self.weights is None:
            raise ValueError("Tensor not calibrated.")
            
        features = self._build_features(EN, IE, R)
        pred_angle = self._resolve_angle(np.dot(features, self.weights))
        print(f"\n[Prediction] {name}: {pred_angle:.4f}°")


if __name__ == "__main__":
    tensor = ID_Tensor()
    tensor.calibrate()
    tensor.verify()
    tensor.predict("Bismuth (Bi)", EN=2.02, IE=7.289, R=1.51)
