from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class PredictOptions:
    threshold: float
    apply_center: bool = True
    apply_nyul: bool = True
    skip_preprocess: bool = False
    use_aggregation: bool = False
    agg_slices: int = 7
    agg_method: str = "mean"


@dataclass
class InferenceMaps:
    input_slice: np.ndarray
    reconstruction: np.ndarray
    error_abs: np.ndarray
    error_smooth: np.ndarray
    score: float


@dataclass
class AggregationResult:
    enabled: bool = False
    k: int = 0
    method: str = "mean"
    rep_slice: int | None = None
    slice_scores: List[float] = field(default_factory=list)
    slice_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "enabled": bool(self.enabled),
        }
        if self.enabled:
            out.update(
                {
                    "k": int(self.k),
                    "method": self.method,
                    "rep_slice": int(self.rep_slice) if self.rep_slice is not None else None,
                    "slice_scores": [float(v) for v in self.slice_scores],
                    "slice_indices": [int(v) for v in self.slice_indices],
                }
            )
        return out


@dataclass
class PredictionResponse:
    anomaly: bool
    risk_level: str
    score: float
    threshold: float
    aggregation: AggregationResult
    debug: Dict[str, Any]
    maps: InferenceMaps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly": bool(self.anomaly),
            "risk_level": self.risk_level,
            "score": float(self.score),
            "threshold": float(self.threshold),
            "aggregation": self.aggregation.to_dict(),
            "debug": self.debug,
            "arrays": {
                "input": self.maps.input_slice.tolist(),
                "reconstruction": self.maps.reconstruction.tolist(),
                "error_abs": self.maps.error_abs.tolist(),
                "error_smooth": self.maps.error_smooth.tolist(),
            },
        }
