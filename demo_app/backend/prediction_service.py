from __future__ import annotations

import io

import numpy as np

from domain_models import AggregationResult, PredictOptions, PredictionResponse
from inference_service import InferenceService, RiskScoringService
from preprocessing_service import PreprocessingService


class PredictionService:
    def __init__(
        self,
        preprocessing_service: PreprocessingService,
        inference_service: InferenceService,
        risk_service: RiskScoringService,
    ):
        self.preprocessing_service = preprocessing_service
        self.inference_service = inference_service
        self.risk_service = risk_service

    def predict(
        self,
        file_bytes: bytes,
        filename: str,
        options: PredictOptions,
        normal_mean: float,
        anomaly_mean: float,
    ) -> PredictionResponse:
        debug = {"skip_preprocess": options.skip_preprocess}

        if options.skip_preprocess:
            x = self._preprocessed_input(file_bytes, filename)
            maps = self.inference_service.compute_score_and_maps(x)
            risk = self.risk_service.compute_risk_level(maps.score, options.threshold, anomaly_mean)

            debug.update(
                {
                    "type": "preprocessed",
                    "shape": list(x.shape),
                    "min": float(x.min()),
                    "max": float(x.max()),
                    "nonzero_ratio": float(np.count_nonzero(x) / x.size),
                }
            )

            return PredictionResponse(
                anomaly=bool(maps.score > options.threshold),
                risk_level=risk,
                score=maps.score,
                threshold=options.threshold,
                aggregation=AggregationResult(enabled=False),
                debug=debug,
                maps=maps,
            )

        if filename.lower().endswith(".nii") or filename.lower().endswith(".nii.gz"):
            return self._predict_from_volume(file_bytes, filename, options, anomaly_mean, debug)

        x, pre_dbg = self.preprocessing_service.preprocess_any(
            file_bytes=file_bytes,
            filename=filename,
            apply_nyul=options.apply_nyul,
            apply_center=options.apply_center,
        )
        debug.update(pre_dbg)

        maps = self.inference_service.compute_score_and_maps(x)
        risk = self.risk_service.compute_risk_level(maps.score, options.threshold, anomaly_mean)

        return PredictionResponse(
            anomaly=bool(maps.score > options.threshold),
            risk_level=risk,
            score=maps.score,
            threshold=options.threshold,
            aggregation=AggregationResult(enabled=False),
            debug=debug,
            maps=maps,
        )

    def _predict_from_volume(
        self,
        file_bytes: bytes,
        filename: str,
        options: PredictOptions,
        anomaly_mean: float,
        debug: dict,
    ) -> PredictionResponse:
        vol01 = self.preprocessing_service.load_nifti_volume_from_bytes(file_bytes, filename)
        best_idx = self.preprocessing_service.pick_middle_slice_index(vol01)

        if options.use_aggregation:
            idxs = self.preprocessing_service.get_slice_indices_around(best_idx, vol01.shape[2], options.agg_slices)
        else:
            idxs = [best_idx]

        slice_scores = []
        payload = []
        for idx in idxs:
            x = self.preprocessing_service.preprocess_single_slice_from_volume(
                vol01=vol01,
                idx=idx,
                apply_nyul=options.apply_nyul,
                apply_center=options.apply_center,
            )
            maps = self.inference_service.compute_score_and_maps(x)
            slice_scores.append(float(maps.score))
            payload.append((idx, maps))

        if options.agg_method == "median":
            final_score = float(np.median(slice_scores))
        else:
            final_score = float(np.mean(slice_scores))

        rep_idx, rep_maps = max(payload, key=lambda pair: pair[1].score)
        risk = self.risk_service.compute_risk_level(final_score, options.threshold, anomaly_mean)

        debug.update(
            {
                "type": "nifti",
                "orig_shape": list(vol01.shape),
                "selected_center_slice": int(best_idx),
                "used_indices": [int(i) for i in idxs],
                "agg_method": options.agg_method,
                "slice_scores": slice_scores,
                "nonzero_ratio_rep": float(np.count_nonzero(rep_maps.input_slice) / rep_maps.input_slice.size),
                "rep_slice": int(rep_idx),
            }
        )

        return PredictionResponse(
            anomaly=bool(final_score > options.threshold),
            risk_level=risk,
            score=final_score,
            threshold=options.threshold,
            aggregation=AggregationResult(
                enabled=bool(options.use_aggregation),
                k=len(idxs),
                method=options.agg_method,
                rep_slice=int(rep_idx),
                slice_scores=slice_scores,
                slice_indices=[int(i) for i in idxs],
            ),
            debug=debug,
            maps=rep_maps,
        )

    @staticmethod
    def _preprocessed_input(file_bytes: bytes, filename: str) -> np.ndarray:
        if filename.lower().endswith(".npy"):
            x = np.load(io.BytesIO(file_bytes), allow_pickle=False).astype(np.float32)
        else:
            from PIL import Image

            img = Image.open(io.BytesIO(file_bytes)).convert("L")
            x = np.array(img).astype(np.float32) / 255.0

        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 2:
            raise ValueError(f"Preprocessed input must be 2D 128×128. Got {x.shape}")
        if x.shape != (128, 128):
            raise ValueError(f"Preprocessed input must be 128×128. Got {x.shape}")

        if x.max() > 1.5:
            x = x / 255.0
        return np.clip(x, 0, 1).astype(np.float32)
