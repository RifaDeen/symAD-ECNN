"""
Volume Inference Tests for SymAD-ECNN Prototype.

This script tests the volume-level inference functionality by
processing multiple slices and testing slice aggregation behavior.

Supports dissertation Chapter 8 functional testing.

Author: SymAD-ECNN Project
Purpose: Verify volume aggregation and multi-slice inference
"""

import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union
import json
import time
from datetime import datetime

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DEFAULT_API_URL, API_PREDICT_ENDPOINT,
    JSON_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR,
    ensure_directories_exist
)
from path_utils import (
    get_drive_project_root, find_data_paths,
    get_patient_slice_groups
)
from io_utils import (
    save_json, save_csv, log_message,
    start_experiment_log, end_experiment_log
)
from plotting_utils import save_figure

# Try to import requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Try to import pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# =============================================================================
# PATIENT SLICE GROUPING
# =============================================================================

def group_slices_by_patient(
    image_dir: Path,
    pattern: str = "*.png"
) -> Dict[str, List[Path]]:
    """
    Group image slices by patient ID.
    
    Assumes naming conventions:
    - IXI: IXI###-Site-####_slice##.png
    - BraTS: BraTS2021_#####_slice##.png
    
    Args:
        image_dir: Directory containing image slices.
        pattern: Glob pattern for image files.
        
    Returns:
        Dictionary mapping patient_id to list of slice paths.
    """
    image_dir = Path(image_dir)
    patient_slices = {}
    
    for img_path in image_dir.glob(pattern):
        filename = img_path.stem
        
        # Try to extract patient ID
        if "IXI" in filename:
            # Format: IXI###-Site-####_slice##
            parts = filename.split("_slice")
            patient_id = parts[0] if parts else filename
        elif "BraTS" in filename:
            # Format: BraTS2021_#####_slice## or similar
            parts = filename.rsplit("_slice", 1)
            patient_id = parts[0] if len(parts) > 1 else filename
        else:
            # Generic: try to split by _slice
            parts = filename.rsplit("_slice", 1)
            patient_id = parts[0] if len(parts) > 1 else filename
        
        if patient_id not in patient_slices:
            patient_slices[patient_id] = []
        patient_slices[patient_id].append(img_path)
    
    # Sort slices within each patient by slice number
    for patient_id in patient_slices:
        patient_slices[patient_id] = sorted(
            patient_slices[patient_id],
            key=lambda p: int(p.stem.split("slice")[-1]) if "slice" in p.stem else 0
        )
    
    return patient_slices


# =============================================================================
# VOLUME INFERENCE CLASSES
# =============================================================================

class SlicePrediction:
    """Container for a single slice prediction result."""
    
    def __init__(
        self,
        slice_path: Path,
        prediction: Optional[str],
        confidence: Optional[float],
        score: Optional[float],
        response_time: float,
        error: Optional[str] = None
    ):
        self.slice_path = slice_path
        self.prediction = prediction
        self.confidence = confidence
        self.score = score
        self.response_time = response_time
        self.error = error
        self.success = error is None and prediction is not None
    
    def to_dict(self) -> Dict:
        return {
            "slice_path": str(self.slice_path),
            "prediction": self.prediction,
            "confidence": self.confidence,
            "score": self.score,
            "response_time_ms": self.response_time * 1000,
            "success": self.success,
            "error": self.error
        }


class VolumeInferenceResult:
    """Container for volume-level inference results."""
    
    def __init__(
        self,
        patient_id: str,
        slice_predictions: List[SlicePrediction]
    ):
        self.patient_id = patient_id
        self.slice_predictions = slice_predictions
        self.timestamp = datetime.now().isoformat()
    
    @property
    def total_slices(self) -> int:
        return len(self.slice_predictions)
    
    @property
    def successful_slices(self) -> int:
        return sum(1 for s in self.slice_predictions if s.success)
    
    @property
    def anomaly_slices(self) -> int:
        return sum(
            1 for s in self.slice_predictions
            if s.success and s.prediction == "anomaly"
        )
    
    @property
    def normal_slices(self) -> int:
        return sum(
            1 for s in self.slice_predictions
            if s.success and s.prediction == "normal"
        )
    
    @property
    def anomaly_ratio(self) -> float:
        """Ratio of slices predicted as anomalous."""
        if self.successful_slices == 0:
            return 0.0
        return self.anomaly_slices / self.successful_slices
    
    @property
    def mean_confidence(self) -> float:
        """Mean confidence across all successful predictions."""
        confidences = [
            s.confidence for s in self.slice_predictions
            if s.success and s.confidence is not None
        ]
        return np.mean(confidences) if confidences else 0.0
    
    @property
    def mean_score(self) -> float:
        """Mean anomaly score across all successful predictions."""
        scores = [
            s.score for s in self.slice_predictions
            if s.success and s.score is not None
        ]
        return np.mean(scores) if scores else 0.0
    
    @property
    def total_time(self) -> float:
        """Total inference time for all slices."""
        return sum(s.response_time for s in self.slice_predictions)
    
    def get_volume_prediction(
        self,
        threshold: float = 0.5,
        method: str = "majority"
    ) -> Tuple[str, float]:
        """
        Aggregate slice predictions to volume-level prediction.
        
        Args:
            threshold: Anomaly ratio threshold for volume classification.
            method: Aggregation method ("majority", "any", "ratio").
            
        Returns:
            Tuple of (prediction, confidence).
        """
        if self.successful_slices == 0:
            return "unknown", 0.0
        
        if method == "any":
            # Volume is anomalous if ANY slice is anomalous
            prediction = "anomaly" if self.anomaly_slices > 0 else "normal"
            confidence = self.mean_confidence
        elif method == "majority":
            # Volume is anomalous if majority of slices are anomalous
            prediction = "anomaly" if self.anomaly_ratio > 0.5 else "normal"
            confidence = abs(self.anomaly_ratio - 0.5) * 2  # Scale to [0,1]
        else:  # ratio
            # Volume is anomalous if ratio exceeds threshold
            prediction = "anomaly" if self.anomaly_ratio > threshold else "normal"
            confidence = self.mean_confidence
        
        return prediction, confidence
    
    def to_dict(self) -> Dict:
        return {
            "patient_id": self.patient_id,
            "total_slices": self.total_slices,
            "successful_slices": self.successful_slices,
            "anomaly_slices": self.anomaly_slices,
            "normal_slices": self.normal_slices,
            "anomaly_ratio": self.anomaly_ratio,
            "mean_confidence": self.mean_confidence,
            "mean_score": self.mean_score,
            "total_time_seconds": self.total_time,
            "timestamp": self.timestamp,
        }


# =============================================================================
# VOLUME INFERENCE FUNCTIONS
# =============================================================================

def predict_single_slice(
    image_path: Path,
    api_url: str,
    timeout: int = 30
) -> SlicePrediction:
    """
    Send a single slice to the API for prediction.
    
    Args:
        image_path: Path to the slice image.
        api_url: Full URL for the prediction endpoint.
        timeout: Request timeout in seconds.
        
    Returns:
        SlicePrediction object.
    """
    if not REQUESTS_AVAILABLE:
        return SlicePrediction(
            image_path, None, None, None, 0.0,
            error="requests library not available"
        )
    
    image_path = Path(image_path)
    
    if not image_path.exists():
        return SlicePrediction(
            image_path, None, None, None, 0.0,
            error="Image file not found"
        )
    
    start_time = time.time()
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/png')}
            response = requests.post(api_url, files=files, timeout=timeout)
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return SlicePrediction(
                image_path,
                prediction=data.get("prediction"),
                confidence=data.get("confidence"),
                score=data.get("anomaly_score") or data.get("score"),
                response_time=elapsed
            )
        else:
            return SlicePrediction(
                image_path, None, None, None, elapsed,
                error=f"HTTP {response.status_code}"
            )
    
    except requests.exceptions.ConnectionError:
        elapsed = time.time() - start_time
        return SlicePrediction(
            image_path, None, None, None, elapsed,
            error="Connection refused"
        )
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        return SlicePrediction(
            image_path, None, None, None, elapsed,
            error="Request timeout"
        )
    except Exception as e:
        elapsed = time.time() - start_time
        return SlicePrediction(
            image_path, None, None, None, elapsed,
            error=str(e)
        )


def process_patient_volume(
    patient_id: str,
    slice_paths: List[Path],
    api_url: str,
    verbose: bool = True
) -> VolumeInferenceResult:
    """
    Process all slices for a single patient.
    
    Args:
        patient_id: Patient identifier.
        slice_paths: List of paths to slice images.
        api_url: Full URL for prediction endpoint.
        verbose: Whether to print progress.
        
    Returns:
        VolumeInferenceResult object.
    """
    predictions = []
    
    for i, slice_path in enumerate(slice_paths):
        if verbose:
            print(f"  Processing slice {i+1}/{len(slice_paths)}...", end="\r")
        
        prediction = predict_single_slice(slice_path, api_url)
        predictions.append(prediction)
    
    if verbose:
        print(f"  Processed {len(slice_paths)} slices.                ")
    
    return VolumeInferenceResult(patient_id, predictions)


def run_volume_inference_tests(
    data_dir: Optional[Path] = None,
    api_url: str = DEFAULT_API_URL,
    max_patients: int = 5,
    max_slices_per_patient: int = 10,
    save_results: bool = True,
    verbose: bool = True
) -> Tuple[List[VolumeInferenceResult], Dict]:
    """
    Run volume inference tests on grouped patient slices.
    
    Args:
        data_dir: Directory containing test images.
        api_url: Base URL for the API.
        max_patients: Maximum number of patients to process.
        max_slices_per_patient: Maximum slices per patient.
        save_results: Whether to save results to Drive.
        verbose: Whether to print progress.
        
    Returns:
        Tuple of (list of VolumeInferenceResult, summary dict).
    """
    ensure_directories_exist()
    
    # Find data directory if not provided
    if data_dir is None:
        data_paths = find_data_paths()
        # Try test sets first
        for key in ["ixi_test", "brats_test", "ixi_val"]:
            if data_paths.get(key):
                data_dir = Path(data_paths[key])
                break
    
    if data_dir is None or not Path(data_dir).exists():
        print("ERROR: No valid data directory found.")
        return [], {"error": "No data directory"}
    
    data_dir = Path(data_dir)
    
    # Start logging
    log_name = start_experiment_log(
        "volume_inference_tests",
        params={
            "data_dir": str(data_dir),
            "api_url": api_url,
            "max_patients": max_patients
        }
    )
    
    if verbose:
        print("=" * 60)
        print("VOLUME INFERENCE TESTS")
        print("=" * 60)
        print(f"Data directory: {data_dir}")
        print(f"API URL: {api_url}")
        print("-" * 60)
    
    # Group slices by patient
    patient_slices = group_slices_by_patient(data_dir)
    
    if not patient_slices:
        print("ERROR: No patient slices found.")
        return [], {"error": "No slices found"}
    
    if verbose:
        print(f"Found {len(patient_slices)} patients")
        print("-" * 60)
    
    # Select subset of patients
    selected_patients = list(patient_slices.keys())[:max_patients]
    
    # Process each patient
    results = []
    predict_url = f"{api_url.rstrip('/')}{API_PREDICT_ENDPOINT}"
    
    for patient_id in selected_patients:
        slices = patient_slices[patient_id][:max_slices_per_patient]
        
        if verbose:
            print(f"\nProcessing patient: {patient_id} ({len(slices)} slices)")
        
        result = process_patient_volume(patient_id, slices, predict_url, verbose)
        results.append(result)
        
        if verbose:
            pred, conf = result.get_volume_prediction()
            print(f"  Volume prediction: {pred} (ratio={result.anomaly_ratio:.2f})")
    
    # Compute summary
    total_slices = sum(r.total_slices for r in results)
    total_success = sum(r.successful_slices for r in results)
    total_time = sum(r.total_time for r in results)
    
    summary = {
        "total_patients": len(results),
        "total_slices": total_slices,
        "successful_predictions": total_success,
        "success_rate": total_success / total_slices if total_slices else 0,
        "total_time_seconds": total_time,
        "avg_time_per_slice_ms": (total_time / total_slices * 1000) if total_slices else 0,
        "data_dir": str(data_dir),
        "api_url": api_url,
        "timestamp": datetime.now().isoformat(),
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Patients processed: {summary['total_patients']}")
        print(f"Total slices: {summary['total_slices']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Total time: {summary['total_time_seconds']:.1f}s")
        print(f"Avg time per slice: {summary['avg_time_per_slice_ms']:.1f}ms")
        print("=" * 60)
    
    # Save results
    if save_results:
        try:
            # Save detailed results
            detailed_results = []
            for r in results:
                vol_dict = r.to_dict()
                pred, conf = r.get_volume_prediction()
                vol_dict["volume_prediction"] = pred
                vol_dict["volume_confidence"] = conf
                vol_dict["slice_predictions"] = [s.to_dict() for s in r.slice_predictions]
                detailed_results.append(vol_dict)
            
            save_json(
                {
                    "results": detailed_results,
                    "summary": summary
                },
                "volume_inference_results.json"
            )
            
            # Save volume-level summary as CSV
            if PANDAS_AVAILABLE:
                vol_rows = []
                for r in results:
                    pred, conf = r.get_volume_prediction()
                    vol_rows.append({
                        "patient_id": r.patient_id,
                        "total_slices": r.total_slices,
                        "successful_slices": r.successful_slices,
                        "anomaly_slices": r.anomaly_slices,
                        "normal_slices": r.normal_slices,
                        "anomaly_ratio": r.anomaly_ratio,
                        "mean_confidence": r.mean_confidence,
                        "mean_score": r.mean_score,
                        "volume_prediction": pred,
                        "inference_time_s": r.total_time
                    })
                df = pd.DataFrame(vol_rows)
                save_csv(df, "volume_inference_summary.csv")
            
            log_message("Volume inference results saved.", log_name)
            
        except Exception as e:
            log_message(f"Error saving results: {e}", log_name)
    
    end_experiment_log(log_name, summary=summary)
    
    return results, summary


# =============================================================================
# AGGREGATION TESTS
# =============================================================================

def test_aggregation_methods(
    results: List[VolumeInferenceResult],
    save_results: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Test different volume aggregation methods.
    
    Args:
        results: List of volume inference results.
        save_results: Whether to save comparison results.
        verbose: Whether to print results.
        
    Returns:
        Comparison summary dictionary.
    """
    methods = ["any", "majority", "ratio"]
    thresholds = [0.3, 0.5, 0.7]
    
    aggregation_results = []
    
    for r in results:
        row = {"patient_id": r.patient_id, "anomaly_ratio": r.anomaly_ratio}
        
        for method in methods:
            if method == "ratio":
                for thresh in thresholds:
                    pred, conf = r.get_volume_prediction(threshold=thresh, method=method)
                    row[f"{method}_{thresh}"] = pred
            else:
                pred, conf = r.get_volume_prediction(method=method)
                row[method] = pred
        
        aggregation_results.append(row)
    
    if verbose:
        print("\nAggregation Method Comparison:")
        print("-" * 60)
        for row in aggregation_results:
            print(f"Patient {row['patient_id']}:")
            print(f"  Anomaly ratio: {row['anomaly_ratio']:.2f}")
            print(f"  'any' method: {row.get('any', 'N/A')}")
            print(f"  'majority' method: {row.get('majority', 'N/A')}")
            for thresh in thresholds:
                print(f"  'ratio' (t={thresh}): {row.get(f'ratio_{thresh}', 'N/A')}")
    
    if save_results and PANDAS_AVAILABLE:
        df = pd.DataFrame(aggregation_results)
        save_csv(df, "aggregation_method_comparison.csv")
    
    return {"methods_tested": methods, "thresholds": thresholds, "results": aggregation_results}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Running Volume Inference Tests...")
    
    results, summary = run_volume_inference_tests(
        max_patients=3,
        max_slices_per_patient=5
    )
    
    if results:
        test_aggregation_methods(results)
    
    print("\nDone!")
