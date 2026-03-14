"""
Path utilities for SymAD-ECNN evaluation package.

This module provides functions to locate project files in Google Drive,
with fallback recursive search when files are not found in expected locations.

Purpose: Drive-aware path resolution for Colab execution
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple
import os


# =============================================================================
# DEFAULT SEARCH PATHS
# =============================================================================

# Primary Google Drive project root
DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/symAD-ECNN")

# Expected locations for various file types
EXPECTED_MODEL_PATHS = [
    "models/saved_models",
    "models",
    "demo_app/backend",
]

EXPECTED_RESULTS_PATHS = [
    "results",
    "results/ecnn_autoencoder",
    "results/cnn_autoencoder",
    "results/baseline",
    "demo_app/backend",
]

EXPECTED_DATA_PATHS = [
    "data/ixi_t1",
    "data/brats_t1",
    "data",
]

EXPECTED_BACKEND_PATHS = [
    "demo_app/backend",
    "backend",
]

EXPECTED_FRONTEND_PATHS = [
    "demo_app/frontend",
    "frontend",
]


# =============================================================================
# CORE PATH FUNCTIONS
# =============================================================================

def get_drive_project_root() -> Path:
    """
    Get the Google Drive project root path.
    
    Returns:
        Path: The project root path in Google Drive.
        
    Raises:
        FileNotFoundError: If Drive is not mounted or project folder not found.
    """
    if DEFAULT_DRIVE_ROOT.exists():
        return DEFAULT_DRIVE_ROOT
    
    # Check if Drive is mounted at all
    drive_mount = Path("/content/drive")
    if not drive_mount.exists():
        raise FileNotFoundError(
            "Google Drive is not mounted. Please run:\n"
            "  from google.colab import drive\n"
            "  drive.mount('/content/drive')"
        )
    
    # Search for project folder in Drive
    my_drive = Path("/content/drive/MyDrive")
    if my_drive.exists():
        # Try common variations of the project name
        variations = ["symAD-ECNN", "symad-ecnn", "SymAD-ECNN", "symAD_ECNN"]
        for name in variations:
            candidate = my_drive / name
            if candidate.exists():
                return candidate
    
    raise FileNotFoundError(
        f"Project folder not found. Expected: {DEFAULT_DRIVE_ROOT}\n"
        "Please ensure the project is uploaded to Google Drive."
    )


def _recursive_search(root: Path, pattern: str, max_depth: int = 5) -> List[Path]:
    """
    Recursively search for files matching a pattern.
    
    Args:
        root: Root directory to search from.
        pattern: Glob pattern to match (e.g., "*.pth", "*.json").
        max_depth: Maximum recursion depth to prevent excessive searching.
        
    Returns:
        List of matching file paths.
    """
    matches = []
    
    def search(directory: Path, depth: int):
        if depth > max_depth:
            return
        try:
            for item in directory.iterdir():
                if item.is_file() and item.match(pattern):
                    matches.append(item)
                elif item.is_dir() and not item.name.startswith('.'):
                    search(item, depth + 1)
        except PermissionError:
            pass
    
    if root.exists():
        search(root, 0)
    
    return matches


def _find_file_with_fallback(
    root: Path,
    expected_paths: List[str],
    patterns: List[str],
    file_type_name: str
) -> Tuple[Optional[Path], List[Path]]:
    """
    Find a file by checking expected paths first, then falling back to recursive search.
    
    Args:
        root: Project root directory.
        expected_paths: List of relative paths to check first.
        patterns: Glob patterns to match.
        file_type_name: Human-readable name for error messages.
        
    Returns:
        Tuple of (found_path or None, list of all candidates found).
    """
    candidates = []
    
    # Check expected locations first
    for rel_path in expected_paths:
        check_dir = root / rel_path
        if check_dir.exists():
            for pattern in patterns:
                for match in check_dir.glob(pattern):
                    if match.is_file():
                        candidates.append(match)
    
    # If found in expected locations, return the first match
    if candidates:
        return candidates[0], candidates
    
    # Fall back to recursive search
    for pattern in patterns:
        found = _recursive_search(root, pattern)
        candidates.extend(found)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)
    
    if unique_candidates:
        return unique_candidates[0], unique_candidates
    
    return None, []


# =============================================================================
# SPECIFIC FILE FINDER FUNCTIONS
# =============================================================================

def find_ecnn_checkpoint(root: Optional[Path] = None) -> Tuple[Optional[Path], List[Path]]:
    """
    Find the ECNN model checkpoint file.
    
    Args:
        root: Project root (defaults to Drive project root).
        
    Returns:
        Tuple of (best match path, list of all candidates).
        
    The function searches for files matching:
        - ecnn_optimized_best.pth
        - ecnn*.pth
        - *ecnn*.pth
    """
    if root is None:
        root = get_drive_project_root()
    
    # Prioritized patterns
    patterns = [
        "ecnn_optimized_best.pth",
        "ecnn_best.pth",
        "ecnn*.pth",
        "*ecnn*.pth",
    ]
    
    return _find_file_with_fallback(root, EXPECTED_MODEL_PATHS, patterns, "ECNN checkpoint")


def find_backend_api(root: Optional[Path] = None) -> Tuple[Optional[Path], List[Path]]:
    """
    Find the Flask backend API file.
    
    Args:
        root: Project root (defaults to Drive project root).
        
    Returns:
        Tuple of (best match path, list of all candidates).
    """
    if root is None:
        root = get_drive_project_root()
    
    patterns = ["api.py", "app.py", "server.py", "main.py"]
    
    return _find_file_with_fallback(root, EXPECTED_BACKEND_PATHS, patterns, "Backend API")


def find_streamlit_app(root: Optional[Path] = None) -> Tuple[Optional[Path], List[Path]]:
    """
    Find the Streamlit frontend application file.
    
    Args:
        root: Project root (defaults to Drive project root).
        
    Returns:
        Tuple of (best match path, list of all candidates).
    """
    if root is None:
        root = get_drive_project_root()
    
    patterns = ["streamlit_app.py", "app.py", "frontend.py", "ui.py"]
    
    return _find_file_with_fallback(root, EXPECTED_FRONTEND_PATHS, patterns, "Streamlit app")


def find_metrics_json(root: Optional[Path] = None) -> Tuple[Optional[Path], List[Path]]:
    """
    Find the ECNN metrics JSON file.
    
    Args:
        root: Project root (defaults to Drive project root).
        
    Returns:
        Tuple of (best match path, list of all candidates).
    """
    if root is None:
        root = get_drive_project_root()
    
    patterns = [
        "metrics_ecnn_v3.json",
        "metrics_ecnn*.json",
        "metrics*.json",
        "*ecnn*metrics*.json",
    ]
    
    expected_paths = EXPECTED_BACKEND_PATHS + EXPECTED_RESULTS_PATHS
    
    return _find_file_with_fallback(root, expected_paths, patterns, "Metrics JSON")


def find_results_jsons(root: Optional[Path] = None) -> Dict[str, List[Path]]:
    """
    Find all model result JSON files.
    
    Args:
        root: Project root (defaults to Drive project root).
        
    Returns:
        Dictionary mapping model type to list of result JSON paths.
    """
    if root is None:
        root = get_drive_project_root()
    
    results = {
        "all": [],
        "ecnn": [],
        "cnn_ae": [],
        "resnet": [],
        "baseline": [],
        "other": [],
    }
    
    # Search in expected results paths
    patterns = ["*.json", "results*.json", "*_results.json", "metrics*.json"]
    
    for rel_path in EXPECTED_RESULTS_PATHS:
        check_dir = root / rel_path
        if check_dir.exists():
            for pattern in patterns:
                for match in check_dir.glob(pattern):
                    if match.is_file():
                        results["all"].append(match)
                        
                        # Categorize by model type
                        name_lower = match.name.lower()
                        if "ecnn" in name_lower or "e2cnn" in name_lower:
                            results["ecnn"].append(match)
                        elif "resnet" in name_lower:
                            results["resnet"].append(match)
                        elif "cnn" in name_lower or "autoencoder" in name_lower:
                            results["cnn_ae"].append(match)
                        elif "baseline" in name_lower:
                            results["baseline"].append(match)
                        else:
                            results["other"].append(match)
    
    # Also do recursive search for any missed JSONs
    all_jsons = _recursive_search(root / "results", "*.json") if (root / "results").exists() else []
    for json_path in all_jsons:
        if json_path not in results["all"]:
            results["all"].append(json_path)
    
    # Remove duplicates
    for key in results:
        results[key] = list(set(results[key]))
    
    return results


def find_brats_raw_dirs(root: Optional[Path] = None) -> List[Path]:
    """
    Find BraTS raw data directories containing NIfTI files.
    
    Args:
        root: Project root (defaults to Drive project root).
        
    Returns:
        List of directories containing BraTS patient data.
    """
    if root is None:
        root = get_drive_project_root()
    
    brats_dirs = []
    
    # Check expected BraTS locations
    expected_brats = [
        root / "data" / "brats2021",
        root / "data" / "brats_raw",
        root / "data" / "BraTS2021",
    ]
    
    for brats_root in expected_brats:
        if brats_root.exists():
            # Look for patient directories (BraTS2021_XXXXX format)
            for item in brats_root.iterdir():
                if item.is_dir() and item.name.startswith("BraTS"):
                    # Verify it contains NIfTI files
                    nifti_files = list(item.glob("*.nii.gz")) + list(item.glob("*.nii"))
                    if nifti_files:
                        brats_dirs.append(item)
    
    return sorted(brats_dirs)


def find_data_paths(root: Optional[Path] = None) -> Dict[str, Optional[Path]]:
    """
    Find all dataset paths.
    
    Args:
        root: Project root (defaults to Drive project root).
        
    Returns:
        Dictionary with paths to different data directories.
    """
    if root is None:
        root = get_drive_project_root()
    
    data_paths = {
        "ixi_train": None,
        "ixi_val": None,
        "ixi_test": None,
        "brats_test": None,
        "brats_raw": None,
    }
    
    # Check for IXI data
    ixi_root = root / "data" / "ixi_t1"
    if ixi_root.exists():
        for split in ["train", "val", "test"]:
            split_dir = ixi_root / split
            if split_dir.exists():
                data_paths[f"ixi_{split}"] = split_dir
    
    # Check for processed BraTS test data
    brats_processed = root / "data" / "brats_t1"
    if brats_processed.exists():
        # Look for test or resized directory
        for subdir in ["test", "resized", "filtered"]:
            test_dir = brats_processed / subdir
            if test_dir.exists():
                data_paths["brats_test"] = test_dir
                break
        if data_paths["brats_test"] is None:
            data_paths["brats_test"] = brats_processed
    
    # Check for raw BraTS data
    brats_raw = root / "data" / "brats2021"
    if brats_raw.exists():
        data_paths["brats_raw"] = brats_raw
    
    return data_paths


# =============================================================================
# PATH VALIDATION AND REPORTING
# =============================================================================

def validate_paths(verbose: bool = True) -> Dict[str, Dict]:
    """
    Validate all expected project paths and report status.
    
    Args:
        verbose: If True, print detailed status report.
        
    Returns:
        Dictionary with validation results for each path type.
    """
    try:
        root = get_drive_project_root()
    except FileNotFoundError as e:
        if verbose:
            print(f"ERROR: {e}")
        return {"error": str(e)}
    
    validation = {
        "project_root": {"path": str(root), "exists": root.exists()},
        "ecnn_checkpoint": {"path": None, "exists": False, "candidates": []},
        "backend_api": {"path": None, "exists": False, "candidates": []},
        "streamlit_app": {"path": None, "exists": False, "candidates": []},
        "metrics_json": {"path": None, "exists": False, "candidates": []},
        "results_jsons": {"count": 0, "paths": []},
        "data_paths": {},
    }
    
    # Check ECNN checkpoint
    ecnn_path, ecnn_candidates = find_ecnn_checkpoint(root)
    validation["ecnn_checkpoint"]["path"] = str(ecnn_path) if ecnn_path else None
    validation["ecnn_checkpoint"]["exists"] = ecnn_path is not None
    validation["ecnn_checkpoint"]["candidates"] = [str(c) for c in ecnn_candidates]
    
    # Check backend API
    api_path, api_candidates = find_backend_api(root)
    validation["backend_api"]["path"] = str(api_path) if api_path else None
    validation["backend_api"]["exists"] = api_path is not None
    validation["backend_api"]["candidates"] = [str(c) for c in api_candidates]
    
    # Check Streamlit app
    st_path, st_candidates = find_streamlit_app(root)
    validation["streamlit_app"]["path"] = str(st_path) if st_path else None
    validation["streamlit_app"]["exists"] = st_path is not None
    validation["streamlit_app"]["candidates"] = [str(c) for c in st_candidates]
    
    # Check metrics JSON
    metrics_path, metrics_candidates = find_metrics_json(root)
    validation["metrics_json"]["path"] = str(metrics_path) if metrics_path else None
    validation["metrics_json"]["exists"] = metrics_path is not None
    validation["metrics_json"]["candidates"] = [str(c) for c in metrics_candidates]
    
    # Check results JSONs
    results = find_results_jsons(root)
    validation["results_jsons"]["count"] = len(results["all"])
    validation["results_jsons"]["paths"] = [str(p) for p in results["all"]]
    
    # Check data paths
    validation["data_paths"] = {
        k: str(v) if v else None for k, v in find_data_paths(root).items()
    }
    
    if verbose:
        print("=" * 60)
        print("Path Validation Report")
        print("=" * 60)
        print(f"\nProject Root: {validation['project_root']['path']}")
        print(f"  Exists: {validation['project_root']['exists']}")
        
        print(f"\nECNN Checkpoint:")
        print(f"  Found: {validation['ecnn_checkpoint']['exists']}")
        if validation['ecnn_checkpoint']['path']:
            print(f"  Path: {validation['ecnn_checkpoint']['path']}")
        elif validation['ecnn_checkpoint']['candidates']:
            print(f"  Candidates found: {len(validation['ecnn_checkpoint']['candidates'])}")
            for c in validation['ecnn_checkpoint']['candidates'][:3]:
                print(f"    - {c}")
        
        print(f"\nBackend API:")
        print(f"  Found: {validation['backend_api']['exists']}")
        if validation['backend_api']['path']:
            print(f"  Path: {validation['backend_api']['path']}")
        
        print(f"\nMetrics JSON:")
        print(f"  Found: {validation['metrics_json']['exists']}")
        if validation['metrics_json']['path']:
            print(f"  Path: {validation['metrics_json']['path']}")
        
        print(f"\nResults JSONs: {validation['results_jsons']['count']} files found")
        
        print(f"\nData Paths:")
        for key, path in validation['data_paths'].items():
            status = "Found" if path else "Not found"
            print(f"  {key}: {status}")
        
        print("=" * 60)
    
    return validation


def require_file(path: Optional[Path], file_type: str, candidates: List[Path] = None):
    """
    Require a file to exist, raising an informative error if not found.
    
    Args:
        path: The path that was found (or None).
        file_type: Human-readable description of the file type.
        candidates: List of candidate paths found during search.
        
    Raises:
        FileNotFoundError: If path is None, with helpful message.
    """
    if path is not None and Path(path).exists():
        return path
    
    msg = f"{file_type} not found."
    
    if candidates:
        msg += f"\n\nCandidates found ({len(candidates)}):"
        for c in candidates[:10]:
            msg += f"\n  - {c}"
        if len(candidates) > 10:
            msg += f"\n  ... and {len(candidates) - 10} more"
    else:
        msg += "\n\nNo candidate files found."
        msg += f"\n\nSearched under: {DEFAULT_DRIVE_ROOT}"
        msg += "\n\nPlease ensure:"
        msg += "\n  1. Google Drive is mounted"
        msg += "\n  2. Project files are uploaded to Drive"
        msg += f"\n  3. The {file_type} exists in the expected location"
    
    raise FileNotFoundError(msg)


if __name__ == "__main__":
    # Run validation when executed directly
    validate_paths(verbose=True)
