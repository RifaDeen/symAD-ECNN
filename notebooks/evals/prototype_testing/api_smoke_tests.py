"""
API Smoke Tests for SymAD-ECNN Prototype.

This script tests the Flask backend API endpoints to verify
the prototype system is functioning correctly.

Supports dissertation Chapter 8 functional testing.

Author: SymAD-ECNN Project
Purpose: Functional testing of the deployed Flask backend
"""

import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DEFAULT_API_URL, API_HEALTH_ENDPOINT, API_PREDICT_ENDPOINT,
    API_TEST_TIMEOUT, JSON_DIR, TABLES_DIR, LOGS_DIR,
    ensure_directories_exist
)
from path_utils import (
    get_drive_project_root, find_backend_api, find_data_paths
)
from io_utils import (
    save_json, save_csv, log_message,
    start_experiment_log, end_experiment_log
)

# Try to import requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests library not available. Install with: pip install requests")


# =============================================================================
# API TEST CLASSES
# =============================================================================

class APITestResult:
    """Container for individual test results."""
    
    def __init__(
        self,
        test_name: str,
        endpoint: str,
        passed: bool,
        response_time: float = None,
        status_code: int = None,
        response_data: Dict = None,
        error_message: str = None
    ):
        self.test_name = test_name
        self.endpoint = endpoint
        self.passed = passed
        self.response_time = response_time
        self.status_code = status_code
        self.response_data = response_data
        self.error_message = error_message
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "endpoint": self.endpoint,
            "passed": self.passed,
            "response_time_ms": self.response_time * 1000 if self.response_time else None,
            "status_code": self.status_code,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }


class APITestSuite:
    """Suite of API tests for the Flask backend."""
    
    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: int = API_TEST_TIMEOUT):
        """
        Initialize test suite.
        
        Args:
            base_url: Base URL of the API.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.results: List[APITestResult] = []
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Tuple[Optional[requests.Response], float, Optional[str]]:
        """
        Make an HTTP request with timing.
        
        Args:
            method: HTTP method (GET, POST, etc.).
            endpoint: API endpoint.
            **kwargs: Additional request arguments.
            
        Returns:
            Tuple of (response, elapsed_time, error_message).
        """
        if not REQUESTS_AVAILABLE:
            return None, 0, "requests library not available"
        
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault('timeout', self.timeout)
        
        start_time = time.time()
        try:
            response = requests.request(method, url, **kwargs)
            elapsed = time.time() - start_time
            return response, elapsed, None
        except requests.exceptions.ConnectionError:
            elapsed = time.time() - start_time
            return None, elapsed, "Connection refused - is the server running?"
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            return None, elapsed, f"Request timed out after {self.timeout}s"
        except Exception as e:
            elapsed = time.time() - start_time
            return None, elapsed, str(e)
    
    def test_health_endpoint(self) -> APITestResult:
        """Test the health check endpoint."""
        response, elapsed, error = self._make_request('GET', API_HEALTH_ENDPOINT)
        
        if error:
            result = APITestResult(
                test_name="health_check",
                endpoint=API_HEALTH_ENDPOINT,
                passed=False,
                response_time=elapsed,
                error_message=error
            )
        elif response.status_code == 200:
            try:
                data = response.json()
                result = APITestResult(
                    test_name="health_check",
                    endpoint=API_HEALTH_ENDPOINT,
                    passed=True,
                    response_time=elapsed,
                    status_code=response.status_code,
                    response_data=data
                )
            except:
                result = APITestResult(
                    test_name="health_check",
                    endpoint=API_HEALTH_ENDPOINT,
                    passed=True,
                    response_time=elapsed,
                    status_code=response.status_code
                )
        else:
            result = APITestResult(
                test_name="health_check",
                endpoint=API_HEALTH_ENDPOINT,
                passed=False,
                response_time=elapsed,
                status_code=response.status_code,
                error_message=f"Unexpected status code: {response.status_code}"
            )
        
        self.results.append(result)
        return result
    
    def test_predict_missing_file(self) -> APITestResult:
        """Test prediction endpoint with missing file."""
        response, elapsed, error = self._make_request(
            'POST',
            API_PREDICT_ENDPOINT,
            data={}  # No file provided
        )
        
        # Expecting an error response (400 or similar)
        if error:
            result = APITestResult(
                test_name="predict_missing_file",
                endpoint=API_PREDICT_ENDPOINT,
                passed=False,
                response_time=elapsed,
                error_message=error
            )
        elif response.status_code in [400, 422]:
            # Expected behavior - API correctly rejects missing file
            result = APITestResult(
                test_name="predict_missing_file",
                endpoint=API_PREDICT_ENDPOINT,
                passed=True,  # This is expected behavior
                response_time=elapsed,
                status_code=response.status_code,
                response_data={"message": "Correctly rejected missing file"}
            )
        elif response.status_code == 200:
            # Unexpected - should have rejected
            result = APITestResult(
                test_name="predict_missing_file",
                endpoint=API_PREDICT_ENDPOINT,
                passed=False,
                response_time=elapsed,
                status_code=response.status_code,
                error_message="API accepted request without file - expected rejection"
            )
        else:
            result = APITestResult(
                test_name="predict_missing_file",
                endpoint=API_PREDICT_ENDPOINT,
                passed=False,
                response_time=elapsed,
                status_code=response.status_code,
                error_message=f"Unexpected status: {response.status_code}"
            )
        
        self.results.append(result)
        return result
    
    def test_predict_invalid_file_type(self) -> APITestResult:
        """Test prediction endpoint with invalid file type."""
        # Create a fake text file
        import io
        fake_file = io.BytesIO(b"This is not an image")
        
        response, elapsed, error = self._make_request(
            'POST',
            API_PREDICT_ENDPOINT,
            files={'file': ('test.txt', fake_file, 'text/plain')}
        )
        
        if error:
            result = APITestResult(
                test_name="predict_invalid_file_type",
                endpoint=API_PREDICT_ENDPOINT,
                passed=False,
                response_time=elapsed,
                error_message=error
            )
        elif response.status_code in [400, 415, 422]:
            # Expected - API rejects invalid file type
            result = APITestResult(
                test_name="predict_invalid_file_type",
                endpoint=API_PREDICT_ENDPOINT,
                passed=True,
                response_time=elapsed,
                status_code=response.status_code,
                response_data={"message": "Correctly rejected invalid file type"}
            )
        else:
            result = APITestResult(
                test_name="predict_invalid_file_type",
                endpoint=API_PREDICT_ENDPOINT,
                passed=False,
                response_time=elapsed,
                status_code=response.status_code,
                error_message=f"Unexpected response to invalid file type: {response.status_code}"
            )
        
        self.results.append(result)
        return result
    
    def test_predict_with_valid_image(
        self,
        image_path: Optional[Path] = None
    ) -> APITestResult:
        """
        Test prediction endpoint with a valid image.
        
        Args:
            image_path: Path to a valid test image.
        """
        if image_path is None or not Path(image_path).exists():
            result = APITestResult(
                test_name="predict_valid_image",
                endpoint=API_PREDICT_ENDPOINT,
                passed=False,
                error_message="No valid test image available"
            )
            self.results.append(result)
            return result
        
        image_path = Path(image_path)
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, 'image/png')}
                response, elapsed, error = self._make_request(
                    'POST',
                    API_PREDICT_ENDPOINT,
                    files=files
                )
        except Exception as e:
            result = APITestResult(
                test_name="predict_valid_image",
                endpoint=API_PREDICT_ENDPOINT,
                passed=False,
                error_message=f"Failed to read test image: {e}"
            )
            self.results.append(result)
            return result
        
        if error:
            result = APITestResult(
                test_name="predict_valid_image",
                endpoint=API_PREDICT_ENDPOINT,
                passed=False,
                response_time=elapsed,
                error_message=error
            )
        elif response.status_code == 200:
            try:
                data = response.json()
                result = APITestResult(
                    test_name="predict_valid_image",
                    endpoint=API_PREDICT_ENDPOINT,
                    passed=True,
                    response_time=elapsed,
                    status_code=response.status_code,
                    response_data=data
                )
            except:
                result = APITestResult(
                    test_name="predict_valid_image",
                    endpoint=API_PREDICT_ENDPOINT,
                    passed=True,
                    response_time=elapsed,
                    status_code=response.status_code
                )
        else:
            result = APITestResult(
                test_name="predict_valid_image",
                endpoint=API_PREDICT_ENDPOINT,
                passed=False,
                response_time=elapsed,
                status_code=response.status_code,
                error_message=f"Prediction failed: {response.status_code}"
            )
        
        self.results.append(result)
        return result
    
    def run_all_tests(
        self,
        test_image_path: Optional[Path] = None,
        verbose: bool = True
    ) -> List[APITestResult]:
        """
        Run all API tests.
        
        Args:
            test_image_path: Optional path to a valid test image.
            verbose: Whether to print results.
            
        Returns:
            List of test results.
        """
        if verbose:
            print("=" * 60)
            print("API SMOKE TESTS")
            print("=" * 60)
            print(f"Base URL: {self.base_url}")
            print(f"Timeout: {self.timeout}s")
            print("-" * 60)
        
        # Run tests
        tests = [
            ("Health Check", self.test_health_endpoint),
            ("Missing File", self.test_predict_missing_file),
            ("Invalid File Type", self.test_predict_invalid_file_type),
        ]
        
        for test_name, test_func in tests:
            if verbose:
                print(f"\nRunning: {test_name}...")
            result = test_func()
            if verbose:
                status = "PASS" if result.passed else "FAIL"
                print(f"  Result: {status}")
                if result.response_time:
                    print(f"  Response time: {result.response_time*1000:.1f}ms")
                if result.error_message:
                    print(f"  Error: {result.error_message}")
        
        # Test with valid image if provided
        if test_image_path:
            if verbose:
                print(f"\nRunning: Valid Image Prediction...")
            result = self.test_predict_with_valid_image(test_image_path)
            if verbose:
                status = "PASS" if result.passed else "FAIL"
                print(f"  Result: {status}")
                if result.response_data:
                    print(f"  Response: {result.response_data}")
        
        if verbose:
            print("\n" + "=" * 60)
            passed = sum(1 for r in self.results if r.passed)
            total = len(self.results)
            print(f"SUMMARY: {passed}/{total} tests passed")
            print("=" * 60)
        
        return self.results
    
    def get_summary(self) -> Dict:
        """Get test summary as dictionary."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        
        return {
            "total_tests": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.results) if self.results else 0,
            "base_url": self.base_url,
            "timestamp": datetime.now().isoformat(),
        }


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_api_smoke_tests(
    base_url: str = DEFAULT_API_URL,
    test_image_path: Optional[Path] = None,
    save_results: bool = True,
    verbose: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Run API smoke tests and optionally save results.
    
    Args:
        base_url: Base URL of the API to test.
        test_image_path: Optional path to a test image.
        save_results: Whether to save results to Drive.
        verbose: Whether to print progress.
        
    Returns:
        Tuple of (list of result dicts, summary dict).
    """
    if not REQUESTS_AVAILABLE:
        print("ERROR: requests library is required for API testing.")
        print("Install with: pip install requests")
        return [], {"error": "requests library not available"}
    
    ensure_directories_exist()
    
    # Start logging
    log_name = start_experiment_log(
        "api_smoke_tests",
        params={"base_url": base_url}
    )
    
    # Create test suite and run
    suite = APITestSuite(base_url=base_url)
    results = suite.run_all_tests(test_image_path, verbose)
    summary = suite.get_summary()
    
    # Convert results to dicts
    results_dicts = [r.to_dict() for r in results]
    
    # Save results
    if save_results:
        try:
            # Save individual test results
            save_json(
                {
                    "results": results_dicts,
                    "summary": summary
                },
                "api_smoke_test_results.json"
            )
            
            # Save health check result separately
            health_result = next((r for r in results if r.test_name == "health_check"), None)
            if health_result:
                save_json(health_result.to_dict(), "api_health_result.json")
            
            # Save predict test results
            predict_results = [r for r in results_dicts if "predict" in r["test_name"]]
            if predict_results:
                save_json(
                    {"tests": predict_results},
                    "api_predict_test_results.json"
                )
            
            # Save as CSV
            import pandas as pd
            df = pd.DataFrame(results_dicts)
            save_csv(df, "api_test_summary.csv")
            
            log_message("Results saved successfully.", log_name)
            
        except Exception as e:
            log_message(f"Error saving results: {e}", log_name)
    
    end_experiment_log(log_name, summary=summary)
    
    return results_dicts, summary


if __name__ == "__main__":
    print("Running API Smoke Tests...")
    
    # Try to find a test image
    try:
        data_paths = find_data_paths()
        test_image = None
        
        for key in ["ixi_test", "ixi_val", "brats_test"]:
            if data_paths.get(key):
                data_dir = Path(data_paths[key])
                images = list(data_dir.glob("*.png"))[:1]
                if images:
                    test_image = images[0]
                    break
        
        if test_image:
            print(f"Using test image: {test_image}")
        
    except Exception as e:
        print(f"Could not find test image: {e}")
        test_image = None
    
    results, summary = run_api_smoke_tests(
        test_image_path=test_image
    )
    
    print(f"\nFinal summary: {summary}")
