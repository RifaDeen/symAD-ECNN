# demo_app/run_demo.ps1
# Run: powershell -ExecutionPolicy Bypass -File demo_app/run_demo.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== SYMAD-ECNN Demo App Runner (Windows) ===" -ForegroundColor Cyan

# Go to repo root automatically (this script lives in demo_app/)
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO_ROOT = Resolve-Path (Join-Path $SCRIPT_DIR "..")
Set-Location $REPO_ROOT

# Optional: use the existing venv at repo root if you have it
$VENV_PATH = Join-Path $REPO_ROOT ".venv\Scripts\Activate.ps1"
if (Test-Path $VENV_PATH) {
    Write-Host "[OK] Activating .venv" -ForegroundColor Green
    & $VENV_PATH
} else {
    Write-Host "[WARN] .venv not found at repo root. Using current Python environment." -ForegroundColor Yellow
}

# Start Flask backend in a new window
Write-Host "[1/2] Starting Flask backend..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$REPO_ROOT\demo_app\backend'; python api.py"

Start-Sleep -Seconds 2

# Start Streamlit frontend in a new window
Write-Host "[2/2] Starting Streamlit frontend..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$REPO_ROOT\demo_app\frontend'; streamlit run streamlit_app.py"

Write-Host "`nDone! If it doesn't open automatically, go to:" -ForegroundColor Green
Write-Host "  Streamlit: http://localhost:8501" -ForegroundColor White
Write-Host "  API Health: http://localhost:5000/health" -ForegroundColor White
