# Streamlit Web Interface Setup Script
# Run this script to set up the prototype environment

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  symAD-ECNN Streamlit Prototype Setup     " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python version
Write-Host "[1/5] Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "Python 3\.([0-9]+)") {
    $minorVersion = [int]$matches[1]
    if ($minorVersion -ge 8) {
        Write-Host "  ✓ Python version: $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Python 3.8+ required. Found: $pythonVersion" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "  ✗ Python not found or version check failed" -ForegroundColor Red
    exit 1
}

# Step 2: Create virtual environment (optional)
Write-Host ""
Write-Host "[2/5] Virtual environment setup..." -ForegroundColor Yellow
$createVenv = Read-Host "Create new virtual environment? (y/N)"
if ($createVenv -eq "y" -or $createVenv -eq "Y") {
    if (Test-Path "venv_streamlit") {
        Write-Host "  ⚠ venv_streamlit already exists. Skipping..." -ForegroundColor Yellow
    } else {
        python -m venv venv_streamlit
        Write-Host "  ✓ Created venv_streamlit" -ForegroundColor Green
        Write-Host "  → Activate with: .\venv_streamlit\Scripts\Activate.ps1" -ForegroundColor Cyan
    }
} else {
    Write-Host "  → Skipping virtual environment creation" -ForegroundColor Gray
}

# Step 3: Install dependencies
Write-Host ""
Write-Host "[3/5] Installing Streamlit dependencies..." -ForegroundColor Yellow
$pip = if (Test-Path "venv_streamlit\Scripts\pip.exe") { ".\venv_streamlit\Scripts\pip.exe" } else { "pip" }

Write-Host "  → Installing from requirements_streamlit.txt..." -ForegroundColor Gray
& $pip install -r requirements_streamlit.txt --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "  ✗ Installation failed. Check errors above." -ForegroundColor Red
    exit 1
}

# Step 4: Check model file
Write-Host ""
Write-Host "[4/5] Checking for trained model..." -ForegroundColor Yellow
$modelPath = "models\saved_models\ecnn_optimized_best.pth"
if (Test-Path $modelPath) {
    $modelSize = (Get-Item $modelPath).Length / 1MB
    Write-Host "  ✓ Model found: $modelPath ($([math]::Round($modelSize, 2)) MB)" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Model not found: $modelPath" -ForegroundColor Yellow
    Write-Host "    The app will still launch, but you'll need to train the model first." -ForegroundColor Gray
    Write-Host "    To train: Open notebooks/models/07_ecnn_autoencoder.ipynb" -ForegroundColor Gray
}

# Step 5: Create .streamlit config directory
Write-Host ""
Write-Host "[5/5] Setting up Streamlit configuration..." -ForegroundColor Yellow
if (-not (Test-Path ".streamlit")) {
    New-Item -ItemType Directory -Path ".streamlit" | Out-Null
    Write-Host "  ✓ Created .streamlit directory" -ForegroundColor Green
}

$configContent = @"
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
"@

Set-Content -Path ".streamlit\config.toml" -Value $configContent
Write-Host "  ✓ Created Streamlit config" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Setup Complete! 🎉                       " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host ""

if ($createVenv -eq "y" -or $createVenv -eq "Y") {
    Write-Host "1. Activate virtual environment:" -ForegroundColor White
    Write-Host "   .\venv_streamlit\Scripts\Activate.ps1" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "2. Launch Streamlit app:" -ForegroundColor White
    Write-Host "   streamlit run streamlit_app.py" -ForegroundColor Cyan
} else {
    Write-Host "1. Launch Streamlit app:" -ForegroundColor White
    Write-Host "   streamlit run streamlit_app.py" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "3. Open in browser:" -ForegroundColor White
Write-Host "   http://localhost:8501" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $modelPath)) {
    Write-Host "⚠ Remember: Train the model first if not already done!" -ForegroundColor Yellow
    Write-Host "  See: notebooks/models/07_ecnn_autoencoder.ipynb" -ForegroundColor Gray
}

Write-Host ""
Write-Host "For more details, see STREAMLIT_README.md" -ForegroundColor Gray
Write-Host ""
