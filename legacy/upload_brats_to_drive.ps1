# BraTS Data Upload Guide
# Quick script to help upload BraTS data to Google Drive

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  BraTS Data Upload to Google Drive" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if BraTS data exists locally
$bratsPath = "C:\Users\rifad\symAD-ECNN\data\brats2021"

if (-not (Test-Path $bratsPath)) {
    Write-Host "[ERROR] BraTS data not found at: $bratsPath" -ForegroundColor Red
    exit
}

# Count folders
$folderCount = (Get-ChildItem -Path $bratsPath -Directory | Where-Object {$_.Name -like "BraTS*"}).Count
$totalSize = (Get-ChildItem -Path $bratsPath -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB

Write-Host "[INFO] BraTS Data Summary:" -ForegroundColor Yellow
Write-Host "  Location: $bratsPath" -ForegroundColor White
Write-Host "  Patient folders: $folderCount" -ForegroundColor White
Write-Host "  Total size: $([math]::Round($totalSize, 2)) GB" -ForegroundColor White
Write-Host ""

# Upload options
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Upload Options" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Choose upload method:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Google Drive Desktop App (RECOMMENDED - Fastest)" -ForegroundColor Green
Write-Host "   - Automatic sync" -ForegroundColor Gray
Write-Host "   - Can pause/resume" -ForegroundColor Gray
Write-Host "   - Works in background" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Web Upload (Slower)" -ForegroundColor Yellow
Write-Host "   - Via drive.google.com" -ForegroundColor Gray
Write-Host "   - Drag and drop folder" -ForegroundColor Gray
Write-Host "   - Takes 30-60 minutes" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Compress First (Faster upload)" -ForegroundColor Cyan
Write-Host "   - Create ZIP file first" -ForegroundColor Gray
Write-Host "   - Upload single file" -ForegroundColor Gray
Write-Host "   - Extract in Colab" -ForegroundColor Gray
Write-Host ""

$choice = Read-Host "Enter choice (1/2/3)"

# Option 1: Drive Desktop
if ($choice -eq "1") {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Google Drive Desktop Setup" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Check if Drive is installed
    $driveExe = "C:\Program Files\Google\Drive File Stream\GoogleDriveFS.exe"
    $driveExeNew = "C:\Program Files\Google\GoogleDrive\GoogleDriveFS.exe"
    
    $driveInstalled = (Test-Path $driveExe) -or (Test-Path $driveExeNew)
    
    if ($driveInstalled) {
        Write-Host "[OK] Google Drive Desktop is installed" -ForegroundColor Green
        Write-Host ""
        Write-Host "Steps to upload:" -ForegroundColor Yellow
        Write-Host "  1. Open File Explorer" -ForegroundColor White
        Write-Host "  2. Go to 'Google Drive' (usually in left sidebar)" -ForegroundColor White
        Write-Host "  3. Navigate to: My Drive\symAD-ECNN\data\" -ForegroundColor White
        Write-Host "  4. Copy this folder into it:" -ForegroundColor White
        Write-Host "     $bratsPath" -ForegroundColor Cyan
        Write-Host "  5. Wait for sync to complete (check Drive icon in taskbar)" -ForegroundColor White
        Write-Host ""
        
        Write-Host "Open Google Drive folder now? (Y/N)" -ForegroundColor Yellow
        $open = Read-Host
        
        if ($open -eq "Y" -or $open -eq "y") {
            # Try to open Drive folder
            if (Test-Path "G:\My Drive") {
                explorer "G:\My Drive"
            } elseif (Test-Path "$env:USERPROFILE\Google Drive") {
                explorer "$env:USERPROFILE\Google Drive"
            } else {
                Write-Host "Could not find Drive folder. Please open manually." -ForegroundColor Yellow
            }
        }
        
    } else {
        Write-Host "[INFO] Google Drive Desktop not detected" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Download and install from:" -ForegroundColor White
        Write-Host "  https://www.google.com/drive/download/" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "After installing:" -ForegroundColor Yellow
        Write-Host "  1. Sign in with your Google account" -ForegroundColor White
        Write-Host "  2. Wait for Drive to sync" -ForegroundColor White
        Write-Host "  3. Run this script again" -ForegroundColor White
        Write-Host ""
        
        Write-Host "Open download page? (Y/N)" -ForegroundColor Yellow
        $openWeb = Read-Host
        
        if ($openWeb -eq "Y" -or $openWeb -eq "y") {
            Start-Process "https://www.google.com/drive/download/"
        }
    }
}

# Option 2: Web Upload
elseif ($choice -eq "2") {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Web Upload Instructions" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "Steps:" -ForegroundColor Yellow
    Write-Host "  1. Go to https://drive.google.com" -ForegroundColor White
    Write-Host "  2. Navigate to: My Drive > symAD-ECNN > data" -ForegroundColor White
    Write-Host "  3. Click 'New' > 'Folder upload'" -ForegroundColor White
    Write-Host "  4. Select folder: $bratsPath" -ForegroundColor Cyan
    Write-Host "  5. Wait for upload (~30-60 minutes for $([math]::Round($totalSize, 1)) GB)" -ForegroundColor White
    Write-Host ""
    
    Write-Host "Open Google Drive? (Y/N)" -ForegroundColor Yellow
    $openDrive = Read-Host
    
    if ($openDrive -eq "Y" -or $openDrive -eq "y") {
        Start-Process "https://drive.google.com"
    }
}

# Option 3: Compress First
elseif ($choice -eq "3") {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Compress BraTS Data" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    $zipPath = "C:\Users\rifad\symAD-ECNN\data\brats2021.zip"
    
    Write-Host "Creating ZIP file..." -ForegroundColor Yellow
    Write-Host "This may take 5-10 minutes..." -ForegroundColor Gray
    Write-Host ""
    
    try {
        Compress-Archive -Path $bratsPath -DestinationPath $zipPath -Force
        
        $zipSize = (Get-Item $zipPath).Length / 1GB
        Write-Host "[OK] ZIP file created!" -ForegroundColor Green
        Write-Host "  Location: $zipPath" -ForegroundColor White
        Write-Host "  Size: $([math]::Round($zipSize, 2)) GB" -ForegroundColor White
        Write-Host ""
        
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "  1. Go to https://drive.google.com" -ForegroundColor White
        Write-Host "  2. Navigate to: My Drive > symAD-ECNN > data" -ForegroundColor White
        Write-Host "  3. Upload file: $zipPath" -ForegroundColor Cyan
        Write-Host "  4. In Colab, extract with:" -ForegroundColor White
        Write-Host "     !unzip /content/drive/MyDrive/symAD-ECNN/data/brats2021.zip -d /content/drive/MyDrive/symAD-ECNN/data/" -ForegroundColor Gray
        Write-Host ""
        
        Write-Host "Open folder in Explorer? (Y/N)" -ForegroundColor Yellow
        $openFolder = Read-Host
        
        if ($openFolder -eq "Y" -or $openFolder -eq "y") {
            explorer "C:\Users\rifad\symAD-ECNN\data"
        }
        
    } catch {
        Write-Host "[ERROR] Failed to create ZIP file" -ForegroundColor Red
        Write-Host "Error: $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Upload Tips" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "- Upload during off-peak hours for faster speed" -ForegroundColor White
Write-Host "- Keep computer on during upload" -ForegroundColor White
Write-Host "- Use stable internet connection" -ForegroundColor White
Write-Host "- Drive Desktop app is most reliable" -ForegroundColor White
Write-Host ""
