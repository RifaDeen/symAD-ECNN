# SymAD-ECNN Git Setup - Quick Start Script
# Run this in PowerShell from your project directory

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SymAD-ECNN Git Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Git is installed
Write-Host "Checking Git installation..." -ForegroundColor Yellow
$gitCheck = Get-Command git -ErrorAction SilentlyContinue
if ($gitCheck) {
    $gitVersion = git --version
    Write-Host "[OK] Git found: $gitVersion" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Git not found! Please install from https://git-scm.com/download/win" -ForegroundColor Red
    exit
}

# Check current directory
$currentDir = Get-Location
Write-Host ""
Write-Host "Current directory: $currentDir" -ForegroundColor Yellow
Write-Host "Is this your SymAD-ECNN project folder? (Y/N)" -ForegroundColor Yellow
$confirm = Read-Host

if ($confirm -ne "Y" -and $confirm -ne "y") {
    Write-Host "Please navigate to your project folder first:" -ForegroundColor Red
    Write-Host "  cd C:\Users\rifad\symAD-ECNN" -ForegroundColor White
    exit
}

# Check if already initialized
if (Test-Path ".git") {
    Write-Host ""
    Write-Host "⚠ Git repository already initialized!" -ForegroundColor Yellow
    Write-Host "Do you want to continue anyway? (Y/N)" -ForegroundColor Yellow
    $continue = Read-Host
    if ($continue -ne "Y" -and $continue -ne "y") {
        exit
    }
} else {
    # Initialize Git
    Write-Host ""
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "[OK] Git initialized" -ForegroundColor Green
}

# Check if .gitignore exists
if (-not (Test-Path ".gitignore")) {
    Write-Host ""
    Write-Host "[ERROR] .gitignore not found!" -ForegroundColor Red
    Write-Host "Please create .gitignore file first (see GITHUB_COLAB_SETUP.md)" -ForegroundColor Yellow
    exit
} else {
    Write-Host "[OK] .gitignore found" -ForegroundColor Green
}

# Configure Git user (if not already set)
Write-Host ""
Write-Host "Checking Git configuration..." -ForegroundColor Yellow
$gitName = git config --global user.name
$gitEmail = git config --global user.email

if ([string]::IsNullOrEmpty($gitName) -or [string]::IsNullOrEmpty($gitEmail)) {
    Write-Host "Git user not configured. Please enter your details:" -ForegroundColor Yellow
    Write-Host ""
    
    $userName = Read-Host "Your name (for example Rifa Badurdeen)"
    $userEmail = Read-Host "Your email (for example w1954060@westminster.ac.uk)"
    
    git config --global user.name "$userName"
    git config --global user.email "$userEmail"
    
    Write-Host "[OK] Git configured" -ForegroundColor Green
} else {
    Write-Host "[OK] Git user: $gitName - $gitEmail" -ForegroundColor Green
}

# Check status
Write-Host ""
Write-Host "Checking repository status..." -ForegroundColor Yellow
git status --short | Select-Object -First 10

$fileCount = (git status --short | Measure-Object).Count
Write-Host ""
Write-Host "Found $fileCount files to track" -ForegroundColor Yellow

# Ask to stage files
Write-Host ""
Write-Host "Stage all files? (Y/N)" -ForegroundColor Yellow
$stageFiles = Read-Host

if ($stageFiles -eq "Y" -or $stageFiles -eq "y") {
    Write-Host "Staging files..." -ForegroundColor Yellow
    git add .
    Write-Host "[OK] Files staged" -ForegroundColor Green
    
    # Commit
    Write-Host ""
    Write-Host "Enter commit message (or press Enter for default):" -ForegroundColor Yellow
    $commitMsg = Read-Host
    
    if ([string]::IsNullOrEmpty($commitMsg)) {
        $commitMsg = "Initial commit: SymAD-ECNN project setup"
    }
    
    git commit -m "$commitMsg"
    Write-Host "[OK] Changes committed" -ForegroundColor Green
}

# GitHub setup
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GitHub Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Have you created a GitHub repository? (Y/N)" -ForegroundColor Yellow
$hasRepo = Read-Host

if ($hasRepo -eq "Y" -or $hasRepo -eq "y") {
    Write-Host ""
    Write-Host "Enter your GitHub repository URL:" -ForegroundColor Yellow
    Write-Host "  (for example https://github.com/yourusername/symAD-ECNN.git)" -ForegroundColor Gray
    $repoUrl = Read-Host
    
    # Check if remote already exists
    $existingRemote = git remote -v 2>$null
    if ($existingRemote) {
        Write-Host ""
        Write-Host "[WARNING] Remote 'origin' already exists" -ForegroundColor Yellow
        Write-Host "Remove existing remote and add new one? (Y/N)" -ForegroundColor Yellow
        $removeRemote = Read-Host
        
        if ($removeRemote -eq "Y" -or $removeRemote -eq "y") {
            git remote remove origin
            git remote add origin $repoUrl
            Write-Host "[OK] Remote updated" -ForegroundColor Green
        }
    } else {
        git remote add origin $repoUrl
        Write-Host "[OK] Remote added" -ForegroundColor Green
    }
    
    # Rename branch to main (if needed)
    $currentBranch = git branch --show-current
    if ($currentBranch -ne "main") {
        Write-Host ""
        Write-Host "Renaming branch to 'main'..." -ForegroundColor Yellow
        git branch -M main
        Write-Host "[OK] Branch renamed to 'main'" -ForegroundColor Green
    }
    
    # Push to GitHub
    Write-Host ""
    Write-Host "Push to GitHub? (Y/N)" -ForegroundColor Yellow
    $pushNow = Read-Host
    
    if ($pushNow -eq "Y" -or $pushNow -eq "y") {
        Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
        Write-Host "(You may need to enter your GitHub username and Personal Access Token)" -ForegroundColor Gray
        git push -u origin main
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Successfully pushed to GitHub!" -ForegroundColor Green
        } else {
            Write-Host "[ERROR] Push failed. Check your credentials and repository URL" -ForegroundColor Red
        }
    }
} else {
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Go to https://github.com/new" -ForegroundColor White
    Write-Host "  2. Create repository named 'symAD-ECNN'" -ForegroundColor White
    Write-Host "  3. Run this script again" -ForegroundColor White
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  - Upload data to Google Drive: MyDrive/symAD-ECNN/data/" -ForegroundColor White
Write-Host "  - Open notebooks in Colab from GitHub" -ForegroundColor White
Write-Host "  - See GITHUB_COLAB_SETUP.md for detailed instructions" -ForegroundColor White
Write-Host ""
