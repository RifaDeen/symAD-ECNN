# GitHub Setup and Google Colab Integration Guide

## 🎯 Overview

This guide shows you how to:
1. Initialize Git repository locally
2. Create GitHub repository
3. Push your project to GitHub
4. Connect GitHub to Google Colab
5. Run your notebooks in Colab from GitHub

---

## 📋 Part 1: Local Git Setup (Windows)

### Step 1: Check if Git is Installed

Open PowerShell and run:
```powershell
git --version
```

**If not installed**:
- Download from: https://git-scm.com/download/win
- Install with default settings
- Restart PowerShell

---

### Step 2: Configure Git (First Time Only)

```powershell
# Set your name and email (used for commits)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

**Example**:
```powershell
git config --global user.name "Rifa Badurdeen"
git config --global user.email "w1954060@westminster.ac.uk"
```

---

### Step 3: Initialize Git Repository

Navigate to your project folder:
```powershell
cd C:\Users\rifad\symAD-ECNN
```

Initialize Git:
```powershell
# Initialize repository
git init

# Check status
git status
```

---

### Step 4: Create .gitignore File

**Important**: Exclude large data files and temporary files from Git!

Create `.gitignore` file in your project root:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/

# Jupyter Notebook
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# Data folders (too large for GitHub)
data/brats2021/
data/processed_ixi/
*.npy
*.nii
*.nii.gz
*.zip

# Model checkpoints (too large)
models/saved_models/*.pth
results/*.pkl

# OS
.DS_Store
Thumbs.db
desktop.ini

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Temporary files
*.log
*.tmp
temp/
```

**Why exclude data?**
- Data files are huge (GBs) - GitHub has 100MB file limit
- You'll use Google Drive for data storage instead
- Only code and documentation go in GitHub

---

### Step 5: Stage and Commit Files

```powershell
# Add all files (respecting .gitignore)
git add .

# Check what will be committed
git status

# Commit with descriptive message
git commit -m "Initial commit: SymAD-ECNN project setup with 3 model notebooks and documentation"
```

---

## 🌐 Part 2: Create GitHub Repository

### Step 1: Create Repository on GitHub

1. Go to https://github.com
2. Log in (or create account if needed)
3. Click **"New repository"** (green button) or go to https://github.com/new

4. **Repository settings**:
   - **Name**: `symAD-ECNN` or `brain-mri-anomaly-detection`
   - **Description**: "Symmetry-Aware Anomaly Detection with E(2)-Equivariant CNN for Brain MRI"
   - **Public** or **Private**: Your choice
     - Public: Anyone can see (good for portfolio)
     - Private: Only you can see (good for thesis until published)
   - **DO NOT** initialize with README (you already have one)
   - **DO NOT** add .gitignore (you already have one)
   - Click **"Create repository"**

---

### Step 2: Connect Local Repository to GitHub

GitHub will show you commands - copy the HTTPS URL (looks like `https://github.com/yourusername/symAD-ECNN.git`)

```powershell
# Add remote repository (replace with YOUR GitHub URL)
git remote add origin https://github.com/yourusername/symAD-ECNN.git

# Verify remote was added
git remote -v

# Push code to GitHub
git push -u origin main
```

**If it says "branch 'master' instead of 'main'"**:
```powershell
# Rename branch to main (GitHub's default)
git branch -M main
git push -u origin main
```

**Authentication**:
- GitHub may ask for username/password
- **Password = Personal Access Token** (not your GitHub password!)
- If needed: https://github.com/settings/tokens → Generate new token → Copy it

---

### Step 3: Verify Upload

Go to your GitHub repository URL:
`https://github.com/yourusername/symAD-ECNN`

You should see:
- ✅ README.md displayed
- ✅ All your folders (notebooks, md_files, etc.)
- ✅ All documentation files
- ❌ No data files (excluded by .gitignore)

---

## 🔗 Part 3: Connect GitHub to Google Colab

### Method 1: Open Notebook Directly from GitHub (Easiest)

1. Go to **Google Colab**: https://colab.research.google.com/
2. Click **"GitHub"** tab in the file dialog
3. Enter your GitHub username or paste repository URL
4. You'll see list of all `.ipynb` files in your repo
5. Click any notebook to open it in Colab!

**Example**:
```
Repository: yourusername/symAD-ECNN
Branch: main

Notebooks shown:
- notebooks/brats2021_t1_preprocessing.ipynb
- notebooks/models/01_baseline_autoencoder.ipynb
- notebooks/models/02_cnn_autoencoder.ipynb
- notebooks/models/03_ecnn_autoencoder.ipynb
```

---

### Method 2: Use Direct URL

Construct Colab URL manually:
```
https://colab.research.google.com/github/yourusername/symAD-ECNN/blob/main/notebooks/models/01_baseline_autoencoder.ipynb
```

**Template**:
```
https://colab.research.google.com/github/[USERNAME]/[REPO]/blob/[BRANCH]/[PATH_TO_NOTEBOOK]
```

**Example**:
```
https://colab.research.google.com/github/rifabadurdeen/symAD-ECNN/blob/main/notebooks/models/01_baseline_autoencoder.ipynb
```

---

### Method 3: Clone Repository in Colab (For Development)

In a Colab notebook cell:

```python
# Clone your repository
!git clone https://github.com/yourusername/symAD-ECNN.git

# Navigate to project
%cd symAD-ECNN

# Verify files
!ls -la
```

**When to use**: If you want to run multiple notebooks or need supporting files

---

## 📊 Part 4: Workflow - Local → GitHub → Colab

### Recommended Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR WORKFLOW                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. DEVELOP LOCALLY (Windows)                               │
│     • Edit notebooks                                        │
│     • Write documentation                                   │
│     • Preprocess data                                       │
│                                                             │
│  2. COMMIT TO GIT                                           │
│     git add .                                               │
│     git commit -m "Update description"                      │
│                                                             │
│  3. PUSH TO GITHUB                                          │
│     git push origin main                                    │
│                                                             │
│  4. TRAIN IN COLAB                                          │
│     • Open notebook from GitHub                             │
│     • Mount Google Drive (data already uploaded)            │
│     • Run training                                          │
│     • Download results                                      │
│                                                             │
│  5. UPDATE LOCALLY                                          │
│     • Add results to documentation                          │
│     • Commit and push updates                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 💾 Part 5: Data Management Strategy

### Why NOT Put Data in GitHub?

❌ **GitHub limitations**:
- 100 MB file size limit
- 1 GB repository size limit
- Your data: ~2-5 GB (too large!)

✅ **Use Google Drive instead**:

```
Your Data Storage:
├── GitHub (Code only)
│   ├── Notebooks (.ipynb files)
│   ├── Documentation (.md files)
│   ├── Scripts (.py files)
│   └── README.md
│
└── Google Drive (Data only)
    ├── MyDrive/symAD-ECNN/
    │   ├── data/
    │   │   ├── processed_ixi/resized_ixi/  (16,771 .npy files)
    │   │   └── brats2021_test/             (~1,500 .npy files)
    │   ├── models/saved_models/            (trained .pth files)
    │   └── results/                        (plots, JSON results)
```

### In Your Colab Notebooks

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Your data paths (already in your notebooks!)
BASE_PATH = "/content/drive/MyDrive/symAD-ECNN"
IXI_PATH = f"{BASE_PATH}/data/processed_ixi/resized_ixi"
BRATS_PATH = f"{BASE_PATH}/data/brats2021_test"
```

---

## 🔄 Part 6: Common Git Commands

### Daily Workflow

```powershell
# Check what changed
git status

# See differences
git diff

# Add files
git add .                    # Add all changes
git add specific_file.py     # Add specific file

# Commit changes
git commit -m "Descriptive message about what you changed"

# Push to GitHub
git push

# Pull latest from GitHub (if working from multiple computers)
git pull
```

### Useful Commands

```powershell
# View commit history
git log --oneline

# Undo changes to a file (before commit)
git checkout -- filename.py

# Create a new branch (for experiments)
git checkout -b experiment-branch

# Switch back to main branch
git checkout main

# View remote URL
git remote -v
```

---

## 🎓 Part 7: Best Practices

### Commit Messages

✅ **Good**:
```
git commit -m "Add rotation invariance test to ECNN notebook"
git commit -m "Update documentation with proposal references"
git commit -m "Fix batch size in CNN-AE training loop"
```

❌ **Bad**:
```
git commit -m "update"
git commit -m "changes"
git commit -m "asdf"
```

### Commit Frequency

- ✅ Commit after completing each logical unit of work
- ✅ Before trying something experimental (easy to undo)
- ✅ At end of each work session
- ❌ Don't commit broken/incomplete code to main branch

### What to Commit

✅ **YES** (Put in GitHub):
- Source code (.py files)
- Notebooks (.ipynb files)
- Documentation (.md files)
- README
- Configuration files
- Small example images (<1MB)

❌ **NO** (Keep out of GitHub):
- Large data files (.npy, .nii, .zip)
- Trained models (.pth files if >100MB)
- Temporary files
- IDE settings
- __pycache__, .ipynb_checkpoints

---

## 📱 Part 8: Adding Colab Badges to README

Make it easy for anyone to open notebooks in Colab!

### Add to Your README.md

```markdown
## 🚀 Quick Start - Open in Colab

[![Open Baseline in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/symAD-ECNN/blob/main/notebooks/models/01_baseline_autoencoder.ipynb)
[![Open CNN in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/symAD-ECNN/blob/main/notebooks/models/02_cnn_autoencoder.ipynb)
[![Open ECNN in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/symAD-ECNN/blob/main/notebooks/models/03_ecnn_autoencoder.ipynb)
```

**Replace `yourusername` with your actual GitHub username!**

Result: Clickable badges that open notebooks directly in Colab! 🎉

---

## 🛠️ Part 9: Troubleshooting

### Problem: "fatal: not a git repository"

**Solution**:
```powershell
cd C:\Users\rifad\symAD-ECNN
git init
```

### Problem: "remote origin already exists"

**Solution**:
```powershell
git remote remove origin
git remote add origin https://github.com/yourusername/symAD-ECNN.git
```

### Problem: "failed to push some refs"

**Solution**:
```powershell
# Pull first, then push
git pull origin main --allow-unrelated-histories
git push origin main
```

### Problem: GitHub asks for password but won't accept it

**Solution**:
- GitHub doesn't accept password anymore
- Need Personal Access Token (PAT)
- Generate at: https://github.com/settings/tokens
- Use token as password

### Problem: File too large (>100MB)

**Solution**:
```powershell
# Remove from staging
git rm --cached large_file.pth

# Add to .gitignore
echo "*.pth" >> .gitignore
echo "*.npy" >> .gitignore

# Commit
git add .gitignore
git commit -m "Add large files to gitignore"
```

---

## 📋 Part 10: Step-by-Step Checklist

### Initial Setup (Do Once)

- [ ] Install Git on Windows
- [ ] Configure Git username and email
- [ ] Create GitHub account (if needed)
- [ ] Navigate to project folder: `cd C:\Users\rifad\symAD-ECNN`
- [ ] Initialize Git: `git init`
- [ ] Create `.gitignore` file (copy from this guide)
- [ ] Stage all files: `git add .`
- [ ] First commit: `git commit -m "Initial commit: SymAD-ECNN project"`
- [ ] Create GitHub repository
- [ ] Connect local to remote: `git remote add origin <URL>`
- [ ] Push to GitHub: `git push -u origin main`
- [ ] Verify on GitHub website

### Daily Workflow (Repeat as Needed)

- [ ] Make changes to files
- [ ] Check status: `git status`
- [ ] Stage changes: `git add .`
- [ ] Commit: `git commit -m "Descriptive message"`
- [ ] Push: `git push`

### Training in Colab

- [ ] Upload preprocessed data to Google Drive
- [ ] Open notebook from GitHub in Colab
- [ ] Mount Google Drive
- [ ] Run training
- [ ] Download results
- [ ] Update documentation locally
- [ ] Commit and push updates

---

## 🎯 Quick Reference Commands

```powershell
# Setup (once)
git init
git remote add origin <URL>

# Daily workflow
git status              # Check changes
git add .               # Stage all
git commit -m "msg"     # Commit
git push                # Upload to GitHub

# Undo changes
git checkout -- file    # Discard changes to file
git reset HEAD file     # Unstage file

# Branch management
git branch              # List branches
git checkout -b new     # Create new branch
git checkout main       # Switch to main

# View history
git log --oneline       # Commit history
git diff                # See changes
```

---

## 🔗 Useful Links

- **Git Download**: https://git-scm.com/download/win
- **GitHub**: https://github.com
- **GitHub Tokens**: https://github.com/settings/tokens
- **Google Colab**: https://colab.research.google.com
- **Git Cheat Sheet**: https://education.github.com/git-cheat-sheet-education.pdf

---

## ✅ Summary

**Your Complete Setup**:
1. ✅ Local code in: `C:\Users\rifad\symAD-ECNN`
2. ✅ GitHub repository: Code and documentation (no data)
3. ✅ Google Drive: Data files (~2-5GB)
4. ✅ Google Colab: Training environment (free tier!)

**Workflow**:
```
Local (Code) → Git → GitHub (Backup) → Colab (Training)
                                          ↓
Google Drive (Data) ←──────────────── Results
```

**You're all set!** 🚀
