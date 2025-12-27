# Connecting Google Colab to VS Code

There are **two ways** to work with Google Colab and VS Code:

## Method 1: Edit Locally, Run in Colab (Recommended)

This is the **simplest and most reliable** approach for your SymAD-ECNN project.

### Workflow
1. **Edit notebooks in VS Code** (locally on your Windows machine)
2. **Commit and push** to GitHub
3. **Open in Colab** from GitHub (gets free GPU/TPU)
4. **Train models** in Colab
5. **Download results** back to your project

### Advantages
- ✅ Use VS Code's powerful editor
- ✅ Full Git integration
- ✅ Free Colab GPU for training
- ✅ No complex setup
- ✅ Works reliably

### How to Use
```bash
# In VS Code
1. Edit your notebook (e.g., 01_baseline_autoencoder.ipynb)
2. Save changes
3. Commit: git add . && git commit -m "Update baseline model"
4. Push: git push

# In Browser
5. Go to: https://github.com/RifaDeen/symAD-ECNN
6. Click on notebook file
7. Click "Open in Colab" badge (or use URL trick)
8. Run cells with GPU
9. Download results
```

---

## Method 2: Connect VS Code to Colab Runtime (Advanced)

This connects your **local VS Code** directly to a **Colab Python runtime**. You edit locally but code runs on Colab's GPU.

### ⚠️ Limitations
- Requires keeping Colab browser tab open
- Connection can be unstable
- More complex setup
- Free Colab has connection time limits

### Prerequisites
1. Google Colab Pro (recommended, but free tier works)
2. VS Code installed
3. Python extension installed in VS Code
4. `colabcode` or `colab-ssh` package

### Setup Steps

#### Step 1: Install VS Code Extensions
```bash
# In VS Code, install these extensions:
1. Python (Microsoft)
2. Jupyter (Microsoft)
3. Remote - SSH (Microsoft)
```

#### Step 2: Install Colab SSH Package

**In Colab notebook**, run:
```python
# Install colab-ssh
!pip install colab-ssh

# Import and setup
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="your_password_here")

# You'll get an SSH connection string like:
# ssh root@some-url.trycloudflare.com
```

#### Step 3: Connect VS Code

**In VS Code:**
1. Press `Ctrl+Shift+P`
2. Type "Remote-SSH: Connect to Host"
3. Enter the SSH string from Colab
4. Enter the password you set
5. VS Code will connect to Colab runtime

#### Step 4: Open Project Folder
```bash
# In VS Code connected to Colab:
1. File > Open Folder
2. Navigate to /content (Colab's working directory)
3. Clone your repo: git clone https://github.com/RifaDeen/symAD-ECNN.git
4. Open the cloned folder
```

### Alternative: Use Jupyter Extension

**Simpler approach** - don't need SSH:

1. **In Colab**, get runtime details:
   ```python
   # Run this in Colab
   from google.colab import auth
   auth.authenticate_user()
   
   # Get connection URL
   !jupyter notebook list
   ```

2. **In VS Code:**
   - Open `.ipynb` file
   - Click "Select Kernel" in top-right
   - Choose "Existing Jupyter Server"
   - Paste Colab URL
   - Enter token

---

## Method 3: Use Google Colab Extension (Easiest for Remote Connection)

### Install Colab Extension

**In Colab notebook:**
```python
!pip install colab-xterm
%load_ext colabxterm
%xterm

# Or use colab-code for VS Code in browser
!pip install git+https://github.com/WassimBenzarti/colab-code.git
from colab_code import ColabCode
ColabCode(port=10000, password="your_password", mount_drive=True)
```

This opens a **VS Code interface inside Colab browser** - not quite the same as local VS Code, but similar.

---

## Recommended Setup for SymAD-ECNN Project

For your use case, I recommend **Method 1** (Edit Locally, Run in Colab) because:

### Why Method 1 is Best
1. ✅ **Reliability** - No connection issues
2. ✅ **Simplicity** - Just push to GitHub and open in Colab
3. ✅ **Version Control** - Everything tracked in Git
4. ✅ **Free Tier Friendly** - No special requirements
5. ✅ **Best Practice** - Industry standard workflow

### Your Workflow (Recommended)
```bash
# Morning: Edit code locally
cd C:\Users\rifad\symAD-ECNN
code .  # Open in VS Code
# Edit 01_baseline_autoencoder.ipynb
git add .
git commit -m "Add data preprocessing"
git push

# Afternoon: Train in Colab
# 1. Go to https://github.com/RifaDeen/symAD-ECNN
# 2. Click on notebook
# 3. Open in Colab
# 4. Runtime > Change runtime type > GPU (T4)
# 5. Run all cells
# 6. Download trained model (.pth file)

# Evening: Save results
# Copy downloaded .pth to models/saved_models/
git add models/saved_models/baseline_ae_final.pth
git commit -m "Add trained baseline model"
git push
```

---

## Comparison Table

| Feature | Method 1 (Local Edit) | Method 2 (SSH Remote) | Method 3 (Browser Extension) |
|---------|----------------------|----------------------|---------------------------|
| **Ease of Setup** | ⭐⭐⭐⭐⭐ Easy | ⭐⭐ Complex | ⭐⭐⭐ Medium |
| **Stability** | ⭐⭐⭐⭐⭐ Very Stable | ⭐⭐ Can Disconnect | ⭐⭐⭐ Moderate |
| **Free Tier** | ✅ Works Great | ⚠️ Limited | ✅ Works |
| **Git Integration** | ✅ Native | ⚠️ Manual | ⚠️ Manual |
| **GPU Access** | ✅ Full Access | ✅ Full Access | ✅ Full Access |
| **Best For** | Production Work | Debugging | Quick Edits |

---

## Quick Start (Method 1 - Recommended)

### 1. Add Colab Badge to README

I'll update your README with a Colab badge:

```markdown
# SymAD-ECNN

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RifaDeen/symAD-ECNN/blob/main/01_baseline_autoencoder.ipynb)

## Quick Start in Colab
1. Click badge above
2. Runtime > Change runtime type > GPU
3. Run all cells
```

### 2. Access from GitHub

Direct URLs for your notebooks:
- **Baseline AE**: https://colab.research.google.com/github/RifaDeen/symAD-ECNN/blob/main/01_baseline_autoencoder.ipynb
- **CNN AE**: https://colab.research.google.com/github/RifaDeen/symAD-ECNN/blob/main/02_cnn_autoencoder.ipynb
- **ECNN AE**: https://colab.research.google.com/github/RifaDeen/symAD-ECNN/blob/main/03_ecnn_autoencoder.ipynb

### 3. Mount Google Drive in Colab

Add this cell at the top of each notebook:
```python
# Mount Google Drive for data access
from google.colab import drive
drive.mount('/content/drive')

# Set data path
DATA_PATH = '/content/drive/MyDrive/symAD-ECNN/data/'
```

---

## Troubleshooting

### Issue: "Can't find my data in Colab"
**Solution**: Upload data to Google Drive first
```bash
# Create folder structure in Google Drive:
MyDrive/
  symAD-ECNN/
    data/
      processed_ixi/
        train/
        val/
      brats2021/
```

### Issue: "Colab session disconnected"
**Solution**: Free tier disconnects after 12 hours. Save checkpoints frequently:
```python
# In notebook, add checkpoint saving
torch.save(model.state_dict(), '/content/drive/MyDrive/symAD-ECNN/models/checkpoint.pth')
```

### Issue: "Out of memory in Colab"
**Solution**: Reduce batch size
```python
# Change from 32 to 16
BATCH_SIZE = 16  # Reduced for free Colab
```

---

## Next Steps

1. **Push your project to GitHub** (you just did this!)
2. **Upload data to Google Drive** (`MyDrive/symAD-ECNN/data/`)
3. **Open notebooks in Colab** from GitHub
4. **Add Drive mount cell** to notebooks
5. **Train models** with free GPU
6. **Download results** and commit to Git

---

## Want Advanced Remote Connection?

If you still want to try Method 2 (SSH connection), let me know and I'll create a detailed setup script. But for your project, **Method 1 is strongly recommended** for reliability and simplicity.
