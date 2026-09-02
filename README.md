# SIH26166 — Lunar Image Registration System
**PS166 | SIH 2026 | ISRO Chandrayaan-2 OHRC Image Registration**

---

## What this project does
This software takes two lunar images (from Chandrayaan-2 OHRC/TMC-2 cameras) and aligns them perfectly on top of each other. It handles different lighting conditions, different camera angles, and different zoom levels.

---

## Setup Instructions (Do this once)

### Step 1 — Clone the repository
```bash
git clone https://github.com/yashhirwani/sih26166-lunar-image-registration.git
cd sih26166-lunar-image-registration
```

### Step 2 — Create a Python virtual environment
```bash
python3 -m venv venv
```

### Step 3 — Activate the virtual environment

Mac/Linux:
```bash
source venv/bin/activate
```

Windows:
```bash
venv\Scripts\activate
```

You should see (venv) at the start of your terminal after this.

### Step 4 — Install all required libraries
```bash
pip install opencv-python opencv-contrib-python numpy scipy matplotlib pillow streamlit kornia torch torchvision
```

This will take 5-10 minutes. Let it finish completely.

### Step 5 — Create required folders
```bash
mkdir -p data/patches data/raw results demo
```

---

## Running the App

Every time you want to run:

Step 1 — Activate venv (if not already active):
```bash
source venv/bin/activate
```

Step 2 — Run the app:
```bash
streamlit run app.py
```

Step 3 — Open browser and go to: http://localhost:8501

---

## How to use the app

1. Upload a Source Image (the image to be aligned)
2. Upload a Reference Image (the target image)
3. Click Run Registration
4. Check the 4 tabs:
   - Match Points — see matched keypoints between images
   - Registration Result — see the aligned output
   - Metrics — see RMSE, inlier count, inlier ratio

---

## Project Structure

```
lunar_registration/
├── app.py                  — Streamlit web UI
├── src/
│   ├── preprocess.py       — CLAHE image cleaning
│   ├── match.py            — SIFT/AKAZE keypoint matching
│   ├── transform.py        — Homography + image warping
│   ├── metrics.py          — RMSE, inlier count, inlier ratio
│   ├── visualize.py        — Match lines, checkerboard views
│   └── pipeline.py         — Connects all modules together
├── data/
│   └── patches/            — Test PNG images go here
├── results/                — Output images saved here
└── demo/                   — Pre-saved demo cases
```

---

## Requirements

- Python 3.9 or higher
- Mac/Linux/Windows
- No GPU required (CPU works fine for demo)

---

## Team
SIH 2026 | PS166 | ISRO
