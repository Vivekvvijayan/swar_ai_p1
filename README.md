---
title: Carnatic Raga AI
emoji: 🎵
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Carnatic Raga Recognition & Classification AI Server

Backend deep audio inference service for the Carnatic Raga Recognition Android Application.

## System Overview
- **Model Architecture:** ResNet Deep Audio Classifier (10 Residual Blocks)
- **Classification Scope:** 150 Carnatic Ragas
- **Target Sampling Rate:** 8,000 Hz Stereo PCM
- **Input Clip Window:** 30 seconds
- **Inference API:** Gradio / FastAPI REST Protocol on port `7860`
- **Client Integration:** Compatible with Android client (`GradioServerApi`)

## Installation
```bash
pip install -r requirements.txt
```

## Starting the Server
```bash
python app.py --host 0.0.0.0 --port 7860
```
