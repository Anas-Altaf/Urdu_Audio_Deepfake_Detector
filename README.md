# Urdu Deepfake Audio Detection

This project detects deepfake audio in Urdu using machine learning, classifying audio as **Bonafide** (genuine) or **Deepfake** (synthetic). It features an interactive Streamlit app for real-time predictions.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
- [Training](#training)
- [References](#references)

## Overview
The system extracts **MFCC features** from audio and trains four models: **SVM**, **Logistic Regression**, **Perceptron**, and **DNN**. The DNN achieves the best performance (AUC: 0.92). The dataset is sourced from [CSALT/deepfake_detection_dataset_urdu](https://huggingface.co/datasets/CSALT/deepfake_detection_dataset_urdu).

## Setup
1. **Clone the repo**:
   ```bash
   git clone https://github.com/Anas-Altaf/Urdu_Audio_Deepfake_Detector.git
   cd Urdu_Audio_Deepfake_Detector
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Launch the app**:
   ```bash
   streamlit run app.py
   ```
2. **Interact**:
   - Upload an audio file (WAV/MP3) or record live.
   - Select a model and click "Predict Deepfake" for results.

## Training
If weights are missing, the app trains models automatically and saves them. Manually train by running the training script or app functions.

## References
- [Librosa](https://librosa.org/doc/)
- [PyTorch](https://pytorch.org/docs/)
- [Scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://docs.streamlit.io/)
