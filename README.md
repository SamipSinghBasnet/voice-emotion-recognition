Voice Emotion Recognition
Speaker-Independent Speech Emotion Classification with Deployable Inference
Live Demo

A publicly accessible cloud demo is available at:

https://voice-emotion-recogn.streamlit.app/

The demo allows users to record speech directly in the browser and receive emotion predictions with confidence estimates.

Overview

Speech Emotion Recognition (SER) systems often perform well in controlled settings but fail to generalize due to speaker identity leakage and lack of uncertainty handling.
This project addresses these issues by implementing a speaker-independent, self-supervised, and deployable SER pipeline.

The system supports:

Offline training and evaluation with strict experimental controls

Local real-time streaming inference

Cloud deployment with browser-based audio recording

The focus is on generalization, reproducibility, and responsible machine learning design, rather than UI complexity.

Key Features

Three-class emotion recognition: Angry, Neutral, Sad

Strict speaker-independent train/validation/test splits

Self-supervised speech embeddings using wav2vec2

Lightweight and interpretable classifier (Logistic Regression)

Confidence-aware predictions with explicit uncertainty handling

Temporal smoothing across recent predictions

Cloud-deployed browser-based demo

Local true real-time streaming inference

Offline evaluation with Macro-F1 and confusion matrix

Problem Definition

Given a short speech utterance, predict the emotional state of the speaker while ensuring:

No speaker overlap between training and evaluation

Robustness to short and noisy inputs

Explicit handling of low-confidence predictions

Separation of training, evaluation, and deployment concerns

Dataset

CREMA-D (Crowd-Sourced Emotional Multimodal Actors Dataset)
Only the audio modality is used.

Emotion labels are mapped to:

angry

neutral

sad

Raw audio files are not included in this repository due to size and licensing constraints.

Local Dataset Setup

Download CREMA-D and place audio files at:

data/raw/CREMA-D/AudioWAV/

Methodology
Audio Preprocessing

Audio resampled to 16 kHz

Converted to mono

Safe handling of short or empty signals

Feature Extraction

Pretrained facebook/wav2vec2-base model

Mean pooling across time to obtain a fixed-length 768-dimensional embedding

Feature extraction performed offline to reduce deployment overhead

Classification

Logistic Regression classifier

Chosen for interpretability, efficiency, and strong baseline performance

Trained on wav2vec2 embeddings

Speaker-Independent Splitting

Train, validation, and test splits contain disjoint speakers

Prevents identity leakage and inflated performance estimates

Uncertainty and Temporal Smoothing

Softmax confidence estimation

Predictions below a configurable threshold are labeled as uncertain

Majority-vote smoothing across recent predictions improves stability

Evaluation

Evaluation is performed offline on a held-out speaker-independent test set.

Metrics include:

Accuracy

Macro-F1 score

Confusion matrix

Per-class precision, recall, and F1 score

Evaluation results are exported to:

reports/metrics.json


The cloud application loads and displays these metrics without recomputation.

Project Structure
voice-emotion-recognition/
├── src/
│   ├── app/
│   │   ├── app_cloud.py      # Cloud-deployable Streamlit app
│   │   └── app_local.py      # Local real-time streaming app
│   ├── infer/
│   │   └── predict.py        # Inference pipeline
│   ├── features/
│   │   └── extract_embeddings.py
│   ├── train/
│   │   ├── make_splits.py
│   │   ├── train_classifier.py
│   │   └── export_metrics.py
│   └── __init__.py
├── models/
│   └── emotion_clf.joblib
├── reports/
│   └── metrics.json
├── requirements.txt
├── runtime.txt
└── README.md

User Guide (Cloud Demo)
Accessing the Demo

Open the demo link:
https://voice-emotion-recogn.streamlit.app/

Allow microphone access in your browser.

Using the Demo

Click the record button and speak for 1–5 seconds.

The application will:

Process the audio

Predict emotion

Display confidence and class probabilities

If confidence is below the selected threshold, the output is labeled as uncertain.

Interpreting Results

Raw prediction: model output for the current recording

Smoothed prediction: majority vote over recent recordings

Confidence: model certainty for the predicted class

Metrics tab: shows offline test-set evaluation results

Reproducibility (Local)
Install Dependencies
pip install -r requirements.txt

Generate Speaker-Independent Splits
python src/train/make_splits.py

Extract Speech Embeddings
python src/features/extract_embeddings.py

Train the Classifier
python src/train/train_classifier.py

Export Evaluation Metrics
python src/train/export_metrics.py

Local Real-Time Streaming

For true low-latency streaming inference (not supported in managed cloud environments):

pip install sounddevice
streamlit run src/app/app_local.py


This version uses continuous microphone callbacks and sliding-window inference.

Responsible Machine Learning Considerations

Speaker leakage explicitly prevented

Uncertainty handled explicitly via confidence thresholding

Macro-F1 prioritized over accuracy

Clear separation between training, evaluation, and deployment

Metrics version-controlled for transparency and reproducibility

Future Work

Multi-emotion and continuous emotion modeling

Multilingual speech emotion recognition

Temporal modeling over embeddings

Robustness evaluation under noise and domain shift

Attention-based pooling strategies

Author
Samip Singh Basnet
Computer Science
Speech Processing, Machine Learning, Trustworthy AI

Acknowledgements

CREMA-D Dataset Authors

Hugging Face Transformers
Streamlit Community2
