# Age & Gender Detection from Webcam

A deep learning system that predicts a person's age and gender from a live webcam photo, built on the UTKFace dataset using transfer learning with MobileNetV2.

## Overview

This project trains a multi-output convolutional neural network to simultaneously predict:
- **Age** (regression)
- **Gender** (binary classification)

from a single face image, then applies the trained model to real-time webcam captures using OpenCV face detection.

## Dataset

- **UTKFace** (via Kaggle: `jangedoo/utkface-new`)
- ~20,000+ labeled face images
- Labels parsed directly from filenames (`age_gender_race_date.jpg`)

## Model Architecture

- **Base model:** MobileNetV2 (pretrained on ImageNet, frozen weights)
- **Head:** Global Average Pooling → two parallel dense branches (512 units each) → separate output heads
  - `age` output: single neuron, linear activation (regression)
  - `gender` output: single neuron, sigmoid activation (binary classification)
- **Loss:** MAE (age) + binary cross-entropy (gender)
- **Optimizer:** Adam

## Features

- End-to-end training pipeline from raw dataset download to trained model
- Data augmentation (rotation, shift, shear, zoom, horizontal flip) for robustness
- Real-time webcam capture directly inside Google Colab
- Haar Cascade face detection before prediction
- Live age and gender prediction on captured photos

## Setup

1. Clone this repository
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Add your Kaggle API token (see [Kaggle API docs](https://www.kaggle.com/docs/api))
4. Run the notebook / script top to bottom in Google Colab (webcam capture requires Colab's JS runtime)

## Usage

1. Run the training pipeline to download data and train the model
2. Run the webcam capture cell — click **Capture** when the video feed appears
3. The script detects your face, crops it, and prints the predicted age and gender

## Tech Stack

Python, TensorFlow/Keras, OpenCV, Pandas, NumPy, Google Colab

## Future Improvements

- Add race/ethnicity as a third prediction head
- Fine-tune deeper MobileNetV2 layers instead of keeping the base fully frozen
- Deploy as a Streamlit app for browser-based use outside Colab
- Rebalance `loss_weights` between age and gender (currently heavily weighted toward gender)

## Author

Fawad Ahmad — [GitHub](https://github.com/FawadAhmad-bilal) | [LinkedIn](https://www.linkedin.com/in/fawad-ahmad-78890236b/)
