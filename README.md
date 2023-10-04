# Person Re-Identification Using CCTV Footage

This repository contains the code and documentation for a person re-identification project using publicly available CCTV footage. The objective of this project is to develop a model that can identify and track individuals across multiple camera views.

## Table of Contents

- [Data Collection and Preprocessing](#data-collection-and-preprocessing)
- [Person Detection and Tracking](#person-detection-and-tracking)
- [Feature Extraction](#feature-extraction)
- [Person Re-Identification Model](#person-re-identification-model)
- [Visualization and Demonstration](#visualization-and-demonstration)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Data Collection and Preprocessing

### Data Collection

- We collected a dataset of publicly available CCTV footage from [source_name]. The dataset includes multiple camera views capturing people walking.

### Data Preprocessing

- The video data was converted into individual frames using OpenCV.
- Bounding boxes were annotated on the frames (if available).
- The dataset was split into training, validation, and testing sets.
- Frames were resized and normalized.

## Person Detection and Tracking

### Person Detection

- A pre-trained YOLOv4 model was used for person detection.

### Person Tracking

- We implemented object tracking using the SORT (Simple Online and Realtime Tracking) algorithm.

## Feature Extraction

- Features were extracted from detected and tracked individuals using a CNN-based method.

## Person Re-Identification Model

### Model Design

- We designed a person re-identification model using PyTorch.

### Training and Evaluation

- The model was trained on the dataset using the extracted features.
- Evaluation metrics include Rank-1 accuracy and mean average precision (mAP).

## Visualization and Demonstration

- Visualizations were created to showcase the effectiveness of the person re-identification model.
- Demonstrations show how the model accurately re-identifies individuals across different camera views.

## Project Structure

person-reidentification/
│
├── data/
│ ├── train_frames/
│ ├── val_frames/
│ └── test_frames/
│
├── src/
│ ├── data_preprocessing.py
│ ├── person_detection.py
│ ├── person_tracking.py
│ ├── feature_extraction.py
│ ├── reidentification_model.py
│ └── visualization.py
│
├── README.md
├── requirements.txt
└── main.py
