# Project Title
Alzheimer's Disease Classification from MRI Images

## Overview
This project aims to develop a deep learning-based system for the classification of Alzheimer's disease (AD) using magnetic resonance imaging (MRI) images. Alzheimer's disease is a neurodegenerative disorder characterized by cognitive decline, and early and accurate diagnosis is crucial for effective treatment and management.

## Dataset
The dataset used in this project is sourced from Kaggle, a publicly available repository containing MRI images of individuals with varying degrees of cognitive impairment, including those diagnosed with Alzheimer's disease. The dataset is preprocessed to extract relevant features from MRI images and prepare them for model training.
[Click here to visit and download the dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset?select=AugmentedAlzheimerDataset)


## Prerequisites
Before running the project, ensure you have the following:
- Python 3.6 or later installed
- Required Python libraries installed (e.g., TensorFlow, Keras, NumPy, scipy, Pandas, Matplotlib, pathlib)
- Access to the Kaggle dataset of MRI images


## Installation Steps

- Download the software setups and follow the on screen instructions
1. **Install Python 3.7.1**: 
   - Visit the [Python official website](https://www.python.org/downloads/release/python-371/) to download and install python 3.7.1
   
2. **Install Required Libraries**:
   - Open a terminal or command prompt.
   - Run the following commands to install the required Python libraries using pip:
     ```
     pip install tensorflow keras numpy scipy pandas
     ```
      ```
     pip install pillow pathlib2
     ```
      ```
     python -m pip install -U pip
     python -m pip install -U matplotlib
     ```


## Getting Started
The code requires python version 3.6 or later as to synchronize with tensorflow.
- The ds8013_project_cnn_model.py file contains code to train a model based on CNN
- The ds8013_project_inception_model.py file contains contains code to train a model based on Inception v3
- Install the required libraries and packages
- Edit the paths to the dataset in the .py files
- Run the .py files


## Methodology
The methodology involves the following steps:
1. Data preprocessing: MRI images are preprocessed to enhance image quality and extract relevant features.
2. Model development: Deep learning models, such as convolutional neural networks (CNNs), are developed to learn patterns from MRI images and classify individuals into AD and non-AD groups.
3. Model training: The developed models are trained on the preprocessed MRI image data using appropriate loss functions and optimization algorithms.
4. Model evaluation: The trained models are evaluated using performance metrics such as accuracy and loss, to assess their effectiveness in AD classification.

## Usage
1. **Data Preparation**: We already have two sets of images - original and augmented.
2. **Model Training**: Train deep learning models using the preprocessed MRI image data.
3. **Model Evaluation**: Evaluate the trained models using appropriate performance metrics.
4. **Deployment**: Deploy the trained models for AD classification in clinical or research settings.

## Results
The results of the project demonstrate the efficacy of deep learning models in classifying individuals with Alzheimer's disease based on MRI images. The trained models achieve high accuracy and demonstrate promising performance in early detection and classification of AD.

## Future Work
Future work may include:
- Fine-tuning existing models to improve performance.
- Incorporating additional clinical and demographic data for multimodal classification.
- Extending the project to include longitudinal data for disease progression monitoring.


## Editors Used
This project was developed using the following editors:
- Visual Studio Code (VSCode): for local development and code editing.
- Google Colab: for running the project in a cloud-based Jupyter notebook environment, enabling collaborative development and access to GPU resources for model training.


## Authors
- Kehinde Hassan
- Nitya Durbha
