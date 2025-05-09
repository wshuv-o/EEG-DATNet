# BCI-EEG: EEG Data Processing for Brain-Computer Interface

This repository contains code and resources for processing EEG data for Brain-Computer Interface (BCI) applications using a novel deep learning model. The goal is to enhance the accuracy and efficiency of EEG signal classification for real-time BCI tasks.
 
## Overview

Electroencephalography (EEG) is a non-invasive method to record brain activity. In this project, we process EEG data and classify mental tasks using a novel machine learning model designed specifically for BCI applications. The model is optimized for handling noisy, high-dimensional, and temporally sensitive EEG signals.

## Objectives

- Preprocess raw EEG data for noise reduction and artifact removal.
- Extract meaningful features from multichannel EEG signals.
- Train and evaluate a novel model for classifying EEG signals.
- Enable real-time or near real-time decision-making for BCI systems.

## Features

- EEG signal preprocessing (filtering, normalization, artifact removal)
- Feature extraction using both classical and deep learning techniques
- Novel model architecture for enhanced classification performance
- Evaluation using standard metrics (accuracy, precision, recall, F1-score)
- Visualization of EEG signals and classification results


##Images
![image](https://github.com/user-attachments/assets/24c62527-ee2f-4db6-abc6-b632a0ce21fe)

# eeg

![image](https://github.com/user-attachments/assets/99d8b62b-955c-4d7f-af7a-1d480221d250)
fig: Visualization of artifact removal from frontal EEG channels.
The signals before (blue) and after (yellow) ICA processing are compared
to illustrate the reduction of ocular artifacts.

---

![download (15)](https://github.com/user-attachments/assets/f8d7e24d-2f2c-43c6-8f25-a76630d02f54)

![download (22)](https://github.com/user-attachments/assets/17444056-785f-42e6-9d48-1da5850fbbf8)

![download (20)](https://github.com/user-attachments/assets/2f2d507c-351e-42bf-9efb-f58954acffa2)

![download (21)](https://github.com/user-attachments/assets/ea0f3702-4bc0-4d0d-bc09-dddbd86f2c2d)
![download (14)](https://github.com/user-attachments/assets/8aee9700-7a9b-46ae-b553-77e346bf4e6c)

![image](https://github.com/user-attachments/assets/301f9e3e-0969-40ed-bb6d-2bbf6d03d885)

![image](https://github.com/user-attachments/assets/8257c725-d247-459c-9842-4476ef7bf713)
------------------
![WhatsApp Image 2025-04-08 at 14 01 54_a1f7a901](https://github.com/user-attachments/assets/b24c9895-ce76-466f-b2f4-f3d91b6fb6d2)

## 📁 Project Structure

=======
NeuroTransNet
Overview
NeuroTransNet is a novel hybrid deep learning architecture designed for robust classification of Electroencephalography (EEG) signals in brain-computer interfaces (BCIs). It addresses challenges such as non-stationarity, high dimensionality, and noise in EEG data by integrating multi-scale temporal convolutions, spatial convolutions, dual statistical pooling, a spiking-inspired activation function, and a dual-attention transformer encoder. NeuroTransNet achieves state-of-the-art performance on EEG benchmark datasets, including motor imagery and cognitive state decoding tasks.
Key Features

Multi-scale Temporal Convolution: Captures EEG dynamics across multiple temporal resolutions (60 ms to 260 ms) using parallel 2D convolutions with kernel sizes of 15, 25, 51, and 65.
Spatial Convolution: Models inter-channel dependencies to reflect functional connectivity across brain regions.
Dual Pooling: Combines average and variance pooling to capture central tendencies and signal variability.
Spiking Activation: Introduces biologically inspired non-linearity, mimicking neuronal firing for temporal sparsity and energy efficiency.
Dual-Attention Transformer Encoder: Integrates spatial attention (for temporal segments) and channel attention (for feature channels) to focus on discriminative features.
Preprocessing Pipeline: Leverages MNE-Python for filtering, Independent Component Analysis (ICA), and epoching to enhance signal quality.

Architecture
NeuroTransNet processes EEG signals through the following components:

Input Layer: Accepts EEG data with shape (1, 1000, 22) (time samples, channels).
Multi-scale Temporal Convolution: Four parallel Conv2D layers with 32 filters each, followed by batch normalization.
Spatial Convolution: Depthwise Conv2D to capture inter-channel correlations.
Dual Pooling: Average and variance pooling to extract statistical features.
Dual-Attention Transformer Encoder: Combines spatial and channel attention for refined feature representations.
Classification Head: Concatenates pooling outputs, applies a convolutional encoder, and uses a linear layer for final classification.

For detailed layer specifications, refer to Table 1 in the original paper.
Installation
Prerequisites

Python 3.8+
PyTorch 1.9+
MNE-Python 1.0+
NumPy, SciPy, Pandas
CUDA-enabled GPU (optional, for faster training)

Steps

Clone the repository:git clone https://github.com/neurotransnet/neurotransnet.git
cd neurotransnet


Create a virtual environment:python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate


Install dependencies:pip install -r requirements.txt

Example requirements.txt:torch>=1.9.0
mne>=1.0.0
numpy>=1.19.0
scipy>=1.7.0
pandas>=1.3.0


Download and preprocess EEG datasets (e.g., BCI IV-2a, BCI IV-2b, ToL) using provided scripts in the data/ directory.

Usage

Prepare Data:

Download datasets from their respective sources (see References).
Run the preprocessing script:python data/preprocess.py --dataset bci_iv_2a --output_dir data/processed


This applies filtering, ICA, and epoching as described in the paper.


Train the Model:
python train.py --dataset bci_iv_2a --data_dir data/processed --epochs 100 --batch_size 32


Arguments:
--dataset: Choose from bci_iv_2a, bci_iv_2b, or tol.
--data_dir: Path to preprocessed data.
--epochs: Number of training epochs.
--batch_size: Batch size for training.




Evaluate the Model:
python evaluate.py --model_path models/neurotransnet.pth --dataset bci_iv_2a --data_dir data/processed


Outputs classification accuracy and Cohen's κ-score.



Performance
NeuroTransNet was evaluated on three EEG benchmark datasets:

BCI Competition IV-2a (4-class motor imagery): 83.51% accuracy
BCI Competition IV-2b (2-class motor imagery): 69.93% accuracy
Thinking Out Loud (ToL) (cognitive state decoding): 39.28% accuracy

It outperforms baselines like EEGNet, Incep-EEGNet, and BiLSTM. Ablation studies confirm the importance of dual attention and variance pooling, with performance drops of up to 33.32% when removed.
Directory Structure
neurotransnet/
├── data/
│   ├── preprocess.py       # Script for EEG data preprocessing
│   └── processed/         # Preprocessed dataset directory
├── models/
│   └── neurotransnet.py    # NeuroTransNet model definition
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── requirements.txt       # Dependencies
└── README.md              # This file

References

Tangermann et al., "Review of the BCI Competition IV," Frontiers in Neuroscience, 2012.
Leeb et al., "BCI Competition IV Dataset 2b," 2008.
Nieto et al., "Thinking Out Loud, an Open-Access EEG-Based BCI Dataset," Scientific Data, 2022.
Gramfort et al., "MEG and EEG Data Analysis with MNE-Python," Frontiers in Neuroscience, 2013.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or contributions, contact:

Dr. Debajyoti Karmaker: dr.debajyotikarmaker@gmail.com
Repository: https://github.com/neurotransnet/neurotransnet

