# Chest X-ray Pneumonia Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.6%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.4%2B-red)](https://keras.io/)

## Summary
An AI-powered diagnostic tool that uses deep learning to automatically detect pneumonia from chest X-ray images with high accuracy. The project implements a convolutional neural network model based on MobileNet architecture, offering two training approaches: dataset balancing through oversampling and class-weight adjustment. The model achieves precision and recall metrics suitable for clinical decision support, demonstrating the potential of AI to assist healthcare professionals in pneumonia screening.

## Overview
This project focuses on detecting pneumonia in chest X-ray images using deep learning techniques. It provides two approaches for training the model: one balances the dataset with oversampling, and the other uses the original unbalanced dataset and applies class weights during the training phase. The model is based on the MobileNet architecture, which is known for its lightweight design and high performance. The project aims to demonstrate the effectiveness of deep learning in medical image analysis and its potential to assist healthcare professionals in diagnosing pneumonia accurately and efficiently.

## Dataset
To reproduce the experiments conducted in this project, you need to download the Chest X-ray dataset from the following link:

[Mendeley](https://data.mendeley.com/datasets/rscbjbr9sj/2).

After downloading, follow these steps:

1. **Unzip the folder:** Extract the contents of the downloaded file named "ChestXRay2017.zip".
2. **Specify the dataset path:** Copy the correct path to the folder containing the dataset inside the code. You may need to update the file paths in the code accordingly.

Make sure to have the dataset available in the specified directory to run the code successfully.

## Usage
Before running the code, ensure you have downloaded the dataset and placed it in the appropriate directory. You should also choose which approach you want to use for training by commenting out the respective sections in the code.

### Prerequisites
- Python 3.6+
- Jupyter Notebook
- TensorFlow 2.0+
- Keras 2.4+
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Seaborn

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/PeretzNiro/Chest-X-Ray-Images-Classification.git
   cd Chest-X-Ray-Images-Classification
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset and place it in the specified directory.

### Running the Code
Open the Jupyter Notebook file (`Chest_X_Ray_Medical_Images_Classification.ipynb`) and execute the code cells.

## Results
The model demonstrates strong performance in pneumonia detection with the following metrics:
- High accuracy on the test set
- Balanced precision and recall
- Strong ROC curve and AUC score

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgements
The codebase for this project was adapted from [Khan Fashee Monowar's Kaggle notebook](https://www.kaggle.com/code/khanfashee/medical-image-classification-for-beginner). We acknowledge and appreciate their work in providing the foundational code for this project.