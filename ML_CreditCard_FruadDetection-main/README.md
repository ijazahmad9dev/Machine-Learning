# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using binary classification.

## Key Features

- **Dataset**: Utilizes a Kaggle dataset containing transaction data with 31 variables.
- **Preprocessing**: Includes data scaling using `StandardScaler`, normalization, and class weight computation to handle class imbalance.
- **Inflated Data Training**: The notebook demonstrates a technique of inflating the dataset (repeating values) to simulate a larger workload.
- **Performance Optimization**: Explores both standard `scikit-learn` and `Snap ML` (IBM's library for accelerated machine learning) to compare training times, especially for larger datasets.

## Implementation

The main logic is contained within `Fruad_Detection.ipynb`, covering data ingestion, exploratory analysis, preprocessing, and model training.