# Multimodal Meme Sentiment Analysis

This project implements a sophisticated sentiment classifier for Internet memes, leveraging both textual and visual features through an ensemble machine learning approach.

## Overview

Internet memes are inherently multimodal, combining imagery and text to convey complex emotions. This project classifies memes into five sentiment categories:
- Very Positive
- Positive
- Neutral
- Negative
- Very Negative

## Key Features

- **Multimodal Analysis**: Processes both OCR-extracted text and image data.
- **Ensemble Learning**: 
    - Trains 6 distinct classifiers (3 for text, 3 for images).
    - Text models include Multinomial Naive Bayes, Bagging Classifier, and Extra Trees.
    - Uses **Majority Voting** to determine the final sentiment label.
- **Performance Evaluation**: Provides detailed metrics including Confusion Matrix, Accuracy, Precision, Recall, and F1-score for all models.
- **Web Interface**: A fully functional **Flask** web application allowing users to upload memes and receive real-time sentiment predictions.

## Project Structure

- `PAI_PROJECT.ipynb`: Detailed Jupyter notebook covering data preprocessing, feature extraction, model training, and evaluation.
- `main.py`: Flask application script for deploying the sentiment analysis model to a web interface.
- `labels.csv`: Dataset containing meme metadata and sentiment labels.

## Getting Started

1. Ensure all dependencies (scikit-learn, Flask, skimage, NLTK) are installed.
2. Run the Jupyter notebook to train the models and export parameters.
3. Launch the web application using `python main.py` to interact with the classifier via your browser.
