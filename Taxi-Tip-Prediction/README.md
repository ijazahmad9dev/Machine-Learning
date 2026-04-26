# Taxi Tip Prediction

A regression task using a 2019 NYC Yellow Taxi dataset to predict the tip amount for a trip.

## Overview

This project demonstrates a full machine learning pipeline from data ingestion to model evaluation:

1.  **Data Cleaning**: Handling missing values and outliers in the taxi trip data.
2.  **Feature Engineering**: Extracting useful features from timestamps, such as hour of the day and day of the week.
3.  **Modeling**:
    *   Implementing a `DecisionTreeRegressor` using `scikit-learn`.
    *   Utilizing `Snap ML` (IBM) for accelerated training.
4.  **Evaluation**: Comparing the performance and training speed of different implementations.

## Dataset

- **Source**: NYC Taxi and Limousine Commission (TLC) Yellow Taxi Trip records.
- **Goal**: Predict the `tip_amount` based on features like trip distance, fare amount, and pickup/drop-off times.
