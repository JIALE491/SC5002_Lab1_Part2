# SC5002 Lab Assignment, Part 2: Linear vs Ridge Regression

Author: LIM JIA LE U2340936L

This repository contains the Google Colab notebook for Part 2 of the SC5002 Artificial Intelligence Fundamentals & Applications lab assignment. The notebook trains Linear Regression and Ridge Regression models on the California Housing dataset and compares their 5-fold cross-validation R² scores.

## Files

| File | Description |
|---|---|
| `SC5002_Lab_Part2.ipynb` | Colab notebook with the code and its saved outputs (first 5 rows, R² scores, bar chart) |
| `README.md` | This file |

## How to run

1. Open `SC5002_Lab_Part2.ipynb` in Google Colab (File → Open notebook → GitHub tab, or upload the file).
2. Click **Runtime → Run all**.
3. The whole assignment runs in a single cell and takes a few seconds. An internet connection is needed the first time, as scikit-learn downloads the dataset.

All libraries used (`numpy`, `pandas`, `matplotlib`, `scikit-learn`) come preinstalled in Colab.

## Dataset

The notebook uses the California Housing dataset from `sklearn.datasets.fetch_california_housing`.

- A random sample of 1,000 rows is taken with `random_state=42`, so the results are the same every time the notebook is run.
- **Target variable:** `MedHouseVal`, the median house value of a block group, in units of $100,000.
- **Input features (8):** `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`.

## Method

1. **Task 2.1, setup:** load the data, separate the target from the features, scale the features with `StandardScaler`, and split the data 80/20 into training and test sets (`random_state=42`).
2. **Task 2.2, training:** train Linear Regression and Ridge Regression with alpha = 0.1, 10.0 and 1000.0. Each model is scored with 5-fold cross-validation on the training set using R².
3. **Task 2.3, visualisation:** plot the mean cross-validation R² of all four models in a bar chart.

## Results

| Model | Mean 5-fold CV R² |
|---|---|
| Linear Regression | 0.6457 |
| Ridge (alpha = 0.1) | 0.6457 |
| Ridge (alpha = 10.0) | 0.6443 |
| Ridge (alpha = 1000.0) | 0.3574 |

- Ridge with alpha = 0.1 matches Linear Regression, as the penalty is too small to change the model.
- Ridge with alpha = 10 scores slightly lower, which suggests Linear Regression was not overfitting this data much.
- Ridge with alpha = 1000 drops sharply. The penalty shrinks the coefficients towards zero and the model underfits.

