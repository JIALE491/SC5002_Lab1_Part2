# SC5002 Lab 1, Part 2: Linear vs Ridge Regression on California Housing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JIALE491/SC5002_Lab1_Part2/blob/main/SC5002_Lab_Part2.ipynb)

**Author:** LIM JIA LE (U2340936L)

**Course:** SC5002 Lab Assignment, Part 2 (Low-Code ML)

This repository holds my Google Colab notebook for Part 2 of the SC5002 lab. It trains one Linear Regression model and three Ridge Regression models on 1,000 rows of the California Housing dataset. Each model is scored with 5-fold cross validation, and the scores are compared in a bar chart. The goal is to see how the Ridge penalty strength $`\alpha`$ changes the fit, and at what point a stronger penalty starts to hurt.

**Key result:** Linear Regression scored a mean cross-validated $`R^2`$ of **0.6457**. Ridge matched it at $`\alpha = 0.1`$, dropped slightly at $`\alpha = 10`$, and fell to **0.3574** at $`\alpha = 1000`$. The linear model was not overfitting, so Ridge had nothing to fix, and a very strong penalty made the model underfit.

---

## Contents

1. [Repository files](#repository-files)
2. [Dataset](#dataset)
3. [Experiment settings](#experiment-settings)
4. [Method](#method)
5. [Results](#results)
6. [Interpretation](#interpretation)
7. [Notes and limitations](#notes-and-limitations)
8. [Changes made to the template](#changes-made-to-the-template)
9. [How to run](#how-to-run)
10. [Part 1 summary](#part-1-summary)
11. [Use of generative AI](#use-of-generative-ai)

---

## Repository files

| File | What it is |
|---|---|
| `SC5002_Lab_Part2.ipynb` | The Google Colab notebook, with all code and saved outputs |
| `README.md` | This file |

---

## Dataset

The notebook loads the **California Housing** dataset with `sklearn.datasets.fetch_california_housing(as_frame=True)`. The full dataset has 20,640 rows. Each row is one census block group from the 1990 U.S. Census. The notebook randomly samples **1,000 rows** with `random_state=42`, so the same rows are picked every time it runs.

| Column | Meaning | Unit |
|---|---|---|
| `MedInc` | Median income of households in the block group | tens of thousands of US dollars |
| `HouseAge` | Median age of the houses in the block group | years |
| `AveRooms` | Average number of rooms per household | rooms |
| `AveBedrms` | Average number of bedrooms per household | bedrooms |
| `Population` | Number of people living in the block group | people |
| `AveOccup` | Average number of people per household | people |
| `Latitude` | North to south position of the block group | degrees |
| `Longitude` | East to west position of the block group | degrees |
| **`MedHouseVal`** (target) | Median house value in the block group | hundreds of thousands of US dollars |

The first 8 columns are the input features, $`X`$. `MedHouseVal` is the continuous target, $`y`$, so this is a **regression** problem.

---

## Experiment settings

| Setting | Value |
|---|---|
| Rows sampled | 1,000 out of 20,640 |
| Random seed | `random_state=42` for sampling and for the split |
| Train and test split | 80 / 20, so 800 training rows and 200 test rows |
| Feature scaling | `StandardScaler` |
| Models | `LinearRegression`, and `Ridge` with $`\alpha \in \{0.1,\ 10.0,\ 1000.0\}`$ |
| Evaluation | 5-fold cross validation on the 800 training rows |
| Metric | $`R^2`$ (coefficient of determination) |

---

## Method

### Task 2.1: Dataset setup

1. Load the dataset as a pandas DataFrame and sample 1,000 rows.
2. Print the first 5 rows (used for AI Prompt 2.1).
3. Split the columns into the features $`X`$ (8 columns) and the target $`y`$ (`MedHouseVal`).
4. Standardise every feature with `StandardScaler`.
5. Split the scaled data 80 / 20 into training and test sets.

#### Feature scaling

`StandardScaler` turns each value $`x_{ij}`$ (row $`i`$, feature $`j`$) into a z-score:

```math
z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
```

Here $`\mu_j`$ is the mean and $`\sigma_j`$ is the standard deviation of feature $`j`$. After scaling, every feature has a mean of 0 and a standard deviation of 1.

**Why this matters for Ridge.** The Ridge penalty adds up the squared weights, $`\sum_j \beta_j^2`$. Without scaling, a feature with large numbers, like `Population` (often in the thousands), only needs a tiny weight, while a feature with small numbers, like `AveBedrms` (around 1), needs a much larger weight for the same effect. The penalty would then punish the features unevenly, just because of their units. Scaling puts every feature on the same footing, so the penalty treats them all fairly.

For plain Linear Regression, scaling does not change the predictions or the $`R^2`$. It only changes the size of the weights.

### Task 2.2: Model training and cross validation

#### Linear Regression (ordinary least squares)

The model predicts each house value as a weighted sum of the 8 scaled features plus an intercept:

```math
\hat{y}_i = \beta_0 + \sum_{j=1}^{8} \beta_j z_{ij}
```

It picks the weights that make the sum of squared errors as small as possible:

```math
\min_{\beta_0,\ \boldsymbol{\beta}} \ \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2
```

In matrix form, with the data centred so the intercept drops out, the solution is:

```math
\hat{\boldsymbol{\beta}}_{\text{OLS}} = \left( X^{\top} X \right)^{-1} X^{\top} \mathbf{y}
```

#### Ridge Regression (L2 regularisation)

Ridge adds a penalty on the size of the weights. The strength of that penalty is set by $`\alpha \ge 0`$:

```math
\min_{\beta_0,\ \boldsymbol{\beta}} \ \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 \; + \; \alpha \sum_{j=1}^{8} \beta_j^2
```

The intercept $`\beta_0`$ is not penalised. With centred data, the solution is:

```math
\hat{\boldsymbol{\beta}}_{\text{Ridge}} = \left( X^{\top} X + \alpha I \right)^{-1} X^{\top} \mathbf{y}
```

What $`\alpha`$ does:

| $`\alpha`$ | Effect on the model |
|---|---|
| Close to 0 | The penalty barely matters, so Ridge gives almost the same weights as Linear Regression |
| Moderate | The weights shrink a little, which can reduce overfitting |
| Very large | The weights are pushed close to 0, so every prediction moves towards the average house value $`\bar{y}`$, and the model underfits |

#### The R² score

Each model is scored with $`R^2`$, which compares its errors to the errors of always guessing the average:

```math
R^2 = 1 - \frac{\sum_{i} \left( y_i - \hat{y}_i \right)^2}{\sum_{i} \left( y_i - \bar{y} \right)^2}
```

| $`R^2`$ value | Meaning |
|---|---|
| 1 | Perfect predictions |
| 0 | No better than always predicting the average |
| Below 0 | Worse than always predicting the average |

#### 5-fold cross validation

`cross_val_score(model, X_train, y_train, cv=5, scoring='r2')` splits the 800 training rows into 5 folds of 160 rows. The model is trained 5 times. Each time, it learns from 4 folds (640 rows) and is scored on the fold it did not see (160 rows). The reported score is the average of the 5 results:

```math
\overline{R^2}_{\text{CV}} = \frac{1}{5} \sum_{k=1}^{5} R^2_{k}
```

This is more reliable than a single train and test split, because one split can give a lucky or unlucky score depending on which rows end up in the test set. Averaging over 5 different splits smooths that out. For regression, `cv=5` uses `KFold` without reshuffling, which is fine here because the rows are already in random order from the sampling and the split.

### Task 2.3: Visualisation

The notebook draws a bar chart of the mean cross-validated $`R^2`$ for all four models, with the value printed above each bar, so the drop at $`\alpha = 1000`$ is easy to see next to the other three.

---

## Results

| Model | $`\alpha`$ | Mean 5-fold CV $`R^2`$ | Change from Linear Regression |
|---|---|---|---|
| Linear Regression | none | **0.6457** | n/a |
| Ridge Regression | 0.1 | **0.6457** | 0.0000 |
| Ridge Regression | 10.0 | 0.6443 | -0.0014 |
| Ridge Regression | 1000.0 | 0.3574 | -0.2883 |

The scores are shown to 4 decimal places, as the notebook prints them. The chart rounds them to 3.

---

## Interpretation

- **Linear Regression explains about 65% of the variation** in median house value on data it did not train on, since $`R^2 = 0.6457`$.
- **At α = 0.1, the penalty is too small to matter.** Ridge gives the same score as Linear Regression to 4 decimal places.
- **At α = 10, the score drops by only 0.0014.** If the linear model were overfitting, a moderate penalty would usually help. A small drop instead suggests it was not overfitting. This makes sense, since there are 800 training rows and only 8 weights to learn, about 100 rows per weight, so there is little room to memorise noise.
- **At α = 1000, the score falls to 0.3574.** The penalty is now much larger than the error term, so the weights are pushed close to 0. The model starts to ignore real patterns, like the link between income and house value, and its predictions drift towards the average. This is **underfitting**.

### Bias and variance

The expected prediction error can be split into three parts:

```math
\mathbb{E}\left[ \left( y - \hat{f}(x) \right)^2 \right] = \mathrm{Bias}\left[ \hat{f}(x) \right]^2 + \mathrm{Var}\left[ \hat{f}(x) \right] + \sigma^2
```

Increasing $`\alpha`$ adds bias to lower variance. That trade only pays off when variance is the main problem. Here the variance of the linear model was already low, so adding bias only made the error worse, and at $`\alpha = 1000`$ the extra bias dominated.

### Conclusion

On this sample, plain Linear Regression, or Ridge with a small $`\alpha`$, gives the best result. A large $`\alpha`$ should not be picked by default. It should be tuned, and cross validation is a good way to do that.

---

## Notes and limitations

- **The test set is never scored.** The 200 test rows are held out, but the template does not evaluate any model on them. Every score above is a cross-validation score on the 800 training rows. Fitting the chosen model on `X_train` and calling `.score(X_test, y_test)` would give a final test score.
- **The scaler sees all 1,000 rows.** As in the provided template, `StandardScaler` is fitted on all 1,000 rows before the split, so the scaling uses information from the rows that later act as validation folds. A stricter setup would put the scaler and the model in a `Pipeline`, so the scaler is fitted inside each fold.
- **Only 1,000 of the 20,640 rows are used.** The scores may be different on the full dataset.
- **Only three values of α are tried.** A wider search, for example with `RidgeCV` over a range of values, would show more precisely where the best $`\alpha`$ lies.

---

## Changes made to the template

The template code, as given in the assignment PDF, did not run in Colab without two fixes:

| Problem | Error it caused | Fix |
|---|---|---|
| The Ridge `print(f"...")` line was split across two lines | Syntax error, because the f-string was left unclosed | Joined it back into one line |
| There was an extra space before `plt.show()` | Indentation error (unexpected indent) | Removed the space |

The logic, parameters and random seeds are otherwise unchanged from the template.

---

## How to run

### Option 1: Google Colab (recommended)

1. Click the **Open in Colab** badge at the top of this page, or in Colab go to **File > Open notebook > GitHub** and paste this repository's URL.
2. Go to **Runtime > Run all**.
3. The dataset downloads automatically on the first run, so an internet connection is needed. `numpy`, `pandas`, `matplotlib` and `scikit-learn` are already installed in Colab.

### Option 2: Run locally

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook SC5002_Lab_Part2.ipynb
```

### Expected output

```text
=== TASK 2.1: Loading California Housing Dataset ===
First 5 rows of dataset (Copy these for AI Prompt 2.1):
...
=== TASK 2.2: Training Linear Regression ===
Linear Regression Mean 5-Fold R2 Score: 0.6457
--- Training Ridge Regression with Varying Alpha Values ---
Ridge Regression (alpha=   0.1) Mean 5-Fold R2 Score: 0.6457
Ridge Regression (alpha=  10.0) Mean 5-Fold R2 Score: 0.6443
Ridge Regression (alpha=1000.0) Mean 5-Fold R2 Score: 0.3574
=== TASK 2.3: Generating Model Comparison Chart ===
```

Followed by the bar chart.

---

## Part 1 summary

Part 1 has no code. The full working is in the PDF report. This section gives the key results and the formulas behind them.

### Task 1.1: BFS vs A* search

The graph is directed, from start $`S`$ to goal $`G`$.

**Edges and costs**

| Edge | S → A | S → B | S → F | A → C | A → D | B → D | B → E | C → G | D → G | F → G |
|---|---|---|---|---|---|---|---|---|---|---|
| Cost | 2 | 1 | 3 | 2 | 3 | 2 | 4 | 4 | 4 | 6 |

**Heuristic values**

| Node | S | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|---|
| $`h(n)`$ | 6 | 4 | 5 | 2 | 2 | 8 | 4 | 0 |

**Breadth First Search** ignores the costs and explores level by level. It expanded S, A, B and F, then reached G through F.

- Path found: S → F → G, with a cost of $`3 + 6 = 9`$. This is the path with the fewest steps, but not the cheapest one.

**A\* Search** always expands the node with the lowest

```math
f(n) = g(n) + h(n)
```

where $`g(n)`$ is the cost from S to $`n`$ so far, and $`h(n)`$ is the estimated cost from $`n`$ to G. Ties were broken in alphabetical order.

| Step | Node expanded | $`g(n)`$ | $`h(n)`$ | $`f(n)`$ |
|---|---|---|---|---|
| 1 | S | 0 | 6 | 6 |
| 2 | A | 2 | 4 | 6 |
| 3 | B | 1 | 5 | 6 |
| 4 | D | 3 | 2 | 5 |
| 5 | C | 4 | 2 | 6 |
| 6 | F | 3 | 4 | 7 |
| 7 | G | 7 | 0 | 7 |

D was first reached through A with $`g = 5`$, then updated to $`g = 3`$ once the cheaper route through B was found.

- Path found: S → B → D → G, with a cost of $`1 + 2 + 4 = 7`$, which is the cheapest path.

**Admissibility.** A heuristic is admissible if it never overestimates the true cheapest cost $`h^{*}(n)`$ to the goal:

```math
h(n) \le h^{*}(n) \quad \text{for every node } n
```

For node B, the cheapest real path to G is B → D → G, so $`h^{*}(B) = 2 + 4 = 6`$. Since $`h(B) = 5 \le 6`$, the estimate for B is admissible.

### Task 1.2: 8-puzzle with A*

- $`g(n)`$ is the number of moves made so far.
- $`h(n)`$ is the number of misplaced tiles, not counting the blank:

```math
h(n) = \sum_{t=1}^{8} \mathbf{1}\left[ \text{tile } t \text{ is not in its goal position} \right]
```

This heuristic is admissible, because every misplaced tile needs at least one move to reach its goal position.

```text
Start        Goal
2 8 3        1 2 3
1 6 4        8 _ 4
7 _ 5        7 6 5
```

At each step, the board with the lowest $`f(n)`$ was expanded next. When two boards had the same $`f(n)`$, the one found first was expanded first. The search reached the goal in 5 moves of the blank: **up, up, left, down, right**.

| Move | Blank moves | $`g(n)`$ | $`h(n)`$ | $`f(n)`$ |
|---|---|---|---|---|
| 0 | start | 0 | 4 | 4 |
| 1 | up | 1 | 3 | 4 |
| 2 | up | 2 | 3 | 5 |
| 3 | left | 3 | 2 | 5 |
| 4 | down | 4 | 1 | 5 |
| 5 | right | 5 | 0 | 5 |

---

## Use of generative AI

Each task in the assignment includes a "Learn with AI" prompt. This is what the AI was asked in each one. The exact prompts and my own summaries are in the PDF report.

| Task | What the AI was asked |
|---|---|
| 1.1 | Explain admissible heuristics using a GPS navigation analogy |
| 1.2 | Check my 8-puzzle working without solving it for me |
| 2.1 | From the first 5 rows, identify the target, the input features, and why scaling helps |
| 2.2 | Explain the Linear vs Ridge results and why cross validation is more reliable, for a non-technical manager |
| 2.3 | Explain why a very high $`\alpha`$ lowers the $`R^2`$ score |
