# Breast Cancer Classification using KNN

This project demonstrates how the K-Nearest Neighbors (KNN) algorithm can be applied to a real-world dataset to classify whether a tumor is malignant or benign. The implementation emphasizes key preprocessing techniques such as feature scaling and also reflects on the intuitive working of the KNN algorithm.

---

## Dataset

We’ve used the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository, fetched via Kaggle:

- Dataset: `breast-cancer-wisconsin-data`
- It contains 569 instances of tumor data with 30 numeric features (like radius, texture, perimeter, area, etc.)
- Target Variable: `diagnosis` — either M (malignant) or B (benign)

We cleaned the dataset by removing:
- `id` column (non-informative)
- `Unnamed: 32` (completely null column)

---

## Tools and Libraries Used

The project is built entirely in Python, using the following tools:

| Tool | Purpose |
|------|---------|
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computing |
| `scikit-learn` | ML algorithms, preprocessing, metrics |
| `StandardScaler` | Feature standardization to zero mean and unit variance |
| `KNeighborsClassifier` | Core KNN algorithm |
| `train_test_split` | Splitting the dataset for training and testing |
| `accuracy_score` | Model performance evaluation |

---

## Model: K-Nearest Neighbors (KNN)

### Why KNN?
KNN is a simple, yet powerful algorithm that works on the principle of similarity. It classifies a new point based on how its neighbors (in terms of feature space distance) are labeled.

### Preprocessing:
- Standardization was crucial since KNN relies on distance metrics.
- Applied `StandardScaler()` to make all features have mean 0 and standard deviation 1.

### Model Details:
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
```

---

## Results

After fitting the model with `n_neighbors=3`:

- Achieved an accuracy of approximately 96% on the test set.
- Model performed well in distinguishing between malignant and benign tumors.

You also explored the effect of changing `k` (number of neighbors) and visualized how accuracy varied accordingly — a great way to understand the bias-variance tradeoff.

---

## Concepts Covered

- Feature scaling (Standardization)
- KNN intuition with Euclidean distance
- Train-test split for unbiased evaluation
- Hyperparameter tuning (`k` in KNN)
- Impact of scaling on distance-based models

---

## What You’ll Learn from This Project

- How preprocessing impacts distance-based models like KNN
- Why feature scaling is essential
- How to evaluate model performance with accuracy
- How to use real-world datasets to solve classification problems

---

## Output Achieved

| Metric | Value |
|--------|-------|
| Accuracy | ~96% |
| Model | `KNeighborsClassifier(n_neighbors=3)` |
| Preprocessing | StandardScaler |

---

## Folder Structure

```
KNN_Concepts_Codes.ipynb       <- Jupyter notebook with code and explanation
README.md                      <- Project description and summary
```

---

## Author's Note

This project was built as a hands-on exercise to not just apply KNN, but to deeply understand how preprocessing like StandardScaler can significantly affect model performance. The goal wasn’t just accuracy, but clarity.

If you're someone starting out with Machine Learning, this project offers a solid foundation on how a basic model can become effective with the right preparation.