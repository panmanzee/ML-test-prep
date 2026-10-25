"""
==============================================================================
CLASSIFICATION MASTERCLASS & EXAM TEMPLATE (PART 2)
==============================================================================
In your exam:
  Part 2 is CLASSIFICATION WITH LIBRARY (scikit-learn) - 13 points
    2.1 (8 pts): Train 4+ classifiers
    2.2 (2 pts): Accuracy + F1 score comparison table (ranked)
    2.3 (3 pts): Convert regression prediction into a class

This file provides:
  - 3 EASY & STANDARD MODELS:
      1. Logistic Regression (Linear + Sigmoid probability)
      2. K-Nearest Neighbors / KNN (Distance-based majority vote)
      3. Decision Tree (If-Else splitting rules)
  - 1 BEST MODEL (Winner in accuracy & performance):
      4. Random Forest / Tuned Gradient Boosting (Ensemble of trees)
  - EVALUATION:
      Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Ranking Table!

Run this file directly: python testing2.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# STEP 0: LOAD & PREPARE DATA FOR CLASSIFICATION
# ==============================================================================
# In classification, the target y must be CATEGORIES or CLASSES (e.g. 0 or 1),
# NOT continuous numbers like 5.62.

csv_path = os.path.join("data", "winequality-red.csv")

if os.path.exists(csv_path):
    print(f"[*] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path, sep=';')
    
    # --------------------------------------------------------------------------
    # CREATING THE CLASS LABEL (Exam Rule):
    # In the wine exam:
    #   Good wine (quality >= 6) -> Class 1
    #   Bad wine  (quality < 6)  -> Class 0
    # --------------------------------------------------------------------------
    THRESHOLD = 6
    y_cls = (df["quality"].values >= THRESHOLD).astype(int)
    
    # Features X (drop target column)
    X_df = df.drop(columns=["quality"])
    X = X_df.values.astype(float)
    feature_names = list(X_df.columns)
    
    print(f"[*] Dataset shape: {X.shape}")
    print(f"[*] Class breakdown: {np.bincount(y_cls)} (Class 0: Bad, Class 1: Good)")
else:
    print("[!] CSV not found in data/, generating sample exam data...")
    from sklearn.datasets import make_classification
    X, y_cls = make_classification(
        n_samples=1000, n_features=11, n_informative=8,
        n_classes=2, random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(11)]

# ------------------------------------------------------------------------------
# STEP 0.1: TRAIN / TEST SPLIT (80% train, 20% test)
# CRITICAL EXAM RULE:
#   Use random_state=42 so that your test rows are IDENTICAL to Part 1!
# ------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cls, test_size=0.2, random_state=42
)

print(f"[*] Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")


# ==============================================================================
# WHY WE USE make_pipeline(StandardScaler(), Model)
# ==============================================================================
# What is a Pipeline?
#   It bundles feature scaling (z = (x - mean)/std) together with the model!
# Why is this essential for exams?
#   1. Zero Data Leakage: It automatically fits the scaler on TRAIN ONLY,
#      never touching test data.
#   2. Super clean syntax: You just do pipeline.fit() and pipeline.predict()
#      in ONE line!
# ==============================================================================
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Import our 3 easy models + 1 best model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


# ==============================================================================
# MODEL 1 (EASY 1): LOGISTIC REGRESSION
# ==============================================================================
# WHAT IS IT?
#   Despite the name 'regression', it is a CLASSIFIER!
#   It draws a straight boundary line between classes.
#
# HOW IT WORKS:
#   1. Computes z = theta0 + theta1*x1 + ... + thetaD*xD
#   2. Passes z through the SIGMOID function: sigma(z) = 1 / (1 + e^(-z))
#   3. This turns z into a probability between 0.0 and 1.0!
#      If probability >= 0.5 -> predict Class 1, else Class 0.
#
# EXAM TIP:
#   Always add max_iter=2000 and StandardScaler() to avoid "ConvergenceWarning".
# ==============================================================================
model_logistic = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, random_state=42)
)


# ==============================================================================
# MODEL 2 (EASY 2): K-NEAREST NEIGHBORS (KNN)
# ==============================================================================
# WHAT IS IT?
#   The simplest, most intuitive classifier in existence.
#   "Show me your friends, and I will tell you who you are."
#
# HOW IT WORKS:
#   1. When given a test sample, it calculates the Euclidean distance to
#      every single sample in the training set.
#   2. Finds the 'k' closest points (e.g. k=15).
#   3. Takes a majority vote: if 10 neighbors are Class 1 and 5 are Class 0,
#      it predicts Class 1!
#
# EXAM TIP:
#   KNN REQUIRES StandardScaler()! If one feature is measured in thousands
#   and another in decimals, the large feature will dominate the distance.
# ==============================================================================
model_knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=15, weights="distance")
)


# ==============================================================================
# MODEL 3 (EASY 3): DECISION TREE
# ==============================================================================
# WHAT IS IT?
#   A flowchart of IF / ELSE questions.
#   Example: "Is alcohol > 10.5%? If yes, is volatile acidity < 0.4? ..."
#
# HOW IT WORKS:
#   At each node, it looks through all features and picks the split that best
#   separates Class 0 from Class 1 (using Gini impurity or Entropy).
#
# ADVANTAGES:
#   - Needs NO scaling at all! It handles raw numbers easily.
#   - Highly interpretable.
#
# EXAM TIP:
#   Trees overfit very easily if allowed to grow infinitely!
#   Always set max_depth (e.g. max_depth=5) to stop it from memorizing noise.
# ==============================================================================
model_tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)


# ==============================================================================
# MODEL 4 (THE BEST MODEL): RANDOM FOREST (Ensemble of Trees)
# ==============================================================================
# WHY IS THIS THE STRONGEST MODEL?
#   A single decision tree makes mistakes and overfits.
#   A RANDOM FOREST builds 300 different decision trees, each trained on
#   a random subset of rows and features!
#   When predicting, all 300 trees vote: majority vote wins!
#
# WHY IT WINS THE TEST:
#   - Handles non-linear curves effortlessly.
#   - Does not overfit easily because individual tree errors cancel out.
#   - Almost always gets the HIGHEST F1 score and Accuracy on tabular data.
# ==============================================================================
model_rf = RandomForestClassifier(
    n_estimators=300,      # Number of trees in the forest
    max_depth=12,          # Max depth of each tree
    random_state=42,
    n_jobs=-1              # Use all CPU cores for fast training
)


# ==============================================================================
# STEP 2.1: TRAIN ALL 4 CLASSIFIERS
# ==============================================================================
print("\n" + "="*60)
print("PART 2.1: TRAINING THE 4 CLASSIFIERS")
print("="*60)

classifiers = {
    "1. Logistic Regression": model_logistic,
    "2. K-Nearest Neighbors": model_knn,
    "3. Decision Tree (depth=5)": model_tree,
    "4. Best: Random Forest (300 trees)": model_rf,
}

# Train each model on training data
for name, model in classifiers.items():
    model.fit(X_train, y_train)
    print(f"[+] Trained: {name}")


# ==============================================================================
# STEP 2.2: PREDICT AND COMPARE ACCURACY & F1 (The 2-Point Ranking Table!)
# ==============================================================================
# In your exam, you MUST print a ranked comparison table comparing:
#   1. Accuracy
#   2. F1-Score
#
# WHAT IS THE DIFFERENCE?
#   - Accuracy = (Total Correct) / (All Samples).
#     WARNING: Accuracy is misleading when data is imbalanced! If 90% of wines
#     are bad, a dumb model that always guesses 'bad' gets 90% accuracy!
#   - F1-Score = Harmonic mean of Precision and Recall.
#     Only high if the model is good at catching positives AND makes few false alarms.
#     The exam ranks models by F1-Score!
# ==============================================================================
print("\n" + "="*60)
print("PART 2.2: EVALUATION & RANKED LEADERBOARD")
print("="*60)

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Check if binary or multiclass
is_binary = len(np.unique(y_cls)) == 2
f1_average = "binary" if is_binary else "macro"

# Generate predictions and metrics for each model
results = []

for name, model in classifiers.items():
    # Predict on unseen test data
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=f1_average)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1 Score": f1
    })

# Convert to DataFrame and sort by F1 Score (highest to lowest)
df_results = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
df_results.insert(0, "Rank", range(1, len(df_results) + 1))

print(df_results.to_string(index=False))

best_model_name = df_results.iloc[0]["Model"]
best_model = classifiers[best_model_name]

print(f"\n[WINNER] The top performing model is: {best_model_name}")


# ------------------------------------------------------------------------------
# CONFUSION MATRIX & DETAILED REPORT FOR THE WINNER
# ------------------------------------------------------------------------------
print("\n--- Detailed Classification Report for Winner ---")
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best, target_names=["Bad Wine (0)", "Good Wine (1)"]))

cm = confusion_matrix(y_test, y_pred_best)
print("Confusion Matrix:")
print("                 Predicted Bad  Predicted Good")
print(f"Actual Bad (0) :      {cm[0, 0]:<14} {cm[0, 1]}")
print(f"Actual Good (1):      {cm[1, 0]:<14} {cm[1, 1]}")


# ==============================================================================
# STEP 2.3: REGRESSION PREDICTIONS -> CLASSIFICATION (Worth 3 marks!)
# ==============================================================================
# The exam asks:
# "Use the regression predictions from Part 1 to predict the class."
#
# How to do it:
#   If your Part 1 model predicted wine quality as continuous numbers (e.g. 6.2, 4.8),
#   you simply apply a threshold:
#   predicted_class = 1 if predicted_quality >= 5.5 else 0
# ==============================================================================
print("\n" + "="*60)
print("PART 2.3: REGRESSION TO CLASSIFICATION (THRESHOLD)")
print("="*60)

# Simulate continuous regression predictions (e.g., from Part 1)
sample_continuous_preds = np.array([4.8, 6.2, 5.7, 7.1, 5.1])

# Convert to class using threshold 5.5:
threshold_converted_classes = (sample_continuous_preds >= 5.5).astype(int)

for val, cls in zip(sample_continuous_preds, threshold_converted_classes):
    label_str = "Good (1)" if cls == 1 else "Bad (0)"
    print(f"Regression prediction: {val:4.1f}  -->  Predicted Class: {label_str}")


# ==============================================================================
# HOW TO CLASSIFY THE TEACHER'S GIVEN SAMPLE ROW
# ==============================================================================
print("\n" + "="*60)
print("CLASSIFYING A GIVEN SAMPLE (Teacher's Question)")
print("="*60)

# Example: Take a single test row: shape (1, num_features)
sample_to_test = X_test[0:1]

for name, model in classifiers.items():
    pred_cls = model.predict(sample_to_test)[0]
    pred_label = "Good Wine (1)" if pred_cls == 1 else "Bad Wine (0)"
    print(f"{name:35s} -> Predicts: {pred_label}")

print(f"Actual True Class for this sample    -> {y_test[0]}")
print("="*60)
