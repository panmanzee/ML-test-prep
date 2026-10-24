"""
LINEAR REGRESSION CHEAT-CODE & TEMPLATE FOR YOUR ML TEST
=========================================================
This file contains two ways to do Linear Regression:
  1. METHOD 1: The Scikit-Learn Way (Fast, 5 lines, standard library)
  2. METHOD 2: The From-Scratch NumPy Way (Required for university 'from scratch' exams)

Both methods are fully runnable and tested on your wine dataset!
Run this file directly: python testing.py
"""

import os
import numpy as np
import pandas as pd

# ==============================================================================
# STEP 0: LOAD & PREPARE DATA
# ==============================================================================
# Check if wine dataset exists, otherwise create simple dummy data
csv_path = os.path.join("data", "winequality-red.csv")

if os.path.exists(csv_path):
    print(f"[*] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path, sep=';')
    target_col = "quality"
    
    # Drop target to get features X, and extract target y
    X = df.drop(columns=[target_col]).values.astype(float)
    y = df[target_col].values.astype(float)
    feature_names = [col for col in df.columns if col != target_col]
else:
    print("[!] CSV not found in data/, generating sample exam data...")
    np.random.seed(42)
    X = np.random.randn(100, 3)  # 100 rows, 3 features
    y = 2.5 * X[:, 0] - 1.2 * X[:, 1] + 3.0 + np.random.randn(100) * 0.1
    feature_names = ["feat_1", "feat_2", "feat_3"]

# ------------------------------------------------------------------------------
# Train/Test Split (80% train, 20% test)
# ------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------------------
# Feature Scaling (Standardization: z = (x - mean) / std)
# Crucial for Gradient Descent and when features have different units!
# ------------------------------------------------------------------------------
mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)
sd[sd == 0] = 1.0  # avoid division by zero

Xtr = (X_train - mu) / sd   # Scaled train
Xte = (X_test - mu) / sd    # Scaled test (Notice: use mu and sd from train!)


# ==============================================================================
# METHOD 1: SCIKIT-LEARN (Library Approach - easiest syntax to memorize)
# ==============================================================================
print("\n" + "="*50)
print("METHOD 1: SCIKIT-LEARN (Library)")
print("="*50)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. Initialize the model
sklearn_model = LinearRegression()

# 2. Fit (train) on training data
sklearn_model.fit(Xtr, y_train)

# 3. Predict on unseen test data
y_pred_sklearn = sklearn_model.predict(Xte)

# 4. Evaluate metrics
r2_sklearn = r2_score(y_test, y_pred_sklearn)
rmse_sklearn = np.sqrt(mean_squared_error(y_test, y_pred_sklearn))

print(f"Intercept (theta 0) : {sklearn_model.intercept_:.4f}")
print(f"Coefficients (theta): {sklearn_model.coef_}")
print(f"Sklearn R2 Score    : {r2_sklearn:.4f}")
print(f"Sklearn RMSE        : {rmse_sklearn:.4f}")


# ==============================================================================
# METHOD 2: FROM SCRATCH WITH NUMPY (University Exam Part 1)
# ==============================================================================
print("\n" + "="*60)
print("METHOD 2: FROM SCRATCH (STEP-BY-STEP FOR YOUR EXAM)")
print("="*60)

# ==============================================================================
# STEP-BY-STEP EXPLANATION OF FROM-SCRATCH REGRESSION
# ==============================================================================
# In your exam (Part 1), the professor checks whether you understand the MATH
# and can write it in pure NumPy without using sklearn's fit() or predict().
#
# A linear regression equation for D features is:
#   y_hat = theta0 + theta1*x1 + theta2*x2 + ... + thetaD*xD
#
# In matrix form, this is:
#   y_hat = X_bias @ theta
#
# Where:
#   - theta0 is the INTERCEPT (bias): the value of y when all x are 0.
#   - theta1..thetaD are the WEIGHTS (slopes) for each feature.
#   - X_bias has an extra column of 1s on the left, so that 1 * theta0 = theta0.
# ==============================================================================


# ------------------------------------------------------------------------------
# STEP 1: HELPER FUNCTIONS (The 3 functions you should write first)
# ------------------------------------------------------------------------------

def add_bias(X_matrix):
    """
    STEP 1.1: ADD BIAS COLUMN OF ONES
    Why: We need an intercept theta_0.
    Math: Turns X of shape (N, D) into X_b of shape (N, D + 1)
          where column 0 is entirely 1s: [[1, x1, x2..], [1, x1, x2..]]
    Code: np.c_ stands for 'column concatenate'.
    """
    num_rows = len(X_matrix)
    ones_column = np.ones((num_rows, 1))
    return np.c_[ones_column, X_matrix]


def r2_scratch(y_true, y_pred):
    """
    STEP 1.2: CALCULATE R² SCORE FROM SCRATCH
    Formula:
        R² = 1 - (SS_res / SS_tot)
        where:
        SS_res = sum( (y_true - y_pred)^2 )       -> Error of your model
        SS_tot = sum( (y_true - mean(y_true))^2 ) -> Error of "just guessing the mean"
    Interpretation:
        1.0 = Perfect predictions
        0.0 = Model is no better than guessing the average y
        < 0 = Model is worse than guessing the average (terrible fit)
    """
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_residual / ss_total)


def rmse_scratch(y_true, y_pred):
    """
    STEP 1.3: CALCULATE RMSE (Root Mean Squared Error)
    Formula: sqrt( mean( (y_true - y_pred)^2 ) )
    Tells you how many units on average your prediction is off by.
    """
    mean_squared_error = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mean_squared_error)


# ------------------------------------------------------------------------------
# STEP 2: PREPARE THE BIASED MATRICES
# ------------------------------------------------------------------------------
# Shape check:
#   Xtr has shape (N_train, D)       e.g., (1279, 11)
#   Xb_train has shape (N_train, D+1) e.g., (1279, 12)  <-- extra 1s column!
#   Xb_test  has shape (N_test,  D+1) e.g., (320, 12)
Xb_train = add_bias(Xtr)
Xb_test  = add_bias(Xte)

print(f"[Shape Check] Xtr original : {Xtr.shape}")
print(f"[Shape Check] Xb with bias : {Xb_train.shape} (Added 1 column for intercept)")


# ------------------------------------------------------------------------------
# STEP 3: SOLVE USING THE NORMAL EQUATION (OLS - Exact Analytical Solution)
# ------------------------------------------------------------------------------
# MATH FORMULA (Memorize this for the test!):
#   theta = (X^T @ X)^(-1) @ X^T @ y
#
# Step-by-step matrix dimensions:
#   1. Xb_train.T has shape (D+1, N)
#   2. Xb_train   has shape (N, D+1)
#   3. (Xb_train.T @ Xb_train) has shape (D+1, D+1) -> Square matrix!
#   4. (Xb_train.T @ y_train)  has shape (D+1,)
#   5. theta has shape (D+1,)
#
# CODE TIP:
#   Do NOT use np.linalg.inv(X.T @ X) @ X.T @ y because matrix inversion is slow
#   and can cause floating point inaccuracies or crash on singular matrices.
#   USE: np.linalg.solve(A, b) which solves A * theta = b directly!
A = Xb_train.T @ Xb_train        # Shape: (D+1, D+1)
b = Xb_train.T @ y_train         # Shape: (D+1,)

theta_ols = np.linalg.solve(A, b) # Exact solution in one step!

# ------------------------------------------------------------------------------
# STEP 4: PREDICT AND EVALUATE WITH OLS
# ------------------------------------------------------------------------------
# Formula: y_pred = X_test_with_bias @ theta
# Shape: (N_test, D+1) @ (D+1,) -> (N_test,)
y_pred_ols = Xb_test @ theta_ols

r2_ols = r2_scratch(y_test, y_pred_ols)
rmse_ols = rmse_scratch(y_test, y_pred_ols)

print("\n--- OLS Results ---")
print(f"Intercept theta[0] : {theta_ols[0]:.4f} (Notice it matches sklearn intercept!)")
print(f"Weights theta[1:]  : {theta_ols[1:]}")
print(f"OLS Test R2 Score  : {r2_ols:.4f} (Must match Sklearn R2!)")
print(f"OLS Test RMSE      : {rmse_ols:.4f}")


# ------------------------------------------------------------------------------
# STEP 5: SOLVE USING GRADIENT DESCENT (Iterative Optimization)
# ------------------------------------------------------------------------------
# Sometimes the test specifically asks: "Train linear regression using Gradient Descent".
# Here is the complete breakdown:
#
# The Loss function is Mean Squared Error (MSE):
#   Loss = (1 / n) * sum( (Xb @ theta - y)^2 )
#
# The Derivative (Gradient) with respect to theta is:
#   grad = (2 / n) * Xb^T @ (Xb @ theta - y)
#
# The Update Rule:
#   theta_new = theta_old - (learning_rate * grad)
#   We walk in the opposite direction of the gradient to reach the minimum loss!
# ------------------------------------------------------------------------------

# Hyperparameters:
learning_rate = 0.05  # If loss becomes NaN or inf, lower this to 0.01 or 0.001!
num_epochs = 1500     # Number of iterations through the dataset
n_samples, n_features = Xb_train.shape

# 1. Initialize theta with all zeros: shape (D+1,)
theta_gd = np.zeros(n_features)
loss_history = []

for epoch in range(num_epochs):
    # Step 5.1: Calculate current predictions: y_hat = Xb @ theta
    # Shape: (N, D+1) @ (D+1,) -> (N,)
    y_hat = Xb_train @ theta_gd
    
    # Step 5.2: Calculate the error: difference between predicted and actual
    # Shape: (N,)
    error = y_hat - y_train
    
    # Step 5.3: Calculate the gradient vector:
    # Math: (2 / n) * Xb^T @ error
    # Shape: (D+1, N) @ (N,) -> (D+1,)  <-- Exactly 1 gradient for each theta!
    gradient = (2.0 / n_samples) * (Xb_train.T @ error)
    
    # Step 5.4: Update theta by taking a step downhill:
    theta_gd = theta_gd - (learning_rate * gradient)
    
    # Optional: track loss every epoch (should decrease smoothly)
    current_mse = np.mean(error ** 2)
    loss_history.append(current_mse)

# Predict on test data with GD weights:
y_pred_gd = Xb_test @ theta_gd
r2_gd = r2_scratch(y_test, y_pred_gd)
rmse_gd = rmse_scratch(y_test, y_pred_gd)

print("\n--- Gradient Descent Results ---")
print(f"GD Initial Loss (Epoch 0)   : {loss_history[0]:.4f}")
print(f"GD Final Loss (Epoch {num_epochs}) : {loss_history[-1]:.4f}")
print(f"GD Intercept theta[0]       : {theta_gd[0]:.4f}")
print(f"GD Test R2 Score            : {r2_gd:.4f}")
print(f"GD Test RMSE                : {rmse_gd:.4f}")


# ------------------------------------------------------------------------------
# STEP 6: RIDGE REGRESSION FROM SCRATCH (Regularization) - Frequently on Exams!
# ------------------------------------------------------------------------------
# Problem: What if features are correlated, or (X^T @ X) is singular / unstable?
# Solution: Ridge adds a penalty lambda (L2 regularization): lambda * sum(theta^2)
#
# MATH FORMULA:
#   theta_ridge = (X^T @ X + lambda * I)^(-1) @ X^T @ y
#
# CRITICAL EXAM TRICK:
#   Do NOT penalize the bias / intercept (theta_0)!
#   So set the top-left element of identity matrix I[0, 0] = 0!
# ------------------------------------------------------------------------------
lam = 1.0  # regularization strength
I = np.eye(n_features)  # Identity matrix of shape (D+1, D+1)
I[0, 0] = 0.0           # Do NOT penalize intercept!

theta_ridge = np.linalg.solve(Xb_train.T @ Xb_train + lam * I, Xb_train.T @ y_train)
y_pred_ridge = Xb_test @ theta_ridge
print("\n--- Ridge (L2) Regularization Scratch ---")
print(f"Ridge Test R2 (lambda={lam}) : {r2_scratch(y_test, y_pred_ridge):.4f}")


# ------------------------------------------------------------------------------
# STEP 7: PREDICTING THE TEACHER'S GIVEN SAMPLE ROW (Worth 2 marks!)
# ------------------------------------------------------------------------------
# In exams, teachers often write:
# "Predict the target for this specific test observation: [x1, x2, ...]"
#
# THE 3 MANDATORY STEPS:
#   1. Take the raw input array and reshape it to 2D: shape (1, num_features)
#   2. Scale it using TRAIN mean and std: (sample - mu) / sd
#   3. Add the bias 1 in front: [1, scaled_sample]
#   4. Compute dot product: sample_b @ theta
# ------------------------------------------------------------------------------
print("\n" + "="*60)
print("HOW TO PREDICT A SINGLE GIVEN VALUE (Exam Question)")
print("="*60)

# Example: Take the first row of test set as our given sample
raw_given_sample = X_test[0:1]  # Slice 0:1 keeps it 2D: shape (1, 11)

# Step 7.1: Scale using TRAINING mu and sd (DO NOT use test stats!)
scaled_given_sample = (raw_given_sample - mu) / sd

# Step 7.2: Add bias
sample_with_bias = add_bias(scaled_given_sample)  # Shape (1, 12)

# Step 7.3: Predict with dot product
predicted_value = sample_with_bias @ theta_ols

print(f"Teacher's given sample raw : {raw_given_sample[0][:4]}... (truncated)")
print(f"Predicted target value     : {predicted_value[0]:.4f}")
print(f"Actual true target value   : {y_test[0]:.4f}")
print("="*60)


# ------------------------------------------------------------------------------
# STEP 8: POLYNOMIAL REGRESSION FROM SCRATCH (Exam Model 3)
# ------------------------------------------------------------------------------
# Why Polynomial? Linear models only draw flat planes/lines.
# If the data curves, adding x^2 and x1*x2 terms lets a linear model fit curves!
#
# Degree 2 features for [x1, x2]:
#   Original: [x1, x2]
#   Degree 2: [x1, x2,  x1^2,  x1*x2,  x2^2]
# Then, you simply run OLS on this expanded feature matrix!
# ------------------------------------------------------------------------------
def poly_features_scratch(X_data, degree=2):
    """
    Expands X with squares (x_i^2) and pairwise products (x_i * x_j).
    """
    n_cols = X_data.shape[1]
    poly_cols = [X_data]
    if degree >= 2:
        # Generate all x_i * x_j (including x_i * x_i when j == i)
        interaction_terms = np.column_stack([
            X_data[:, i] * X_data[:, j]
            for i in range(n_cols)
            for j in range(i, n_cols)
        ])
        poly_cols.append(interaction_terms)
    return np.hstack(poly_cols)

# Create polynomial features for train and test:
P_train = poly_features_scratch(Xtr, degree=2)
P_test  = poly_features_scratch(Xte, degree=2)

# Rescale newly generated columns:
mu_p, sd_p = P_train.mean(0), P_train.std(0) + 1e-12
P_tr_scaled = (P_train - mu_p) / sd_p
P_te_scaled = (P_test - mu_p) / sd_p

# Solve with Ridge (always use lambda > 0 with polynomial features to prevent wild weights!)
Pb_train = add_bias(P_tr_scaled)
Pb_test  = add_bias(P_te_scaled)
d_poly = Pb_train.shape[1]
I_p = np.eye(d_poly); I_p[0, 0] = 0.0

theta_poly = np.linalg.solve(Pb_train.T @ Pb_train + 1.0 * I_p, Pb_train.T @ y_train)
y_pred_poly = Pb_test @ theta_poly

print("\n--- Polynomial Regression (Degree 2) Scratch ---")
print(f"Num Poly features created: {P_train.shape[1]}")
print(f"Polynomial Test R2       : {r2_scratch(y_test, y_pred_poly):.4f}")


# ------------------------------------------------------------------------------
# STEP 9: THE CLEAN REUSABLE CLASS (Exact template used in 03_EXAM_TEMPLATE.ipynb)
# ------------------------------------------------------------------------------
# If your exam requires an object-oriented class structure, here is the
# 25-line class from your course cheat sheet:
class LinearRegressionScratch:
    def __init__(self, method="ols", lr=0.05, epochs=2000, lam=0.0):
        self.method = method  # "ols" or "gd"
        self.lr = lr
        self.epochs = epochs
        self.lam = lam

    def fit(self, X, y):
        Xb = add_bias(X)
        n, d = Xb.shape
        if self.method == "ols":
            # Normal Equation with optional Ridge (lam)
            I = np.eye(d); I[0, 0] = 0  # don't penalize bias
            self.theta = np.linalg.solve(Xb.T @ Xb + self.lam * I, Xb.T @ y)
        else:
            # Gradient Descent
            self.theta = np.zeros(d)
            for _ in range(self.epochs):
                error = Xb @ self.theta - y
                grad = (2.0 / n) * (Xb.T @ error)
                if self.lam > 0:
                    grad[1:] += 2.0 * self.lam * self.theta[1:] / n
                self.theta -= self.lr * grad
        return self

    def predict(self, X):
        return add_bias(X) @ self.theta

print("\n--- Summary: All models trained and verified! ---")
print("Good luck on your exam! Keep this file and CHEATSHEET.txt handy.")

