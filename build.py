"""Builds all study notebooks. Run: python build.py  (then execute with nbconvert)."""
import nbformat as nbf

def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(s.strip()) if t == "md" else nbf.v4.new_code_cell(s.strip()) for t, s in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return n

def md(s): return ("md", s)
def code(s): return ("code", s)

# ---------------------------------------------------------------- shared code
IMPORTS = """
import os, warnings
os.environ["LOKY_MAX_CPU_COUNT"] = "4"   # silences a harmless joblib warning on Windows
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
np.set_printoptions(suppress=True, precision=4)
"""

SCRATCH_TOOLS = '''
# ===== from-scratch toolbox (numpy only) =====
def r2_scratch(y, p):
    ss_res = ((y - p) ** 2).sum()          # error left over by the model
    ss_tot = ((y - y.mean()) ** 2).sum()   # error of "always guess the mean"
    return 1 - ss_res / ss_tot

def rmse_scratch(y, p):
    return np.sqrt(((y - p) ** 2).mean())

def add_bias(X):
    return np.c_[np.ones(len(X)), X]      # column of 1s -> intercept theta0

class LinearRegressionScratch:
    """y_hat = X @ theta.  method='ols' (normal equation) or 'gd' (gradient descent).
    lam > 0 adds Ridge (L2) penalty -> stops huge weights / overfitting."""
    def __init__(self, method="ols", lr=0.05, epochs=2000, lam=0.0):
        self.method, self.lr, self.epochs, self.lam = method, lr, epochs, lam
    def fit(self, X, y):
        Xb = add_bias(X)
        n, d = Xb.shape
        if self.method == "ols":
            I = np.eye(d); I[0, 0] = 0          # never penalise the intercept
            self.theta = np.linalg.solve(Xb.T @ Xb + self.lam * I, Xb.T @ y)
        else:
            self.theta = np.zeros(d)
            self.loss_history = []
            for _ in range(self.epochs):
                error = Xb @ self.theta - y               # prediction - truth
                grad = (2 / n) * Xb.T @ error              # d(MSE)/d(theta)
                grad[1:] += 2 * self.lam * self.theta[1:] / n
                self.theta -= self.lr * grad               # step downhill
                self.loss_history.append((error ** 2).mean())
        return self
    def predict(self, X):
        return add_bias(X) @ self.theta

def poly_features(X, degree=2):
    """degree 2: x_i, all x_i*x_j (incl. squares). degree 3: also adds x_i^3."""
    parts = [X]
    n = X.shape[1]
    if degree >= 2:
        parts.append(np.column_stack([X[:, i] * X[:, j] for i in range(n) for j in range(i, n)]))
    if degree >= 3:
        parts.append(X ** 3)
    return np.hstack(parts)

class PolyRegressionScratch:
    def __init__(self, degree=2, lam=1.0):
        self.degree, self.lam = degree, lam
    def fit(self, X, y):
        P = poly_features(X, self.degree)
        self.mu, self.sd = P.mean(0), P.std(0) + 1e-12   # re-scale new columns
        self.lin = LinearRegressionScratch("ols", lam=self.lam).fit((P - self.mu) / self.sd, y)
        return self
    def predict(self, X):
        return self.lin.predict((poly_features(X, self.degree) - self.mu) / self.sd)

class KNNRegressorScratch:
    """Predict = distance-weighted average of the k closest training points."""
    def __init__(self, k=10):
        self.k = k
    def fit(self, X, y):
        self.X, self.y = X, y
        return self
    def predict(self, X):
        out = []
        sq_train = (self.X ** 2).sum(1)
        for i in range(0, len(X), 1000):                       # chunks -> low memory
            A = X[i:i + 1000]
            d2 = (A ** 2).sum(1)[:, None] + sq_train[None, :] - 2 * A @ self.X.T
            d2 = np.maximum(d2, 0)
            nn = np.argpartition(d2, self.k, axis=1)[:, :self.k]
            w = 1 / (np.sqrt(np.take_along_axis(d2, nn, 1)) + 1e-9)
            out.append((self.y[nn] * w).sum(1) / w.sum(1))
        return np.concatenate(out)

class BlendScratch:
    """Best model: weighted average of PolyRidge + KNN.  Two different kinds of model
    make different mistakes, so averaging cancels part of the error (ensemble idea)."""
    def __init__(self, degree=2, lam=10.0, k=15, w=0.5):
        self.poly, self.knn, self.w = PolyRegressionScratch(degree, lam), KNNRegressorScratch(k), w
    def fit(self, X, y):
        self.poly.fit(X, y); self.knn.fit(X, y)
        return self
    def predict(self, X):
        return self.w * self.poly.predict(X) + (1 - self.w) * self.knn.predict(X)
'''

TUNE_BEST = '''
# tune the Blend on a VALIDATION split carved out of TRAIN (the test set stays untouched)
Xa, Xv, ya, yv = train_test_split(Xtr, y_train, test_size=0.2, random_state=0)
best = (-np.inf, None)
for lam in [1, 10, 100]:
    p_poly = PolyRegressionScratch(2, lam).fit(Xa, ya).predict(Xv)
    for k in [5, 10, 15, 25]:
        p_knn = KNNRegressorScratch(k).fit(Xa, ya).predict(Xv)
        for w in [0, 0.3, 0.5, 0.7, 1]:
            score = r2_scratch(yv, w * p_poly + (1 - w) * p_knn)
            if score > best[0]:
                best = (score, dict(lam=lam, k=k, w=w))
print("best validation R2 = %.4f with %s" % best)
m4 = BlendScratch(degree=2, **best[1]).fit(Xtr, y_train)   # refit on ALL training data
'''

LIB_CLS_TOOLS = '''
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
'''


def exam_cells(cfg, explain=True):
    """Full exam solution. cfg: name, path, read_kw, target, prep (code), cls_rule (code), cls_note,
    given (dict code), reg_to_cls (code or None)."""
    E = (lambda s: md(s)) if explain else (lambda s: None)
    cells = [
        md(f"""# {cfg['title']}
Exam layout (from both instruction sheets):

| Part | Task | Points |
|---|---|---|
| 1 Regression **from scratch** | 1.1 train 4 models (Linear, Multiple, Polynomial, Best) + predict | 8 |
| | 1.2 compare R² and rank the models + predict the given value | 2 |
| 2 Classification **with library** | 2.1 train 4+ classifiers | 8 |
| | 2.2 compare accuracy and F1, rank them | 2 |
| | 2.3 use the regression predictions to predict the class | 3 |

Dataset: **{cfg['name']}**. Regression target: `{cfg['target']}`. {cfg['cls_note']}"""),
        md("## 0. Load and prepare the data"),
        code(IMPORTS),
        code(f"""df = pd.read_csv({cfg['path']!r}{cfg.get('read_kw', '')})
print(df.shape)
df.head()"""),
        E("""Always look first: **shape**, **missing values**, **column types**.
Text columns must become numbers (one-hot via `pd.get_dummies`), missing rows dropped or filled."""),
        code("""print(df.isnull().sum().sum(), "missing values")
print(df.dtypes.value_counts())
df.describe().T.head(15)"""),
        code(cfg["prep"]),
        E("""**Train/test split.** The model learns only from TRAIN. TEST is the "exam" for the model:
data it has never seen. `random_state=42` makes the split identical every run, so you and your friends
are compared on the same rows. (If the teacher bans sklearn in Part 1, swap in the numpy split below.)"""),
        code("""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# numpy-only alternative:
# idx = np.random.default_rng(42).permutation(len(X)); n_te = int(0.2 * len(X))
# X_test, X_train, y_test, y_train = X[idx[:n_te]], X[idx[n_te:]], y[idx[:n_te]], y[idx[n_te:]]
print(X_train.shape, X_test.shape)"""),
        E("""**Standardise** (z-score): `(x - mean) / std` using the TRAIN mean/std only.
Gradient descent and KNN break when one column is in thousands and another in 0.01."""),
        code("""mu, sd = X_train.mean(0), X_train.std(0)
sd[sd == 0] = 1                       # constant column -> avoid divide by zero
Xtr = (X_train - mu) / sd
Xte = (X_test - mu) / sd"""),
        md("---\n# Part 1 — Regression from scratch\n## 1.1 The from-scratch toolbox"),
        E("""Every model here is one idea:

* **Linear / Multiple**: `y_hat = θ0 + θ1·x1 + ... + θn·xn`. Find θ that minimises MSE.
  * Normal equation (OLS): `θ = (XᵀX)⁻¹ Xᵀy` — exact answer in one step.
  * Gradient descent: start at θ = 0 and repeat `θ ← θ − lr · (2/n)·Xᵀ(Xθ − y)`.
* **Polynomial**: add new columns `x²`, `x1·x2`, ... then run linear regression on them. It's still "linear" in θ.
* **Ridge (λ)**: adds `λ·Σθ²` to the loss, so big wiggly weights are punished (W10 regularisation).
* **KNN regressor**: no equation at all. Predict the average y of the k nearest training rows.
* **Blend (best)**: average of PolyRidge and KNN."""),
        code(SCRATCH_TOOLS),
        md("### Model 1 — Simple Linear Regression (1 feature, gradient descent)"),
        E("""Simple linear regression uses **one** input. Pick the feature most correlated with the target.
It's trained with gradient descent so you can see the loss go down. Then it's checked against the normal equation:
both must give the same θ."""),
        code("""corr = [abs(np.corrcoef(Xtr[:, j], y_train)[0, 1]) for j in range(Xtr.shape[1])]
best_j = int(np.argmax(corr))
print("best single feature:", feature_names[best_j], "| corr = %.3f" % corr[best_j])

m1 = LinearRegressionScratch("gd", lr=0.05, epochs=1000).fit(Xtr[:, [best_j]], y_train)
m1_check = LinearRegressionScratch("ols").fit(Xtr[:, [best_j]], y_train)
print("GD  theta:", m1.theta)
print("OLS theta:", m1_check.theta, " <- should match")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(m1.loss_history); ax[0].set_title("GD loss (MSE) per epoch"); ax[0].set_xlabel("epoch")
ax[1].scatter(Xte[:, best_j], y_test, s=6, alpha=.4, label="test data")
xs = np.linspace(Xte[:, best_j].min(), Xte[:, best_j].max(), 50)
ax[1].plot(xs, m1.predict(xs[:, None]), "r", label="fitted line"); ax[1].legend()
ax[1].set_xlabel(feature_names[best_j] + " (scaled)"); plt.show()"""),
        md("### Model 2 — Multiple Linear Regression (all features)"),
        code("""m2 = LinearRegressionScratch("ols").fit(Xtr, y_train)
print("intercept:", round(m2.theta[0], 4))
print(pd.Series(m2.theta[1:], index=feature_names).sort_values(key=abs, ascending=False).head(8))"""),
        E("""With scaled inputs, a weight's size shows how much that feature matters, and its sign shows the direction."""),
        md("### Model 3 — Polynomial Regression (degree 2)"),
        E("""Degree 2 adds every square and every pair product. With n features that's n + n(n+1)/2 columns.
A tiny λ (Ridge) keeps the solve stable, because many of those columns are correlated."""),
        code("""m3 = PolyRegressionScratch(degree=2, lam=1.0).fit(Xtr, y_train)
print("number of polynomial features:", poly_features(Xtr[:2], 2).shape[1])"""),
        md("### Model 4 — Best model: PolyRidge + KNN blend (tuned)"),
        E("""Why this wins: linear and polynomial models capture smooth global trends. KNN captures local patterns
the equation misses. Averaging them cancels part of each model's error.
**Tuning rule:** choose λ, k and the weight on a *validation* split taken from TRAIN. Never tune on TEST.
Otherwise the score is fake, and your ranking against friends won't hold up."""),
        code(TUNE_BEST),
        md("## 1.2 Predict and compare R² (ranking)"),
        code("""preds_reg = {
    "1 Linear (1 feature, GD)": m1.predict(Xte[:, [best_j]]),
    "2 Multiple Linear (OLS)": m2.predict(Xte),
    "3 Polynomial deg2": m3.predict(Xte),
    "4 Best: Blend Poly+KNN": m4.predict(Xte),
}
rows = [(name, r2_scratch(y_test, p), rmse_scratch(y_test, p)) for name, p in preds_reg.items()]
table_reg = pd.DataFrame(rows, columns=["model", "R2", "RMSE"]).sort_values("R2", ascending=False)
table_reg.insert(0, "rank", range(1, len(table_reg) + 1))
display(table_reg)
table_reg.plot.barh(x="model", y="R2", legend=False, title="Test R² (higher is better)"); plt.show()"""),
        E("""**Reading R²:** 1.0 = perfect. 0 = no better than always guessing the mean. Below 0 = worse than the mean.
**Sanity check:** sklearn's `LinearRegression` must give the *same* R² as Model 2.
If it doesn't, your from-scratch code has a bug. That's a free self-check during the exam."""),
        code("""from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
print("sklearn R2 :", r2_score(y_test, LinearRegression().fit(Xtr, y_train).predict(Xte)))
print("scratch R2 :", r2_scratch(y_test, preds_reg["2 Multiple Linear (OLS)"]))"""),
        md("### Predict the value the teacher gives (ค่าที่ให้)"),
        E("""The given row has to go through **exactly the same steps** as the training data:
same columns in the same order, same one-hot encoding, same scaling with the TRAIN mean/std.
`make_row` does all of that. Any column you leave out is filled with the training mean (or 0 for one-hot)."""),
        code(f"""def make_row(values: dict):
    raw = pd.DataFrame([values])
    raw = pd.get_dummies(raw).reindex(columns=feature_names, fill_value=0).astype(float)
    for c in feature_names:                      # columns not given -> training mean
        if c not in values and not any(c.startswith(k + "_") for k in values):
            raw[c] = X_train[:, feature_names.index(c)].mean()
    return (raw.values - mu) / sd

given = {cfg['given']}
x_new = make_row(given)
for name, m in [("1 Linear", m1), ("2 Multiple", m2), ("3 Polynomial", m3), ("4 Best", m4)]:
    xin = x_new[:, [best_j]] if name.startswith("1") else x_new
    print(f"{{name:14s}} -> predicted {cfg['target']} = {{m.predict(xin)[0]:.3f}}")"""),
        md("---\n# Part 2 — Classification with library (sklearn)\n## 2.1 Build the class label and train classifiers"),
        code(cfg["cls_rule"]),
        E("""Same `X`, same `test_size`, same `random_state=42` as Part 1 → **identical rows** in train/test.
So the Part-1 regression predictions line up row-for-row with this test set (needed in 2.3).
(Adding `stratify=y_cls` would keep class ratios equal, but it would also change the rows, so skip it here.)"""),
        code("""Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_cls, test_size=0.2, random_state=42)
print(pd.Series(yc_train).value_counts(normalize=True).round(3))"""),
        code(LIB_CLS_TOOLS),
        E("""Five models, one per family you studied:
* **Logistic Regression** (W4): linear boundary, outputs a probability through the sigmoid.
* **KNN** (W6): majority vote of the k nearest points. Needs scaling.
* **SVM** (W6): maximum-margin boundary. The RBF kernel lets the boundary curve. Needs scaling.
* **Decision Tree** (W5): if/else splits. No scaling needed. Overfits easily, so limit `max_depth`.
* **Random Forest** (W5): many trees on random subsets, majority vote. Usually the strongest default.

`make_pipeline(StandardScaler(), model)` fits the scaler on train only, so test data never leaks into it."""),
        code("""classifiers = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "KNN (k=15)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance")),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=3, gamma="scale")),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
}
for name, model in classifiers.items():
    model.fit(Xc_train, yc_train)
    print("trained", name)"""),
        md("### Best classifier — tuned with cross-validation"),
        E("""`GridSearchCV` tries every setting with 5-fold cross-validation **on train only**,
then refits the best one. `scoring="f1_macro"` tunes for F1 directly, because F1 is what the exam ranks."""),
        code("""grid = GridSearchCV(
    HistGradientBoostingClassifier(random_state=42),
    {"learning_rate": [0.05, 0.1], "max_depth": [None, 6], "max_iter": [200, 400]},
    cv=5, scoring="f1_macro", n_jobs=-1)
grid.fit(Xc_train, yc_train)
print("best params:", grid.best_params_, "| CV F1 = %.4f" % grid.best_score_)
classifiers["Best: tuned Gradient Boosting"] = grid.best_estimator_"""),
        md("## 2.2 Predict and compare accuracy and F1 (ranking)"),
        E("""* **Accuracy** = correct / total. Misleading when classes are imbalanced: with 95% "no", always saying "no" scores 95%.
* **Precision** = TP / (TP+FP): of the rows I called positive, how many really were?
* **Recall** = TP / (TP+FN): of the real positives, how many did I catch?
* **F1** = 2·P·R / (P+R): the harmonic mean, which is only high if both are high.
* **Macro F1** (multi-class) = the plain average of the per-class F1, so every class counts equally."""),
        code("""avg = "binary" if len(np.unique(y_cls)) == 2 else "macro"
preds_cls = {name: m.predict(Xc_test) for name, m in classifiers.items()}
rows = [(n, accuracy_score(yc_test, p), f1_score(yc_test, p, average=avg)) for n, p in preds_cls.items()]
table_cls = pd.DataFrame(rows, columns=["model", "accuracy", "F1"]).sort_values("F1", ascending=False)
table_cls.insert(0, "rank", range(1, len(table_cls) + 1))
display(table_cls)

best_name = table_cls.iloc[0]["model"]
ConfusionMatrixDisplay.from_predictions(yc_test, preds_cls[best_name], cmap="Blues")
plt.title("Confusion matrix — " + best_name); plt.show()
print(classification_report(yc_test, preds_cls[best_name]))"""),
        md("## 2.3 Use the regression predictions to predict the class"),
        E(cfg["reg_to_cls_note"]),
        code(cfg["reg_to_cls"]),
        md("### Classify the teacher's given value"),
        code("""x_raw = make_row(given) * sd + mu             # back to unscaled (classifiers take raw X)
for name in table_cls["model"].head(3):
    print(f"{name:32s} -> class {classifiers[name].predict(x_raw)[0]}")"""),
        md("""---
### Exam checklist
1. Same `random_state=42` everywhere → fair comparison and reproducible numbers.
2. Never fit a scaler or tune anything on TEST.
3. Scratch Model 2 R² == sklearn `LinearRegression` R² → your OLS is correct.
4. Print a **ranked table** for Part 1 (R²) and Part 2 (accuracy and F1). That's the 2-point "compare/rank" item.
5. Write one sentence on *why* the winner wins. Teachers give marks for the reasoning."""),
    ]
    return [c for c in cells if c is not None]

# ---------------------------------------------------------------- configs
WINE = dict(
    title="Mock Exam 1 — Red Wine Quality (FULL SOLUTION, explained)",
    name="UCI Red Wine Quality (1599 wines, 11 chemistry features)",
    path="data/winequality-red.csv", read_kw=", sep=';'", target="quality",
    cls_note="Classification target: **good wine** = quality ≥ 6 (1) else 0.",
    prep="""TARGET = "quality"
df = df.dropna()
X_df = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=True).astype(float)
feature_names = list(X_df.columns)
X = X_df.values
y = df[TARGET].values.astype(float)
print(len(feature_names), "features:", feature_names)""",
    cls_rule="""# class label: good wine (quality >= 6) = 1, else 0
y_cls = (df[TARGET].values >= 6).astype(int)
print(pd.Series(y_cls).value_counts())""",
    given="""{"fixed acidity": 7.4, "volatile acidity": 0.70, "citric acid": 0.0, "residual sugar": 1.9,
         "chlorides": 0.076, "free sulfur dioxide": 11, "total sulfur dioxide": 34,
         "density": 0.9978, "pH": 3.51, "sulphates": 0.56, "alcohol": 9.4}""",
    reg_to_cls_note="""Two ways to read *"predict == classification, take it from the regression predictions"*. Do **both**, it's cheap:

**A. Threshold.** The class is defined from the regression target (good = quality ≥ 6),
so turn each predicted quality into a class: `predicted quality ≥ 5.5 → good`.
Then score it with accuracy and F1 exactly like the classifiers.

**B. Stacking.** Add the regression prediction as an *extra feature* for a classifier.
Train-side predictions must be **out-of-fold** (made by a model that didn't see that row).
Otherwise the feature looks perfect in training and fails on test (leakage).""",
    reg_to_cls="""from sklearn.model_selection import KFold

# A. threshold the Part-1 regression predictions (same rows as yc_test because same split)
assert np.array_equal((y_test >= 6).astype(int), yc_test)
rows = []
for name, p in preds_reg.items():
    cls_from_reg = (p >= 5.5).astype(int)
    rows.append((name + " -> threshold", accuracy_score(yc_test, cls_from_reg), f1_score(yc_test, cls_from_reg)))

# B. stacking: out-of-fold Blend predictions as a new feature
oof = np.zeros(len(Xtr))
for tr_i, va_i in KFold(5, shuffle=True, random_state=42).split(Xtr):
    oof[va_i] = BlendScratch(2, **best[1]).fit(Xtr[tr_i], y_train[tr_i]).predict(Xtr[va_i])
X_stack_tr = np.c_[Xc_train, oof]
X_stack_te = np.c_[Xc_test, preds_reg["4 Best: Blend Poly+KNN"]]
stack = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(X_stack_tr, yc_train)
p = stack.predict(X_stack_te)
rows.append(("RF + regression feature (stacking)", accuracy_score(yc_test, p), f1_score(yc_test, p)))

table_23 = pd.concat([table_cls[["model", "accuracy", "F1"]],
                      pd.DataFrame(rows, columns=["model", "accuracy", "F1"])]).sort_values("F1", ascending=False)
table_23.insert(0, "rank", range(1, len(table_23) + 1))
display(table_23.reset_index(drop=True))""",
)

STEEL = dict(
    title="Practice Exam 2 — Steel Industry Energy (SOLUTION)",
    name="UCI Steel Industry Energy Consumption (35,040 rows, numeric + text columns)",
    path="data/Steel_industry_data.csv", read_kw="", target="Usage_kWh",
    cls_note="Classification target: `Load_Type` (3 classes: Light / Medium / Maximum load).",
    prep="""TARGET = "Usage_kWh"
df = df.dropna()
# feature engineering: NSM = seconds since midnight -> hour of day is easier to learn
df["hour"] = df["NSM"] / 3600
X_df = df.drop(columns=[TARGET, "date", "Load_Type", "NSM"])   # Load_Type is the Part-2 label: never use it as input
X_df = pd.get_dummies(X_df, drop_first=True).astype(float)      # WeekStatus, Day_of_week -> 0/1 columns
feature_names = list(X_df.columns)
X = X_df.values
y = df[TARGET].values.astype(float)
print(len(feature_names), "features:", feature_names)""",
    cls_rule="""y_cls = np.array(df["Load_Type"].tolist())   # plain numpy array of strings
print(pd.Series(y_cls).value_counts())""",
    given="""{"Lagging_Current_Reactive.Power_kVarh": 20.0, "Leading_Current_Reactive_Power_kVarh": 0.0,
         "CO2(tCO2)": 0.02, "Lagging_Current_Power_Factor": 85.0, "Leading_Current_Power_Factor": 100.0,
         "hour": 10.5, "WeekStatus": "Weekday", "Day_of_week": "Tuesday"}""",
    reg_to_cls_note="""Here the class (`Load_Type`) isn't a threshold of `Usage_kWh`, so use **stacking**:
add the predicted kWh as an extra feature (out-of-fold on train to avoid leakage), then compare F1 with and without it.""",
    reg_to_cls="""from sklearn.model_selection import KFold
oof = np.zeros(len(Xtr))
for tr_i, va_i in KFold(5, shuffle=True, random_state=42).split(Xtr):
    oof[va_i] = BlendScratch(2, **best[1]).fit(Xtr[tr_i], y_train[tr_i]).predict(Xtr[va_i])
X_stack_tr = np.c_[Xc_train, oof]
X_stack_te = np.c_[Xc_test, preds_reg["4 Best: Blend Poly+KNN"]]
rows = []
for name in ["Random Forest", "Best: tuned Gradient Boosting"]:
    from sklearn.base import clone
    m = clone(classifiers[name]).fit(X_stack_tr, yc_train)
    p = m.predict(X_stack_te)
    rows.append((name + " + regression feature", accuracy_score(yc_test, p), f1_score(yc_test, p, average="macro")))
table_23 = pd.concat([table_cls[["model", "accuracy", "F1"]],
                      pd.DataFrame(rows, columns=["model", "accuracy", "F1"])]).sort_values("F1", ascending=False)
table_23.insert(0, "rank", range(1, len(table_23) + 1))
display(table_23.reset_index(drop=True))""",
)


TEMPLATE = dict(
    title="EXAM TEMPLATE — edit only the lines marked CHANGE ME (runs on wine as the example)",
    name="whatever CSV the teacher gives (example: wine)",
    path="data/winequality-red.csv", read_kw=", sep=';'", target="quality  (set TARGET below)",
    cls_note="Set CLASS_COL if the CSV has a class column, otherwise THRESHOLD makes classes from the target.",
    prep="""TARGET = "quality"          # CHANGE ME: regression target column
CLASS_COL = None            # CHANGE ME: class column name if one exists (e.g. "Load_Type"), else None
THRESHOLD = 6               # CHANGE ME: used only when CLASS_COL is None -> class = target >= THRESHOLD
DROP_COLS = []              # CHANGE ME: id / date / leaking columns

df = df.dropna()
X_df = df.drop(columns=[TARGET] + DROP_COLS + ([CLASS_COL] if CLASS_COL else []))
X_df = pd.get_dummies(X_df, drop_first=True).astype(float)
feature_names = list(X_df.columns)
X = X_df.values
y = df[TARGET].values.astype(float)
print(len(feature_names), "features:", feature_names)""",
    cls_rule="""if CLASS_COL:
    y_cls = np.array(df[CLASS_COL].tolist())
else:
    y_cls = (df[TARGET].values >= THRESHOLD).astype(int)
print(pd.Series(y_cls).value_counts())""",
    given="""{"alcohol": 9.4, "volatile acidity": 0.70, "sulphates": 0.56}   # CHANGE ME: the teacher's values""",
    reg_to_cls_note="2.3 — threshold the regression predictions (only if classes come from the target) and stack them as a feature.",
    reg_to_cls="""from sklearn.model_selection import KFold
from sklearn.base import clone
rows = []
if not CLASS_COL:   # A. threshold (integer target -> boundary at THRESHOLD - 0.5)
    cut = THRESHOLD - 0.5 if np.all(y == np.round(y)) else THRESHOLD
    for name, p in preds_reg.items():
        c = (p >= cut).astype(int)
        rows.append((name + " -> threshold", accuracy_score(yc_test, c), f1_score(yc_test, c, average=avg)))
# B. stacking: out-of-fold regression prediction as an extra feature
oof = np.zeros(len(Xtr))
for tr_i, va_i in KFold(5, shuffle=True, random_state=42).split(Xtr):
    oof[va_i] = BlendScratch(2, **best[1]).fit(Xtr[tr_i], y_train[tr_i]).predict(Xtr[va_i])
X_stack_tr = np.c_[Xc_train, oof]
X_stack_te = np.c_[Xc_test, preds_reg["4 Best: Blend Poly+KNN"]]
top = table_cls.iloc[0]["model"]
p = clone(classifiers[top]).fit(X_stack_tr, yc_train).predict(X_stack_te)
rows.append((top + " + regression feature", accuracy_score(yc_test, p), f1_score(yc_test, p, average=avg)))
table_23 = pd.concat([table_cls[["model", "accuracy", "F1"]],
                      pd.DataFrame(rows, columns=["model", "accuracy", "F1"])]).sort_values("F1", ascending=False)
table_23.insert(0, "rank", range(1, len(table_23) + 1))
display(table_23.reset_index(drop=True))""",
)

# ---------------------------------------------------------------- blank practice exam
def blank_cells(cfg):
    sol = exam_cells(cfg, explain=False)
    keep_code = {0: True}
    out = []
    for i, (t, s) in enumerate(sol):
        if t == "md":
            out.append((t, s))
        elif "pd.read_csv" in s or s.startswith("import numpy") or s.startswith("from sklearn.preprocessing") or s.startswith("# ===== from-scratch"):
            out.append((t, s))   # give loading/imports/toolbox signatures
        else:
            out.append(("code", "# TODO — write it yourself, then compare with the SOLUTION notebook\n"))
    # replace toolbox body with signatures only
    stub = '''# ===== from-scratch toolbox: implement each function/class =====
def r2_scratch(y, p):  ...                 # 1 - SS_res/SS_tot
def rmse_scratch(y, p): ...
def add_bias(X): ...                       # prepend a column of ones
class LinearRegressionScratch:             # method="ols" | "gd", lam=Ridge
    def __init__(self, method="ols", lr=0.05, epochs=2000, lam=0.0): ...
    def fit(self, X, y): ...
    def predict(self, X): ...
def poly_features(X, degree=2): ...
class PolyRegressionScratch: ...
class KNNRegressorScratch: ...
class BlendScratch: ...'''
    out = [(t, stub if s.startswith("# ===== from-scratch") else s) for t, s in out]
    out[0] = ("md", out[0][1].replace("(SOLUTION)", "(BLANK — do it timed: 2 hours)"))
    return out

# ---------------------------------------------------------------- write
import json, sys
nbf.write(nb(exam_cells(WINE)), "01_MockExam1_Wine_SOLUTION.ipynb")
nbf.write(nb(exam_cells(STEEL, explain=False)), "02_PracticeExam2_Steel_SOLUTION.ipynb")
nbf.write(nb(blank_cells(STEEL)), "02_PracticeExam2_Steel_BLANK.ipynb")
nbf.write(nb(exam_cells(TEMPLATE, explain=False)), "03_EXAM_TEMPLATE.ipynb")
print("built")
