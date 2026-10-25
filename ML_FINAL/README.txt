ML FINAL — open in this order
==============================

1_MAIN          YOUR code (what you use in the test), plus a ranking table and better graphs
  Q1_Regression_from_scratch.ipynb          Linear / Multi / Poly / Boost  -> 4 graphs + R2 ranking
  Q2_Classification_then_Regression.ipynb   Logistic / KNN / DecisionTree / GradientBoosting
                                            -> accuracy+F1 bar chart + ranking
                                            then + pred_class: Linear / Multi / Poly / GradientBoosting
                                            -> 4 graphs + R2 ranking

2_DATA_PREP     Data_Prep_Examples.ipynb    every cleaning case: drop rows/cols, fill 0 / mean / median /
                                            mode / group / ffill, text->number, dates, outliers, errors table

3_SPARE         only if needed
  Spare_Q1_auto_better.ipynb / Spare_Q2_auto_better.ipynb
                                            same questions, but "Better" is picked by cross-validation
                                            (stronger score: Q1 R2 0.66 vs Boost)
  Spare_More_Models.ipynb                   GD, Ridge, KNN, SVM, RandomForest, XGBoost, Voting, Stacking, tuning
  Spare_Metrics_and_Predict.ipynb           metrics by hand, confusion matrix, predict a given row, multi-class
  ALL_IN_ONE_auto.ipynb                     old big version

In the exam
  1. Put the teacher's csv in this ML_FINAL folder.
  2. Change the lines marked  # <<<  (file, cols, target).
  3. Run All.
