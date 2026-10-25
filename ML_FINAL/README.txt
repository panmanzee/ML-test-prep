ML FINAL — open in this order
==============================

1_MAIN          YOUR code (what you use in the test), plus a ranking table and better graphs
  Q1_Regression_from_scratch.ipynb          Linear / Multi / Poly / Boost  -> 4 graphs + R2 ranking
  Q2_Classification_then_Regression.ipynb   Logistic / KNN / DecisionTree / HistGradientBoosting
                                            -> accuracy+F1 bar chart + ranking
                                            then + pred_class: Linear / Multi / Poly / HistGradientBoosting
                                            -> 4 graphs + R2 ranking
  Q3_Other_Metrics.ipynb                    precision, recall, confusion matrix, report, ROC/AUC,
                                            MAE, MSE, RMSE, MAPE (+ by hand), ranking tables with them
  Q4_Poly_Best_Degree.ipynb                 which poly degree? -> train vs CV R2 graph, picks it for you
  Q5_Auto_Prepare_One_Function.ipynb        ONE def auto_prepare(): load any file + clean + split + best poly degree
  Q6_Make_Your_Own_Data.ipynb               ONE def make_data(): no data given? paste a table or build one from the rule, then auto_prepare()
                                            copy cell 1 into your exam notebook, then d = auto_prepare("file", target="y")

2_DATA_PREP
  Load_Any_Data.ipynb                       NO csv? xlsx / json / txt / zip / URL / sklearn dataset / numbers in the question
  Auto_Clean.ipynb                          auto_clean(): cleans the whole table by itself, prints every decision
                                            (titanic: accuracy 0.70 -> 0.82 with the same model)
  Data_Prep_Examples.ipynb                  every cleaning case: drop rows/cols, fill 0 / mean / median /
                                            mode / group / ffill, text->number, dates, outliers, errors table

3_SPARE         only if needed
  Spare_Q1_auto_better.ipynb / Spare_Q2_auto_better.ipynb
                                            same questions, but "Better" is picked by cross-validation
                                            (stronger score: Q1 R2 0.66 vs Boost)
  Spare_More_Models.ipynb                   GD, Ridge, KNN, SVM, RandomForest, XGBoost, Voting, Stacking, tuning
  Spare_Metrics_and_Predict.ipynb           metrics by hand, confusion matrix, predict a given row, multi-class
  ALL_IN_ONE_auto.ipynb                     old big version

In the exam
  EMERGENCY: dirty data or not a csv -> clean_tools.py (in this folder):
      df = auto_clean(load_any("../file.xlsx"), target="y")   (ready, commented, in Q1/Q2 cell)
  1. Put the teacher's csv in this ML_FINAL folder.
  2. Change the lines marked  # <<<  (file, cols, target).
  3. Run All.
