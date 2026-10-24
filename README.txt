ML PRACTICE EXAM PREP — START HERE
==================================

Open the notebooks in Jupyter or VS Code from this folder, in this order:

  00_CrashCourse.ipynb                 ML from zero: every idea the exam needs, with runnable cells
                                       and "check yourself" questions. ~2-3 h.
  01_MockExam1_Wine_SOLUTION.ipynb     A full exam in the real Part 1 / Part 2 layout, explained cell by cell.
  02_PracticeExam2_Steel_BLANK.ipynb   The same exam on a NEW dataset with empty cells. Do it timed.
  02_PracticeExam2_Steel_SOLUTION.ipynb   Check your work against this.
  03_EXAM_TEMPLATE.ipynb               Generic version. In the exam, change only the CHANGE ME lines.
  CHEATSHEET.txt                       Formulas, must-know code, common errors. Keep it open on the laptop.
  data/                                The datasets (offline copies).

STUDY PLAN (1 week)
  Day 1   00_CrashCourse sections 1-5 (regression, GD, R2, polynomial/Ridge)
  Day 2   00_CrashCourse sections 6-10 (classification, metrics, the sklearn models)
  Day 3   Read 01 Wine solution. Then retype Part 1 from memory into an empty notebook.
  Day 4   Retype Part 2 from memory. Hand-compute precision/recall/F1 from a confusion matrix.
  Day 5   02 Steel BLANK, timed (2 h), no peeking. Then compare with the solution.
  Day 6   Redo whatever broke. Practise make_row / predicting a given value on both datasets.
  Day 7   Run 03 template on a dataset you've never seen (ask Hermes for one). Read the cheat sheet once.

Retyping matters: the public PC may not let you copy files from the laptop.
The from-scratch toolbox cell (~100 lines) is the thing to be able to type fast.

Rebuild the notebooks: python build.py ; python build00.py   (the source lives on the minipc in /root/ml-test-prep)
