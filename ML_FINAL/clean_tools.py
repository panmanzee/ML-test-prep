"""
clean_tools.py  ->  one function that cleans almost any table.

    import sys; sys.path.insert(0, "..")        # notebooks in a subfolder
    from clean_tools import auto_clean, load_any

    df = load_any("data.xlsx")                   # csv / xlsx / json / txt / tsv / parquet ...
    df = auto_clean(df, target="Price")          # prints every decision it makes

auto_clean steps (each one prints what it did):
  1  clean column names (strip spaces)
  2  "?", "", "NA", "-", "null", -999 ...  ->  NaN
  3  drop duplicate rows
  4  drop rows where the TARGET is missing (never fill y)
  5  text that is really numbers ("1,200", "$50", "30%")  ->  numbers
  6  date columns  ->  year / month / day / weekday (then drop the date)
  7  drop useless columns: id-like, text where >90% rows differ (names), one single value, > max_missing empty
  8  fill missing: numbers -> median (+ a "_missing" 0/1 flag if many missing), text -> mode
  9  text -> numbers: 2 values -> 0/1, few values -> one-hot, too many values -> frequency
 The target column is only cleaned (step 2/4/5), never encoded or dropped,
 except text target -> numbers (0, 1, 2 ...) so classifiers work.
"""
import re
import numpy as np
import pandas as pd

MISSING = ["?", "", " ", "NA", "N/A", "na", "n/a", "NaN", "nan", "null", "NULL", "None", "none", "-", "--", "missing"]


def load_any(path, **kw):
    """Open csv / tsv / txt / xlsx / xls / json / parquet / zip by its extension."""
    p = str(path).lower()
    if p.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, **kw)
    if p.endswith(".json"):
        try:
            return pd.read_json(path, **kw)
        except ValueError:
            return pd.read_json(path, lines=True, **kw)
    if p.endswith(".parquet"):
        return pd.read_parquet(path, **kw)
    # csv / tsv / txt / zip: let pandas guess the separator (, ; tab |)
    for enc in ["utf-8-sig", "cp874", "latin-1"]:
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=enc, **kw)
            break
        except UnicodeDecodeError:
            continue
    # first row was data, not a header? (all header names look like numbers)
    if all(re.fullmatch(r"-?\d+(\.\d+)?", str(c).strip()) for c in df.columns):
        df = pd.read_csv(path, sep=None, engine="python", header=None, encoding=enc, **kw)
        df.columns = [f"col{i}" for i in range(df.shape[1])]
        print("load_any: no header found -> columns named col0, col1, ...")
    return df


def _to_number(s):
    """'1,200' '$50' '30%' ' 7 ' -> 1200 50 30 7 ; returns None if the column is not numeric text."""
    t = s.astype(str).str.strip().str.replace(r"[,\s$฿€£%]", "", regex=True)
    num = pd.to_numeric(t, errors="coerce")
    ok = s.notna().sum()
    if ok and num.notna().sum() >= 0.9 * ok:
        return num
    return None


def _to_date(s):
    if pd.api.types.is_numeric_dtype(s):
        return None
    sample = s.dropna().astype(str).head(200)
    if sample.empty or not sample.str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}").mean() > 0.8:
        return None
    d = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return d if d.notna().mean() > 0.8 else None


def auto_clean(df, target=None, max_missing=0.5, max_onehot=10, flag_missing=0.05, drop_dup=True, verbose=True):
    df = df.copy()
    say = print if verbose else (lambda *a: None)
    say(f"start: {df.shape[0]} rows x {df.shape[1]} cols")

    # 1 column names
    df.columns = [str(c).strip() for c in df.columns]
    if target is not None:
        target = str(target).strip()
        assert target in df.columns, f"target '{target}' not in columns: {list(df.columns)}"

    # 2 missing markers
    df = df.replace(MISSING, np.nan)
    for c in df.select_dtypes("number"):
        if (df[c] == -999).any():
            df[c] = df[c].replace(-999, np.nan)
            say(f"  {c}: -999 -> NaN")
    for c in df.select_dtypes(exclude="number"):
        t = df[c].astype(str).str.strip()
        if c != target:
            t = t.str.lower()                      # " BKK" and "bkk " -> "bkk"
        df[c] = df[c].where(df[c].isna(), t)

    # 3 duplicates
    n = df.duplicated().sum()
    if n and drop_dup:
        df = df.drop_duplicates()
        say(f"  dropped {n} duplicate rows")

    # 5 numeric text -> numbers (before step 4 so the target is numeric too)
    for c in list(df.select_dtypes(exclude="number")):
        num = _to_number(df[c])
        if num is not None:
            df[c] = num
            say(f"  {c}: text -> number")

    # 4 target missing
    if target is not None:
        n = df[target].isna().sum()
        if n:
            df = df[df[target].notna()]
            say(f"  dropped {n} rows with missing target '{target}'")

    feats = [c for c in df.columns if c != target]

    # 6 dates
    for c in list(feats):
        d = _to_date(df[c])
        if d is not None:
            df[c + "_year"], df[c + "_month"] = d.dt.year, d.dt.month
            df[c + "_day"], df[c + "_weekday"] = d.dt.day, d.dt.dayofweek
            if (d.dt.hour > 0).any():
                df[c + "_hour"] = d.dt.hour
            df = df.drop(columns=c)
            say(f"  {c}: date -> year/month/day/weekday")
    feats = [c for c in df.columns if c != target]

    # 7 useless columns
    rows = len(df)
    for c in feats:
        miss = df[c].isna().mean()
        nun = df[c].nunique()
        name = c.lower()
        if miss > max_missing:
            why = f"{miss:.0%} missing"
        elif nun <= 1:
            why = "only one value"
        elif re.search(r"(id|index|_no)$|^unnamed", name) and nun > 0.9 * rows:
            why = "id column"
        elif not pd.api.types.is_numeric_dtype(df[c]) and nun > 0.9 * rows:
            why = f"text with {nun} different values (names / free text)"
        elif (rows > 50 and pd.api.types.is_integer_dtype(df[c]) and nun == rows
              and (df[c].to_numpy() == np.arange(df[c].iloc[0], df[c].iloc[0] + rows)).all()):
            why = "row number (0, 1, 2, ...)"
        else:
            continue
        df = df.drop(columns=c)
        say(f"  drop {c}: {why}")
    feats = [c for c in df.columns if c != target]

    # 8 fill missing
    for c in feats:
        miss = df[c].isna().mean()
        if miss == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if miss >= flag_missing:
                df[c + "_missing"] = df[c].isna().astype(int)
            med = df[c].median()
            df[c] = df[c].fillna(med)
            say(f"  {c}: {miss:.0%} missing -> median {med:g}" + (" (+ _missing flag)" if miss >= flag_missing else ""))
        else:
            mode = df[c].mode()[0]
            df[c] = df[c].fillna(mode)
            say(f"  {c}: {miss:.0%} missing -> mode '{mode}'")

    # 9 text -> numbers
    for c in [c for c in feats if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]:
        vals = df[c].astype(str)
        nun = vals.nunique()
        if nun == 2:
            top = sorted(vals.unique())
            df[c] = (vals == top[1]).astype(int)
            say(f"  {c}: 2 values -> 0/1  ({top[0]}=0, {top[1]}=1)")
        elif nun <= max_onehot:
            dummies = pd.get_dummies(df[c], prefix=c, drop_first=True, dtype=int)
            df = pd.concat([df.drop(columns=c), dummies], axis=1)
            say(f"  {c}: {nun} values -> one-hot ({dummies.shape[1]} cols)")
        else:
            df[c] = df[c].map(df[c].value_counts(normalize=True))
            say(f"  {c}: {nun} values -> frequency (how common each value is)")

    # target text -> 0, 1, 2 ...
    if target is not None and not pd.api.types.is_numeric_dtype(df[target]):
        cats = sorted(df[target].astype(str).unique())
        df[target] = df[target].astype(str).map({v: i for i, v in enumerate(cats)})
        say(f"  target {target}: " + ", ".join(f"{v}={i}" for i, v in enumerate(cats)))

    bool_cols = df.select_dtypes("bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    df = df.reset_index(drop=True)
    left = int(df.isna().sum().sum())
    say(f"done: {df.shape[0]} rows x {df.shape[1]} cols, missing left = {left}")
    return df
