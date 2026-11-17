"""Load the UCI Polish Companies Bankruptcy ARFF files into pandas.

Careful with the file naming (from the official UCI description):
``1year.arff`` holds ratios from the *1st year of a 5-year forecasting
period*, and the label is the bankruptcy status *after 5 years*. So the file
number N means the data is ``6 - N`` years before the outcome:

    file        years_before_outcome
    1year.arff  5
    2year.arff  4
    3year.arff  3
    4year.arff  2
    5year.arff  1

The label ``class`` is 1 if the company went bankrupt, 0 otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.io import arff

from src.download_data import RAW_DIR

FEATURE_COLUMNS = [f"Attr{i}" for i in range(1, 65)]
FILE_YEARS = [1, 2, 3, 4, 5]


def years_before_outcome(file_year: int) -> int:
    """Convert the N in ``Nyear.arff`` to years between the data and the outcome."""
    return 6 - file_year


def load_year(file_year: int, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load one ARFF file (e.g. ``3year.arff``) as a clean DataFrame.

    Cleaning steps:
      * ``?`` entries are read by scipy as NaN (missing values).
      * The label arrives as bytes (``b'0'`` / ``b'1'``); convert to int 0/1.
      * Add ``file_year`` (N from the file name) and ``years_before_outcome``
        (6 - N) so the files can be stacked without losing track of timing.

    Args:
        file_year: The N in ``Nyear.arff``, 1 to 5.
        raw_dir: Folder containing the ARFF files.

    Returns:
        DataFrame with columns Attr1..Attr64 (float), ``class`` (int),
        ``file_year`` (int) and ``years_before_outcome`` (int).
    """
    if file_year not in FILE_YEARS:
        raise ValueError(f"file_year must be one of {FILE_YEARS}, got {file_year}")

    path = raw_dir / f"{file_year}year.arff"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.download_data` first."
        )

    records, _meta = arff.loadarff(path)
    df = pd.DataFrame(records)

    # Sanity check: the schema should match the dataset documentation.
    expected = FEATURE_COLUMNS + ["class"]
    if list(df.columns) != expected:
        raise ValueError(f"Unexpected columns in {path.name}: {list(df.columns)}")

    df["class"] = df["class"].str.decode("utf-8").astype(int)
    df["file_year"] = file_year
    df["years_before_outcome"] = years_before_outcome(file_year)
    return df


def load_polish(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load all five files and stack them into one DataFrame.

    Note: rows from different files may not be independent. The same
    company could appear in several files, and the dataset has no company
    ID to check this. That matters when designing the validation split.
    """
    frames = [load_year(n, raw_dir) for n in FILE_YEARS]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    data = load_polish()
    summary = data.groupby("years_before_outcome")["class"].agg(
        rows="size", bankrupt="sum"
    )
    summary["bankrupt_rate_%"] = (100 * summary["bankrupt"] / summary["rows"]).round(2)
    print(summary)
    print(f"\nTotal rows: {len(data):,}")
