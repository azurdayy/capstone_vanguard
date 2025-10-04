import sys

from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve()
sys.path.insert(0, str(PROJECT_DIR))

def find_date_col(df: pd.DataFrame) -> str:
    """find date column in dataframe"""
    for c in df.columns:
        if c.lower() in ("date", "month", "time", "period"):
            return c
    return df.columns[0]

def load_feature_data(file_path: str | Path, table: int) -> pd.DataFrame:
    """load feature data from excel file"""
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    try:
        if file_path.suffix.lower() in ['.xlsx', '.xls']:
            xls = pd.ExcelFile(file_path)
            df = pd.read_excel(file_path, sheet_name=xls.sheet_names[table])
        elif file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Only .xlsx, .xls, and .csv are supported.")
    except Exception as e:
        raise IOError(f"An error occurred while reading the file: {e}")

    dtcol = find_date_col(df)
    df[dtcol] = pd.to_datetime(df[dtcol])
    return df.set_index(dtcol).sort_index()

def normalized(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std(ddof=0)
