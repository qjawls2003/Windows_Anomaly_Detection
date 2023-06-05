import pandas as pd
import os
from pathlib import Path

def csv_to_df():


    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../data/WinEvent4688.csv')
    print(filename)

    df = pd.read_csv(filename)
    print(df)

