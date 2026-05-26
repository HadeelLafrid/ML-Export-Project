import os
import pandas as pd

DASH_DIR = os.path.join('dashboard', '..', 'data', 'res', 'dashboard')

def load_csv(name):
    path = os.path.join(DASH_DIR, name)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    print(f'{name}: exists={exists}, size={size}')
    if exists and size > 35:
        df = pd.read_csv(path)
        print(f'  → loaded {len(df)} rows, empty={df.empty}')
        return df
    print(f'  → returned empty DataFrame')
    return pd.DataFrame()

cf = load_csv('classification_results.csv')
print()
print(f'Final check: cf.empty = {cf.empty}')
print(f"Dashboard will show: {'RESULTS' if not cf.empty else 'PENDING'}")