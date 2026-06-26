"""Validate every produced VoltUp CSV against the loader's contract."""
import os
import numpy as np
import pandas as pd

OUT_DIR = 'data/VOLTUP DATA'
NOMINAL = 30.0
EXPECTED = ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
            'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
            'current mean', 'current std', 'current kurtosis', 'current skewness',
            'CV Q', 'CV charge time', 'current slope', 'current entropy', 'capacity']

files = sorted(f for f in os.listdir(OUT_DIR) if f.lower().endswith('.csv'))
print(f'cells: {len(files)}  (need >=5)')
assert len(files) >= 5, 'need >=5 cells'

fail = 0
total_rows = 0
soh_all = []
cyc_counts = []
for f in files:
    p = os.path.join(OUT_DIR, f)
    df = pd.read_csv(p)
    errs = []
    # 1. exact 17 cols in order, capacity last
    if list(df.columns) != EXPECTED:
        errs.append(f'columns mismatch: {list(df.columns)}')
    if df.columns[-1] != 'capacity':
        errs.append('capacity not last')
    # 2. numeric, no NaN/inf
    if not all(np.issubdtype(dt, np.number) for dt in df.dtypes):
        errs.append('non-numeric column present')
    arr = df.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        errs.append(f'NaN/inf present ({(~np.isfinite(arr)).sum()} cells)')
    # 3. SOH range
    soh = df['capacity'].to_numpy() / NOMINAL
    if soh.min() < 0.5 or soh.max() > 1.1:
        errs.append(f'SOH out of [0.5,1.1]: [{soh.min():.3f},{soh.max():.3f}]')
    # 4. one row per cycle, has rows
    if len(df) < 1:
        errs.append('no rows')
    soh_all.append(soh)
    cyc_counts.append(len(df))
    total_rows += len(df)
    if errs:
        fail += 1
        print(f'  FAIL {f}: ' + '; '.join(errs))

soh_all = np.concatenate(soh_all)
print(f'\ntotal cycles: {total_rows}')
print(f'cycles/cell: min={min(cyc_counts)} median={int(np.median(cyc_counts))} max={max(cyc_counts)}')
print(f'SOH overall: min={soh_all.min():.3f} p50={np.percentile(soh_all,50):.3f} '
      f'max={soh_all.max():.3f} mean={soh_all.mean():.3f}')
print(f'SOH in [0.5,1.1]: {((soh_all>=0.5)&(soh_all<=1.1)).mean()*100:.1f}%')
print(f'\n{"ALL CHECKS PASSED" if fail==0 else f"{fail} CELL(S) FAILED"}')
