"""Batch-extract every raw VoltUp cell into data/VOLTUP DATA/<cell>.csv (17-col schema)."""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from voltup_extract import extract_cell, COLS, NOMINAL

RAW_DIR = 'Voltup_raw_data'
OUT_DIR = 'data/VOLTUP DATA'
MIN_CYCLES = 12          # require at least this many clean cycles to emit a cell file
NPROC = 8


def process(fname):
    path = os.path.join(RAW_DIR, fname)
    stem = fname[:-4]
    try:
        df = extract_cell(path)
    except Exception as e:
        return (stem, 0, None, None, f'ERROR: {type(e).__name__}: {e}')
    n = len(df)
    if n < MIN_CYCLES:
        return (stem, n, None, None, 'skipped (too few cycles)')
    # final hard guarantees before writing
    df = df[COLS].astype(float)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(df) < MIN_CYCLES:
        return (stem, len(df), None, None, 'skipped after NaN drop')
    out = os.path.join(OUT_DIR, stem + '.csv')
    df.to_csv(out, index=False)
    soh = (df['capacity'] / NOMINAL).values
    return (stem, len(df), float(soh.min()), float(soh.max()), 'ok')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith('.csv'))
    print(f'extracting {len(files)} raw cells with {NPROC} workers -> {OUT_DIR}')
    t0 = time.time()
    results = []
    with Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(process, files), 1):
            results.append(r)
            print(f'[{i:3d}/{len(files)}] {r[0]:12s} cyc={r[1]:4d} '
                  f"soh=[{('%.3f'%r[2]) if r[2] is not None else '  -  '},"
                  f"{('%.3f'%r[3]) if r[3] is not None else '  -  '}] {r[4]}", flush=True)
    ok = [r for r in results if r[4] == 'ok']
    total_cyc = sum(r[1] for r in ok)
    print(f'\n=== DONE in {time.time()-t0:.0f}s ===')
    print(f'cells written: {len(ok)} / {len(files)}   total cycles: {total_cyc}')
    if ok:
        allmin = min(r[2] for r in ok); allmax = max(r[3] for r in ok)
        print(f'SOH range across all written cells: [{allmin:.3f}, {allmax:.3f}]')
    skipped = [r for r in results if r[4] != 'ok']
    if skipped:
        print(f'skipped/errored ({len(skipped)}):')
        for r in skipped:
            print(f'   {r[0]:12s} cyc={r[1]} {r[4]}')


if __name__ == '__main__':
    main()
