"""
Fleet-health screening report from the debiased per-cell VoltUp capacities.

Per-cycle SOH is noisy (~7% floor), but averaging ~33 cycles/cell drops the uncertainty to ~1-2%,
so per-cell SOH LEVEL and its observed change ARE trustworthy (unlike the per-cycle model). For
each cell we report (all SOH in % of 30 Ah nominal):
  * SOH_med     - robust current health (median of all cycles)
  * SOH_early / SOH_recent - median of first / last third of cycles
  * dObs        - observed change recent-early (%pts); the robust degradation signal
  * sig         - Theil-Sen trend 95% CI excludes 0 (statistically significant decline/rise)
  * status      - DECLINING / LOW / WATCH / OK
Cells are ranked weakest-first. Output: console summary + voltup_fleet_health.csv.

Caveats: trends are over each cell's OBSERVED window (12-68 cycles, a fraction of full life), so
they are local rates, not life-long. Calendar/temperature effects are not separated from cycle
aging. Use as a SCREEN to prioritise inspection, not as a calibrated SOH certificate.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import theilslopes, median_abs_deviation

ROOT = 'data/VOLTUP DATA'
NOMINAL = 30.0
OUT = 'voltup_fleet_health.csv'
MIN_N_TREND = 15          # below this, trend is too short to flag DECLINING (level still used)


def analyze(path):
    soh = pd.read_csv(path)['capacity'].to_numpy(float) / NOMINAL * 100.0   # SOH in %
    n = len(soh); x = np.arange(n)
    med = float(np.median(soh))
    k = max(2, n // 3)
    early = float(np.median(soh[:k]))
    recent = float(np.median(soh[-k:]))
    d_obs = recent - early                                  # %pts observed change (robust)
    rstd = float(median_abs_deviation(soh, scale='normal'))
    se = rstd / np.sqrt(n)
    slope, _, lo, hi = theilslopes(soh, x)                  # %pts per cycle
    return dict(n=n, soh_med=med, soh_early=early, soh_recent=recent, d_obs=d_obs,
                se=se, per100=slope * 100, lo100=lo * 100, hi100=hi * 100,
                sig_decline=(hi < 0), sig_rise=(lo > 0))


def main():
    files = sorted(f for f in os.listdir(ROOT) if f.lower().endswith('.csv'))
    df = pd.DataFrame([{**analyze(os.path.join(ROOT, f)), 'cell': f[:-4]} for f in files])

    fleet_med = df['soh_med'].median()
    p10 = df['soh_med'].quantile(0.10)

    def status(r):
        if r['n'] >= MIN_N_TREND and r['sig_decline'] and r['d_obs'] <= -3.0:
            return 'DECLINING'
        if r['soh_med'] <= p10 or r['soh_med'] < 85.0:
            return 'LOW'
        if r['sig_decline'] or r['d_obs'] <= -3.0 or r['soh_med'] < fleet_med - 3.0:
            return 'WATCH'
        return 'OK'
    df['status'] = df.apply(status, axis=1)
    # risk: low current health + observed decline (weakest first)
    df['risk'] = (100 - df['soh_med']) - np.minimum(df['d_obs'], 0)
    df = df.sort_values('risk', ascending=False).reset_index(drop=True)

    print('=' * 84)
    print(f'VOLTUP FLEET HEALTH REPORT   ({len(df)} cells, {int(df["n"].sum())} cycles)   SOH = capacity/30Ah')
    print('=' * 84)
    print(f'Fleet SOH%: median={fleet_med:.1f}  p10={p10:.1f}  '
          f'min={df["soh_med"].min():.1f}  max={df["soh_med"].max():.1f}')
    sc = df['status'].value_counts()
    print('Status: ' + '   '.join(f'{k}={int(v)}' for k, v in sc.items()))
    cm = df['d_obs'].mean()
    print(f'Fleet-wide common-mode drift (mean dObs) = {cm:+.1f}%pts  '
          f'[{(df.d_obs<0).mean()*100:.0f}% of cells decline, {(df.d_obs>0).mean()*100:.0f}% rise] '
          f'-> small seasonal/measurement component; flagged decliners sit well below this.')
    print('\nWEAKEST 20 CELLS (ranked by risk):')
    print(f'{"cell":11s} {"N":>3s} {"SOH%":>5s} {"early":>5s} {"recent":>6s} {"dObs":>6s} '
          f'{"trend %/100cyc":>16s} {"status":>9s}')
    print('-' * 84)
    for _, r in df.head(20).iterrows():
        sig = '*' if r['sig_decline'] else ('+' if r['sig_rise'] else ' ')
        print(f'{r["cell"]:11s} {int(r["n"]):3d} {r["soh_med"]:5.1f} {r["soh_early"]:5.1f} '
              f'{r["soh_recent"]:6.1f} {r["d_obs"]:+6.1f} '
              f'{r["per100"]:+6.1f}{sig}[{r["lo100"]:+.0f},{r["hi100"]:+.0f}]   {r["status"]:>9s}')
    print('-' * 84)
    print('dObs = recent-third minus early-third SOH (%pts, robust).  '
          '* = significant decline (95% CI<0), + = significant rise.')

    decl = df[df['status'] == 'DECLINING'].sort_values('d_obs')
    print(f'\n{len(decl)} cell(s) flagged DECLINING (N>={MIN_N_TREND}, significant trend, >=3%pts observed loss):')
    for _, r in decl.iterrows():
        print(f'  {r["cell"]}: SOH {r["soh_med"]:.1f}%  (early {r["soh_early"]:.1f} -> recent {r["soh_recent"]:.1f}, '
              f'{r["d_obs"]:+.1f}%pts over {int(r["n"])} cycles)')

    low = df[df['status'] == 'LOW'].sort_values('soh_med')
    print(f'\n{len(low)} cell(s) flagged LOW (current SOH in fleet bottom-10% or <85%):')
    for _, r in low.iterrows():
        print(f'  {r["cell"]}: SOH {r["soh_med"]:.1f}%  (N={int(r["n"])})')

    cols = ['cell', 'status', 'n', 'soh_med', 'soh_early', 'soh_recent', 'd_obs',
            'per100', 'lo100', 'hi100', 'sig_decline', 'se']
    df[cols].round(3).to_csv(OUT, index=False)
    print(f'\nFull ranked table ({len(df)} cells) -> {OUT}')


if __name__ == '__main__':
    main()
