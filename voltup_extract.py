"""
VoltUp raw -> per-charge-cycle feature extractor.

Raw format (one row = one multiplexed telemetry record; fields are SPARSE, different
record types carry different signals, so we merge by timestamp):
    sequence, timestamp(unix s), soc_user(%), amp(A, +=charge / -=discharge),
    volt(pack V, ~16S LFP ~51-56V), soh(BMS %), volt_1..volt_16 (per-cell V)

Cycle model (robust to real-world pulsed/partial field charging):
  * Merge rows by timestamp (mean over the few rows sharing a timestamp), ffill volt/soc.
  * Detect charge cycles from the SOC trajectory: each (local SOC valley -> next SOC peak)
    with a sufficient rise is one charge cycle, spanning [t_valley, t_peak]. This is robust
    to pulsing and to mixed charge/discharge sessions (uses the monotonic-ish SOC progress
    variable, not the noisy bouncing current).
  * Within a cycle window, "actively charging" samples = amp > I_MIN.
  * CC phase = charging samples with soc <  SOC_SPLIT  (bulk charge, voltage rising)
    CV phase = charging samples with soc >= SOC_SPLIT  (top-off/taper near full)
    (LFP voltage is flat, so SOC is a far more reliable CC/CV splitter than voltage.)
  * capacity (label) = net Ah delivered over the window / SOC swing fraction
    = full usable capacity estimate (~30 Ah fresh). SOH = capacity/30 (loader divides).

The 16 features (positional order the loader reads), capacity last:
  voltage mean/std/kurtosis/skewness (CC pack voltage),
  CC Q (Ah), CC charge time (s), voltage slope (V/s), voltage entropy (CC voltage hist),
  current mean/std/kurtosis/skewness (CV current),
  CV Q (Ah), CV charge time (s), current slope (A/s), current entropy (CV current hist),
  capacity (Ah)
"""
import warnings
import numpy as np
import pandas as pd
from scipy import stats as sstats
from scipy.signal import find_peaks

warnings.filterwarnings('ignore', category=RuntimeWarning)  # near-constant moments

COLS = ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
        'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
        'current mean', 'current std', 'current kurtosis', 'current skewness',
        'CV Q', 'CV charge time', 'current slope', 'current entropy', 'capacity']

# ---- tunable parameters (held constant across ALL cells for consistency) ----
I_MIN = 0.5          # A; |amp| above this = actively charging
SOC_SPLIT = 90.0     # % SOC dividing CC (bulk) from CV (taper)
SWING_MIN = 40.0     # % min SOC rise for a cycle to be kept (reliable capacity normalisation)
SOC_START_MAX = 40.0 # % cycle must start at/below this (so >=40% of the linear band is traversed)
SOC_END_MIN = 90.0   # % cycle must reach at/above this (ensures a real CV region for current feats)
PROM = 12.0          # % prominence for SOC peak/valley detection
MIN_CC_SAMPLES = 5
MIN_CV_SAMPLES = 5
NBINS = 50           # histogram bins for Shannon entropy
NOMINAL = 30.0
DT_CAP = 120.0       # s; cap per-step dt in integrals so multi-min/hour gaps don't inflate Ah/time
CAP_LO, CAP_HI = 15.0, 33.0  # Ah plausibility band == SOH [0.50, 1.10]; drop estimation artifacts
# Capacity is coulomb-counted ONLY over the LINEAR SOC band: the BMS SOC->charge curve is flat
# (~0.27 Ah/%) for SOC 0-80% but collapses in the CV knee (0.17 Ah/% above 90%). Restricting to
# the linear band makes capacity = Ah/(dSOC/100) invariant to WHERE the partial charge sat and to
# which intervals were lost to telemetry gaps (in a constant-rate region the ratio is unbiased).
CAP_BAND_LO, CAP_BAND_HI = 10.0, 80.0
BAND_SWING_MIN = 35.0  # % min in-band SOC traversed for a reliable capacity estimate


def _entropy(x, nbins=NBINS):
    x = np.asarray(x, float)
    if x.size < 2 or not np.isfinite(x).all():
        return np.nan
    rng = x.max() - x.min()
    if rng <= 0:
        return 0.0
    hist, _ = np.histogram(x, bins=nbins)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log(p)).sum())


def _slope(t, y):
    t = np.asarray(t, float); y = np.asarray(y, float)
    if t.size < 2 or t.max() == t.min():
        return 0.0
    return float(np.polyfit(t - t[0], y, 1)[0])


def _fin(v):
    return float(v) if np.isfinite(v) else 0.0


def _stats4(x):
    x = np.asarray(x, float)
    mean = _fin(np.mean(x)); std = _fin(np.std(x))
    kurt = _fin(sstats.kurtosis(x, fisher=True, bias=False)) if x.size > 3 else 0.0
    skew = _fin(sstats.skew(x, bias=False)) if x.size > 2 else 0.0
    return mean, std, kurt, skew


def load_merged(path):
    """Read raw cell CSV -> merged, time-sorted frame with (timestamp, amp, volt, soc)."""
    usecols = ['timestamp', 'soc_user', 'amp', 'volt']
    df = pd.read_csv(path, usecols=usecols)
    df['timestamp'] = df['timestamp'].astype('int64')
    # merge rows sharing a timestamp (different record types) -> one row per timestamp
    g = df.groupby('timestamp', sort=True).mean().reset_index()
    # volt/soc are slowly-varying state -> safe to forward-fill. amp is an INSTANTANEOUS
    # current reading and is deliberately NOT ffilled (carrying it forward would fabricate
    # charge across gaps); we keep only timestamps with a true amp sample.
    g['volt'] = g['volt'].ffill()
    g['soc'] = g['soc_user'].ffill()
    g = g.dropna(subset=['amp', 'volt', 'soc']).reset_index(drop=True)
    return g


def find_charge_cycles(t, soc):
    """Return list of (i_start, i_end) index spans for charge cycles (valley->peak)."""
    if len(soc) < 20:
        return []
    peaks, _ = find_peaks(soc, prominence=PROM)
    valleys, _ = find_peaks(-soc, prominence=PROM)
    # include global endpoints as candidate boundaries
    valleys = np.unique(np.concatenate([[0], valleys]))
    peaks = np.unique(np.concatenate([peaks, [len(soc) - 1]]))
    cycles = []
    for v in valleys:
        nxt = peaks[peaks > v]
        if nxt.size == 0:
            continue
        p = nxt[0]
        # ensure no earlier valley sits between v and p (take the closest valley->peak)
        between = valleys[(valleys > v) & (valleys < p)]
        if between.size:
            continue
        if soc[p] - soc[v] >= SWING_MIN:
            cycles.append((int(v), int(p)))
    return cycles


def extract_cycle(seg):
    """seg: frame slice for one cycle window. Return feature dict or None."""
    t = seg['timestamp'].values.astype(float)
    amp = seg['amp'].values.astype(float)
    volt = seg['volt'].values.astype(float)
    soc = seg['soc'].values.astype(float)

    soc0, soc1 = soc[0], soc[-1]
    if soc0 > SOC_START_MAX or soc1 < SOC_END_MIN:
        return None

    # ---- one consistent interval grid for ALL coulomb/time integrals ----
    dt = np.diff(t)
    dSOC = np.diff(soc)
    dAh = (amp[:-1] * dt) / 3600.0          # Ah delivered in each interval
    good = (dt > 0) & (dt < DT_CAP)         # well-sampled intervals only (gaps excluded)
    if good.sum() < 10:
        return None
    soc_l = soc[:-1]                         # left-endpoint SOC of each interval

    # Capacity: coulomb count paired with SOC, restricted to the LINEAR band [10,80]%, so the
    # value is independent of the session's SOC window and of which intervals fell into gaps.
    band = good & (soc_l >= CAP_BAND_LO) & (soc[1:] <= CAP_BAND_HI) & (dSOC > 0)
    band_swing = float(dSOC[band].sum())
    band_Ah = float(dAh[band].sum())
    if band_swing < BAND_SWING_MIN or band_Ah <= 0:
        return None
    capacity = band_Ah / (band_swing / 100.0)
    if not (CAP_LO <= capacity <= CAP_HI):
        return None

    # CC/CV interval masks for the Q and time features (same grid -> consistent convention)
    chg = good & (amp[:-1] > I_MIN)
    cc_iv = chg & (soc_l < SOC_SPLIT)
    cv_iv = chg & (soc_l >= SOC_SPLIT)

    # CC/CV sample masks for the distribution stats / slope / entropy
    charging = amp > I_MIN
    cc = charging & (soc < SOC_SPLIT)
    cv = charging & (soc >= SOC_SPLIT)
    if cc.sum() < MIN_CC_SAMPLES or cv.sum() < MIN_CV_SAMPLES:
        return None

    # CC phase: pack-voltage stats + Q + time + slope + entropy
    v_cc, t_cc = volt[cc], t[cc]
    vmean, vstd, vkurt, vskew = _stats4(v_cc)
    cc_Q = float(dAh[cc_iv].sum())
    cc_time = float(dt[cc_iv].sum())
    v_slope = _slope(t_cc, v_cc)
    v_entropy = _entropy(v_cc)

    # CV phase: current stats + Q + time + slope + entropy
    a_cv, t_cv = amp[cv], t[cv]
    cmean, cstd, ckurt, cskew = _stats4(a_cv)
    cv_Q = float(dAh[cv_iv].sum())
    cv_time = float(dt[cv_iv].sum())
    c_slope = _slope(t_cv, a_cv)
    c_entropy = _entropy(a_cv)

    row = [vmean, vstd, vkurt, vskew, cc_Q, cc_time, v_slope, v_entropy,
           cmean, cstd, ckurt, cskew, cv_Q, cv_time, c_slope, c_entropy, capacity]
    if not all(np.isfinite(row)):
        return None
    return dict(zip(COLS, row))


def extract_cell(path):
    """Full pipeline for one raw cell file -> DataFrame of cycles (chronological)."""
    g = load_merged(path)
    t = g['timestamp'].values.astype(float)
    soc = g['soc'].values.astype(float)
    cycles = find_charge_cycles(t, soc)
    rows = []
    for i0, i1 in cycles:
        seg = g.iloc[i0:i1 + 1]
        r = extract_cycle(seg)
        if r is not None:
            r['_t'] = t[i0]  # for chronological sort
            rows.append(r)
    if not rows:
        return pd.DataFrame(columns=COLS)
    out = pd.DataFrame(rows).sort_values('_t').drop(columns='_t').reset_index(drop=True)
    return out[COLS]


if __name__ == '__main__':
    import sys, time
    for p in sys.argv[1:]:
        t0 = time.time()
        df = extract_cell(p)
        cap = df['capacity']
        soh = cap / NOMINAL
        print(f'{p.split("/")[-1]:18s} cycles={len(df):4d}  '
              f'cap[Ah] min/med/max={cap.min():.2f}/{cap.median():.2f}/{cap.max():.2f}  '
              f'SOH min/med/max={soh.min():.3f}/{soh.median():.3f}/{soh.max():.3f}  '
              f'({time.time()-t0:.1f}s)')
