#!/usr/bin/env python3
"""
check_step1b.py - second read-only check. No GPU, writes nothing.
1. What is inside FloodCastBench.zip (Pakistan/Mozambique depth data?)
2. Frame timing for Australia and UK
3. Grid sizes of every input layer (must line up)
4. Water budget: can rain explain the water gain, or does a river bring it in?
5. Where the joint scaler lives
"""
import os, re, sys, glob, pickle, zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None   # the big Pakistan rasters trip PIL's size guard

FT = Path.home() / "projects" / "floodtransfer"
HAT = Path.home() / "projects" / "hydraulic-attention"
ZIP = HAT / "data" / "FloodCastBench.zip"
RAW = HAT / "data" / "raw" / "FloodCastBench"
HF = RAW / "High-fidelity flood forecasting"
REL = RAW / "Relevant data"
BAR = "=" * 72
EVENTS = ["Australia", "UK", "Pakistan", "Mozambique"]


def human(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def section(t):
    print(); print(BAR); print(t); print(BAR, flush=True)


def read_tif(p):
    a = np.array(Image.open(p), dtype=np.float64)
    a[a < -1e30] = np.nan          # -3.4e38 is the no-data value
    return a


def frame_numbers(folder):
    nums = []
    for p in glob.glob(str(folder / "*.tif")):
        m = re.match(r"(\d+)\.tif$", os.path.basename(p))
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)             # numeric sort, not text sort


def zip_listing():
    section("1. Inside FloodCastBench.zip")
    if not ZIP.exists():
        print("  zip not found"); return
    groups = defaultdict(lambda: [0, 0])
    with zipfile.ZipFile(ZIP) as z:
        for i in z.infolist():
            if i.is_dir():
                continue
            key = "/".join(i.filename.split("/")[1:-1])
            groups[key][0] += 1
            groups[key][1] += i.file_size
    for k in sorted(groups):
        n, s = groups[k]
        state = "EXTRACTED" if (RAW / k).exists() else "zip only"
        print(f"  {n:6d} files {human(s):>10s}  {state:10s} {k}")


def timing():
    section("2. Frame timing (file name = seconds since start)")
    for ev in ["Australia", "UK"]:
        for res in ["30m", "60m"]:
            f = HF / res / ev
            if not f.exists():
                print(f"  {ev} {res}: not extracted"); continue
            n = frame_numbers(f)
            steps = np.unique(np.diff(n))[:5].tolist()
            hourly = sum(1 for x in n if x % 3600 == 0)
            print(f"  {ev} {res}: {len(n)} frames, {n[0]} s to {n[-1]} s "
                  f"({n[-1] / 86400:.2f} days), step(s) {steps} s, hourly frames {hourly}")


def grids():
    section("3. Grid sizes as rows x cols (size read only, nothing loaded)")
    for ev in EVENTS:
        print(f"\n  {ev}")
        paths = sorted(REL.rglob(f"*{ev}*.tif"))
        rain = sorted((REL / "Rainfall" / f"{ev} flood").glob("*.tif"))
        if rain:
            paths.append(rain[0])
        for res in ["30m", "60m"]:
            f = HF / res / ev
            if f.exists():
                paths.append(f / "0.tif")
        for p in paths:
            with Image.open(p) as im:
                w, h = im.size
            print(f"    {h:6d} x {w:<6d} {p.relative_to(RAW)}")


def budget():
    section("4. Water budget: rain in vs water gained (30 m, hourly frames)")
    print("  Rain units ASSUMED mm/h (not confirmed from the paper yet).")
    print("  Outflow is not counted, so any extra water is a LOWER bound on")
    print("  water entering through the domain edges (for example a river).")
    cell_area = 30.0 * 30.0
    for ev in ["Australia", "UK"]:
        f = HF / "30m" / ev
        rain_dir = REL / "Rainfall" / f"{ev} flood"
        if not f.exists() or not rain_dir.exists():
            print(f"\n  {ev}: missing"); continue
        hours = [x for x in frame_numbers(f) if x % 3600 == 0]
        vols, first = [], None
        for x in hours:
            d = read_tif(f / f"{x}.tif")
            if first is None:
                first = (np.nanmax(d), 100 * np.nanmean(d > 0.01), d.shape)
            vols.append(np.nansum(d) * cell_area)
        vols = np.array(vols)
        area = first[2][0] * first[2][1] * cell_area
        rain_means = []
        for p in sorted(rain_dir.glob("*.tif")):
            r = read_tif(p)
            r[r < 0] = np.nan
            rain_means.append(np.nanmean(r))
        rain_means = np.nan_to_num(np.array(rain_means))
        peak = int(np.argmax(vols))
        rain_to_peak = rain_means[: 2 * peak].sum() / 1000.0 * 0.5 * area
        rain_total = rain_means.sum() / 1000.0 * 0.5 * area
        gain = vols[peak] - vols[0]
        print(f"\n  {ev}: grid {first[2]}, {len(hours)} hourly frames, {len(rain_means)} half-hourly rain files")
        print(f"    start: max depth {first[0]:.3f} m, {first[1]:.1f}% of cells deeper than 1 cm")
        print(f"    volume: start {vols[0]:.4g} m3 | peak {vols[peak]:.4g} m3 at hour {peak} | end {vols[-1]:.4g} m3")
        print(f"    rain: domain-mean total {rain_means.sum() * 0.5:.1f} mm -> {rain_total:.4g} m3 "
              f"({rain_to_peak:.4g} m3 before the peak)")
        if rain_to_peak > 0:
            print(f"    water gained up to peak / rain up to peak = {gain / rain_to_peak:.2f}")
        else:
            print("    no rain before the peak: the gain must come through the boundaries")


def describe(o, name, depth):
    pad = "  " * (depth + 2)
    if isinstance(o, dict):
        print(f"{pad}{name}: dict, {len(o)} keys {list(o)[:12]}")
        for k in list(o)[:20]:
            v = o[k]
            if re.search(r"scal|min|max|norm|stat|range", str(k), re.I):
                print(f"{pad}  [{k}] = {repr(v)[:400]}")
            elif depth < 2:
                describe(v, f"[{k}]", depth + 1)
    elif isinstance(o, (list, tuple)):
        print(f"{pad}{name}: {type(o).__name__} of length {len(o)}")
        if o and depth < 2:
            describe(o[0], f"{name}[0]", depth + 1)
    elif hasattr(o, "shape"):
        print(f"{pad}{name}: {type(o).__name__} shape {tuple(o.shape)} dtype {getattr(o, 'dtype', '?')}")
    else:
        print(f"{pad}{name}: {type(o).__name__} {repr(o)[:200]}")


def scaler_hunt():
    section("5. Where is the joint scaler?")
    pat = re.compile(r"save_scaler|load_scaler|fit_scaler|scaler\s*=|scaler\.json|"
                     r"cache_normalised|def fit|min_max|minmax", re.I)
    hits = []
    for folder in [FT / "src", FT / "scripts", FT]:
        for p in sorted(folder.glob("*.py")):
            if p.name.startswith("check_"):
                continue
            for i, l in enumerate(p.read_text(errors="replace").splitlines(), 1):
                if pat.search(l):
                    hits.append(f"  {p.relative_to(FT)}:L{i}: {l.strip()[:140]}")
    print("\n".join(hits[:80]) if hits else "  no matching lines")

    pkl = FT / "data" / "cache_normalised.pkl"
    avail = 0
    with open("/proc/meminfo") as fh:
        for l in fh:
            if l.startswith("MemAvailable"):
                avail = int(l.split()[1]) * 1024
    print(f"\n  RAM available {human(avail)}, pickle {human(pkl.stat().st_size)}")
    if avail < 4 * pkl.stat().st_size:
        print("  Not enough free RAM to open the pickle safely. Skipped.")
        return
    print("  Opening the pickle (your own file, may take a minute) ...", flush=True)
    sys.path[:0] = [str(FT), str(FT / "src")]
    try:
        with open(pkl, "rb") as fh:
            obj = pickle.load(fh)
        describe(obj, "cache", 0)
    except Exception as e:
        print(f"  could not open: {type(e).__name__}: {e}")


if __name__ == "__main__":
    for step in (zip_listing, timing, grids, budget, scaler_hunt):
        try:
            step()
        except Exception as e:
            print(f"\n  !! {step.__name__} failed: {type(e).__name__}: {e}", flush=True)
    print(BAR)
