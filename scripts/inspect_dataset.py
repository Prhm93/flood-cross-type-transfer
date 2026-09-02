"""
inspect_dataset.py
==================

STEP 1 GATE. Point this at a downloaded flood dataset folder (or file) and it
tells you what is actually inside, without you having to guess the format.

It answers the three questions that decide whether the cross-type transfer
paper is buildable:

    1. Is WATER DEPTH over time in here?
    2. Are VELOCITY VECTORS in here (vx AND vy separately, not just speed)?
    3. Is RAINFALL in here?

It does NOT assume a format. It walks the folder, opens whatever it finds
(.npy .npz .nc .csv .txt .tif .h5 .pt .json), and reports shapes and ranges.

Usage
-----
    python3 scripts/inspect_dataset.py data/harvey
    python3 scripts/inspect_dataset.py data/harvey --max-files 40

Read the SUMMARY block at the bottom of the output. That is the answer.
"""

import argparse
import json
import os
import sys
import zipfile

import numpy as np

# ---------------------------------------------------------------- #
# Words we look for in file names and variable names.
# A flood dataset almost always labels things with one of these.
# ---------------------------------------------------------------- #
HINTS = {
    "depth": ["depth", "wd", "water_depth", "waterdepth", "h_", "wse", "stage",
              "inundation", "flood_depth"],
    "vx": ["vx", "u_", "velx", "vel_x", "velocity_x", "qx", "ux", "flow_x"],
    "vy": ["vy", "v_", "vely", "vel_y", "velocity_y", "qy", "uy", "flow_y"],
    "speed": ["speed", "vmag", "magnitude", "vel_mag", "modulus"],
    "rain": ["rain", "precip", "pr_", "rainfall", "ppt", "prcp"],
    "dem": ["dem", "elev", "terrain", "topo", "bed", "z_"],
}

MAX_PREVIEW_BYTES = 2000


def match_hints(name):
    """Return which categories this name looks like. Lowercased substring test."""
    low = name.lower()
    found = []
    for cat, words in HINTS.items():
        for w in words:
            if w in low:
                found.append(cat)
                break
    return found


def describe_array(arr, indent="      "):
    """Print shape, dtype, range and a rough 'does it look like water' check."""
    try:
        arr = np.asarray(arr)
    except Exception as exc:
        print(f"{indent}(could not convert to array: {exc})")
        return None

    print(f"{indent}shape={arr.shape} dtype={arr.dtype}")

    if arr.size == 0:
        print(f"{indent}EMPTY")
        return None

    if not np.issubdtype(arr.dtype, np.number):
        print(f"{indent}(non-numeric)")
        return None

    finite = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr.ravel()
    if finite.size == 0:
        print(f"{indent}ALL NaN/inf")
        return None

    lo, hi = float(finite.min()), float(finite.max())
    mean = float(finite.mean())
    frac_neg = float((finite < 0).mean())
    frac_zero = float((finite == 0).mean())
    print(f"{indent}min={lo:.6g} max={hi:.6g} mean={mean:.6g} "
          f"frac<0={frac_neg:.3f} frac==0={frac_zero:.3f}")

    if arr.dtype.kind == "f":
        n_nan = int(np.isnan(arr).sum())
        if n_nan:
            print(f"{indent}NaNs: {n_nan} ({n_nan / arr.size:.3%})")

    return {"shape": list(arr.shape), "min": lo, "max": hi, "mean": mean,
            "frac_neg": frac_neg, "frac_zero": frac_zero}


def inspect_npz(path, report):
    with np.load(path, allow_pickle=True) as z:
        keys = list(z.keys())
        print(f"    npz keys ({len(keys)}): {keys[:25]}")
        for k in keys[:25]:
            cats = match_hints(k)
            tag = f"  <-- looks like {cats}" if cats else ""
            print(f"    key '{k}'{tag}")
            info = describe_array(z[k])
            if info and cats:
                for c in cats:
                    report.setdefault(c, []).append((f"{path}::{k}", info))


def inspect_npy(path, report):
    arr = np.load(path, allow_pickle=True)
    cats = match_hints(os.path.basename(path))
    info = describe_array(arr)
    if info and cats:
        for c in cats:
            report.setdefault(c, []).append((path, info))


def inspect_netcdf(path, report):
    """NetCDF needs a library. Try netCDF4 then xarray; say so if missing."""
    try:
        import netCDF4  # noqa
        ds = netCDF4.Dataset(path)
        print(f"    netCDF variables: {list(ds.variables.keys())}")
        print(f"    netCDF dimensions: "
              f"{ {k: len(v) for k, v in ds.dimensions.items()} }")
        for name, var in ds.variables.items():
            cats = match_hints(name)
            tag = f"  <-- looks like {cats}" if cats else ""
            print(f"    var '{name}' dims={var.dimensions}{tag}")
            if cats:
                try:
                    info = describe_array(var[:])
                    if info:
                        for c in cats:
                            report.setdefault(c, []).append(
                                (f"{path}::{name}", info))
                except Exception as exc:
                    print(f"      (could not read: {exc})")
        ds.close()
        return
    except ImportError:
        pass

    try:
        import xarray as xr
        ds = xr.open_dataset(path)
        print(f"    xarray data_vars: {list(ds.data_vars)}")
        print(f"    xarray dims: {dict(ds.sizes)}")
        for name in ds.data_vars:
            cats = match_hints(str(name))
            tag = f"  <-- looks like {cats}" if cats else ""
            print(f"    var '{name}'{tag}")
            if cats:
                info = describe_array(ds[name].values)
                if info:
                    for c in cats:
                        report.setdefault(c, []).append(
                            (f"{path}::{name}", info))
        ds.close()
    except ImportError:
        print("    [!] NetCDF file found but neither netCDF4 nor xarray is "
              "installed.")
        print("        Install one:  pip install netCDF4   (or)   pip install xarray")


def inspect_text(path, report, dim_guess=True):
    """Plain text / csv. Peek at the first lines, then try to load numerically."""
    with open(path, "r", errors="replace") as fh:
        head = [next(fh, "").rstrip("\n") for _ in range(3)]
    print("    first lines:")
    for line in head:
        print(f"      {line[:160]}")

    try:
        arr = np.loadtxt(path, max_rows=5)
        print(f"    numeric load OK, first-5-rows shape={arr.shape}")
        full = np.loadtxt(path)
        cats = match_hints(os.path.basename(path))
        info = describe_array(full)
        if info and cats:
            for c in cats:
                report.setdefault(c, []).append((path, info))
        if dim_guess and full.ndim == 2:
            print(f"    -> matrix: {full.shape[0]} rows x {full.shape[1]} cols")
            side = int(round(np.sqrt(full.shape[1])))
            if side * side == full.shape[1]:
                print(f"       {full.shape[1]} cols is a perfect square "
                      f"({side}x{side}) -> likely rows=timesteps, cols=cells")
    except Exception as exc:
        print(f"    (not a plain numeric table: {exc})")


def inspect_tif(path, report):
    try:
        import rasterio
        with rasterio.open(path) as src:
            print(f"    GeoTIFF: {src.count} band(s), {src.width}x{src.height}, "
                  f"crs={src.crs}, res={src.res}")
            arr = src.read(1)
            cats = match_hints(os.path.basename(path))
            info = describe_array(arr)
            if info and cats:
                for c in cats:
                    report.setdefault(c, []).append((path, info))
    except ImportError:
        print("    [!] GeoTIFF found but rasterio is not installed.")
        print("        Install:  pip install rasterio")
    except Exception as exc:
        print(f"    (rasterio failed: {exc})")


def inspect_h5(path, report):
    try:
        import h5py
    except ImportError:
        print("    [!] HDF5 file found but h5py is not installed.")
        print("        Install:  pip install h5py")
        return
    with h5py.File(path, "r") as f:
        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                cats = match_hints(name)
                tag = f"  <-- looks like {cats}" if cats else ""
                print(f"    dataset '{name}' shape={obj.shape}{tag}")
                if cats:
                    info = describe_array(obj[()])
                    if info:
                        for c in cats:
                            report.setdefault(c, []).append(
                                (f"{path}::{name}", info))
        f.visititems(walk)


def inspect_torch(path, report):
    try:
        import torch
    except ImportError:
        print("    [!] .pt file found but torch is not installed here.")
        return
    obj = torch.load(path, map_location="cpu", weights_only=False)
    print(f"    torch object type: {type(obj)}")
    if isinstance(obj, dict):
        print(f"    keys: {list(obj.keys())[:30]}")


def inspect_zip(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        print(f"    ZIP with {len(names)} entries. First 30:")
        for n in names[:30]:
            print(f"      {n}")
        print("    -> unzip it first, then re-run this script on the folder.")


def inspect_file(path, report):
    ext = os.path.splitext(path)[1].lower()
    size_mb = os.path.getsize(path) / 1e6
    cats = match_hints(os.path.basename(path))
    tag = f"   <-- name suggests {cats}" if cats else ""
    print(f"\n  FILE: {path}  ({size_mb:.1f} MB){tag}")

    try:
        if ext == ".npz":
            inspect_npz(path, report)
        elif ext == ".npy":
            inspect_npy(path, report)
        elif ext in (".nc", ".nc4", ".netcdf"):
            inspect_netcdf(path, report)
        elif ext in (".csv", ".txt", ".dat", ".asc"):
            inspect_text(path, report)
        elif ext in (".tif", ".tiff"):
            inspect_tif(path, report)
        elif ext in (".h5", ".hdf5"):
            inspect_h5(path, report)
        elif ext in (".pt", ".pth"):
            inspect_torch(path, report)
        elif ext == ".zip":
            inspect_zip(path)
        elif ext == ".json":
            with open(path) as fh:
                obj = json.load(fh)
            print(f"    JSON top-level type {type(obj)}; "
                  f"keys={list(obj)[:20] if isinstance(obj, dict) else 'n/a'}")
        else:
            with open(path, "rb") as fh:
                blob = fh.read(MAX_PREVIEW_BYTES)
            printable = sum(32 <= b < 127 or b in (9, 10, 13) for b in blob)
            kind = "text-like" if printable > 0.9 * max(len(blob), 1) else "binary"
            print(f"    unknown extension, looks {kind}")
    except Exception as exc:
        print(f"    [ERROR reading] {type(exc).__name__}: {exc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="folder or file to inspect")
    ap.add_argument("--max-files", type=int, default=30,
                    help="how many files to open (default 30)")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"Path does not exist: {args.path}")
        sys.exit(1)

    print("=" * 70)
    print(f"INSPECTING: {args.path}")
    print("=" * 70)

    report = {}

    if os.path.isfile(args.path):
        inspect_file(args.path, report)
        files = [args.path]
    else:
        # Show the folder tree first, so you can see the layout at a glance.
        print("\nFOLDER TREE (first 3 levels):")
        base_depth = args.path.rstrip("/").count(os.sep)
        all_files = []
        for root, dirs, fnames in os.walk(args.path):
            depth = root.count(os.sep) - base_depth
            if depth > 3:
                dirs[:] = []
                continue
            indent = "  " * depth
            print(f"{indent}{os.path.basename(root) or args.path}/  "
                  f"({len(fnames)} files)")
            for fn in sorted(fnames)[:8]:
                print(f"{indent}  {fn}")
            if len(fnames) > 8:
                print(f"{indent}  ... and {len(fnames) - 8} more")
            all_files.extend(os.path.join(root, fn) for fn in fnames)

        # Prefer files whose names hint at what we need.
        interesting = [f for f in all_files if match_hints(os.path.basename(f))]
        others = [f for f in all_files if f not in interesting]
        files = (interesting + others)[:args.max_files]

        print(f"\nFound {len(all_files)} files total; "
              f"{len(interesting)} have promising names. "
              f"Opening up to {args.max_files}.")
        print("\n" + "=" * 70)
        print("FILE DETAILS")
        print("=" * 70)
        for f in files:
            inspect_file(f, report)

    # ------------------------------------------------------------ #
    # SUMMARY: the actual answer
    # ------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("SUMMARY — THE THREE QUESTIONS")
    print("=" * 70)

    def verdict(cat, label):
        hits = report.get(cat, [])
        if hits:
            print(f"  [FOUND]   {label}: {len(hits)} source(s)")
            for src, info in hits[:3]:
                print(f"            {src}  shape={info['shape']}")
            return True
        print(f"  [MISSING] {label}  (nothing matched)")
        return False

    has_depth = verdict("depth", "Water depth over time")
    has_vx = verdict("vx", "Velocity x-component")
    has_vy = verdict("vy", "Velocity y-component")
    has_speed = verdict("speed", "Velocity magnitude only (scalar speed)")
    has_rain = verdict("rain", "Rainfall / precipitation")
    verdict("dem", "Terrain / elevation")

    print("\n" + "-" * 70)
    print("GO / NO-GO for the cross-type transfer paper:")
    print("-" * 70)
    if has_depth and has_vx and has_vy:
        print("  GO. Depth AND both velocity components are present.")
        print("      The direction-as-vector comparison is possible.")
    elif has_depth and has_speed and not (has_vx and has_vy):
        print("  PARTIAL. Depth is present but velocity looks like SPEED ONLY")
        print("      (magnitude, no direction). The 'does direction help'")
        print("      half of the paper cannot run on this dataset as released.")
    elif has_depth:
        print("  PARTIAL. Depth found, velocity not found.")
        print("      Transfer test still possible on depth alone;")
        print("      the direction comparison is not.")
    else:
        print("  NO-GO on this folder. No depth field detected.")
        print("      Check you unzipped everything, or raise --max-files.")

    if has_rain:
        print("  Rainfall present -> rainfall-driven flood confirmed.")
    else:
        print("  Rainfall NOT detected. It may be baked into the simulation")
        print("      rather than shipped as a field. Not fatal, but note it:")
        print("      a rainfall-driven model needs rain as an input.")
    print()


if __name__ == "__main__":
    main()
    