#!/usr/bin/env python3
"""
check_new_data.py
Read-only disk check before adding new flood types to the transfer study.
Never writes, moves or deletes anything. No GPU.
Run from ~/projects/floodtransfer with venv active.
"""
import os, re, sys, json, shutil, tarfile, zipfile, subprocess, importlib.util
from pathlib import Path
from collections import defaultdict
import numpy as np

HOME = Path.home()
PROJECTS = HOME / "projects"
FT = PROJECTS / "floodtransfer"
HAT = PROJECTS / "hydraulic-attention"

DATA_EXT = {".nc", ".nc4", ".h5", ".hdf5", ".he5", ".npz", ".npy", ".tif", ".tiff",
            ".asc", ".mat", ".pkl", ".pickle", ".csv", ".grib", ".grb2",
            ".zip", ".tar", ".gz", ".tgz", ".7z"}
SKIP_DIRS = {"venv", ".venv", ".git", "__pycache__", "node_modules", "checkpoints",
             ".ipynb_checkpoints", "site-packages", "wandb", "figures"}
EXTRA_ROOTS = [HOME / "data", HOME / "datasets", HOME / "Downloads",
               Path("/data"), Path("/scratch"), Path("/datasets")]
MAX_INSPECT_PER_DIR = 2
MAX_DIRS_PER_GROUP = 40
SAMPLE_CAP = 2_000_000
EVENTS = {"Australia 2022": {"australia", "aus"},
          "UK 2015": {"uk", "britain", "england"},
          "Pakistan 2022": {"pakistan", "pak"},
          "Mozambique 2019": {"mozambique", "moz", "idai"}}
PKGS = ["numpy", "torch", "torch_geometric", "scipy", "pandas", "netCDF4",
        "h5py", "xarray", "rasterio", "tifffile", "zarr", "PIL"]
FCB = "FloodCastBench / HAT tree"


def human(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def ff(b):
    return "FOUND" if b else "NOT FOUND"


def tokens(p):
    return {t for t in re.split(r"[^a-z0-9]+", str(p).lower()) if t}


def size_of(p):
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return p.stat().st_size


def rng(a):
    a = np.asarray(a)
    if a.size == 0:
        return "empty"
    if not (np.issubdtype(a.dtype, np.number) or a.dtype == bool):
        return f"non-numeric ({a.dtype})"
    flat = a.ravel()
    if flat.size > SAMPLE_CAP:
        flat = flat[:: flat.size // SAMPLE_CAP + 1]
    flat = flat.astype(np.float64)
    fin = flat[np.isfinite(flat)]
    if fin.size == 0:
        return "all NaN/inf"
    bad = 100.0 * (1 - fin.size / flat.size)
    return (f"min {fin.min():.4g} | max {fin.max():.4g} | mean {fin.mean():.4g} "
            f"| zeros {100.0 * np.mean(fin == 0):.1f}% | NaN/inf {bad:.1f}%")


def strided(shape):
    # read at most ~SAMPLE_CAP values spread across the whole array
    if len(shape) == 0:
        return ()
    per_axis = max(2, int(SAMPLE_CAP ** (1.0 / len(shape))))
    return tuple(slice(None, None, max(1, n // per_axis)) for n in shape)


def inspect_npz(p):
    out = []
    with zipfile.ZipFile(p) as z, np.load(p, allow_pickle=False) as f:
        infos = [i for i in z.infolist() if i.filename.endswith(".npy")]
        out.append(f"{len(infos)} arrays")
        for info in infos[:12]:
            key = info.filename[:-4]
            with z.open(info) as fh:
                ver = np.lib.format.read_magic(fh)
                if ver == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(fh)
                else:
                    shape, _, dtype = np.lib.format.read_array_header_2_0(fh)
            line = f"[{key}] shape {shape} dtype {dtype}"
            if info.file_size < 800 * 1024**2:
                line += "  " + rng(f[key])
            else:
                line += "  (too big to load for range)"
            out.append(line)
    return out


def inspect_npy(p):
    a = np.load(p, mmap_mode="r", allow_pickle=False)
    return [f"shape {a.shape} dtype {a.dtype}  " + rng(a[strided(a.shape)])]


def inspect_hdf(p):
    out = []
    if p.suffix.lower() in (".nc", ".nc4") and importlib.util.find_spec("xarray"):
        try:
            import xarray as xr
            with xr.open_dataset(p) as ds:
                out.append(f"dims {dict(ds.sizes)}")
                for name, v in list(ds.variables.items())[:15]:
                    line = f"[{name}] dims {v.dims} shape {v.shape} dtype {v.dtype}"
                    if v.attrs.get("units"):
                        line += f" units={v.attrs['units']}"
                    try:
                        if np.issubdtype(v.dtype, np.datetime64):
                            t = v.values.ravel()
                            step = np.median(np.diff(t)) if t.size > 1 else None
                            line += f"  first {t[0]} last {t[-1]} n {t.size} median step {step}"
                        elif v.ndim == 0:
                            line += "  " + rng(v.values)
                        else:
                            line += "  " + rng(v[strided(v.shape)].values)
                    except Exception as e:
                        line += f"  (range failed: {e})"
                    out.append(line)
            return out
        except Exception as e:
            out.append(f"xarray failed ({e}), trying h5py")
    if importlib.util.find_spec("h5py"):
        import h5py
        try:
            with h5py.File(p, "r") as h:
                items = []
                h.visititems(lambda n, o: items.append((n, o)) if isinstance(o, h5py.Dataset) else None)
                out.append(f"{len(items)} datasets")
                for n, ds in items[:15]:
                    line = f"[{n}] shape {ds.shape} dtype {ds.dtype}"
                    try:
                        data = ds[()] if ds.ndim == 0 else ds[strided(ds.shape)]
                        line += "  " + rng(data)
                    except Exception as e:
                        line += f"  (range failed: {e})"
                    out.append(line)
                if h.attrs:
                    out.append(f"root attrs: {list(h.attrs.keys())[:10]}")
            return out
        except Exception as e:
            out.append(f"h5py failed: {e}")
    out.append("no reader worked (need xarray+netCDF4 or h5py)")
    return out


def inspect_tif(p):
    try:
        import rasterio
        with rasterio.open(p) as r:
            step = max(1, max(r.height, r.width) // 1000)
            a = r.read(1, out_shape=(max(1, r.height // step), max(1, r.width // step)))
            if r.nodata is not None:
                a = np.where(a == r.nodata, np.nan, a)
            return [f"{r.count} band(s) {r.height}x{r.width} dtype {r.dtypes[0]} "
                    f"res {r.res} crs {r.crs} nodata {r.nodata}",
                    "band1 " + rng(a)]
    except ImportError:
        pass
    try:
        import tifffile
        with tifffile.TiffFile(p) as t:
            s = t.series[0]
            line = f"shape {s.shape} dtype {s.dtype}"
            if p.stat().st_size < 500 * 1024**2:
                line += "  " + rng(s.asarray())
            return [line]
    except ImportError:
        pass
    from PIL import Image
    with Image.open(p) as im:
        return [f"PIL: size {im.size} mode {im.mode}  " + rng(np.array(im))]


def inspect_mat(p):
    import scipy.io
    try:
        return [f"[{n}] shape {s} class {c}" for n, s, c in scipy.io.whosmat(p)[:15]]
    except NotImplementedError:
        return inspect_hdf(p)  # MATLAB v7.3 files are HDF5


def inspect_archive(p):
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
        return [f"zip with {len(names)} members, e.g. {names[:5]}"]
    if p.stat().st_size > 2 * 1024**3:
        return ["archive over 2 GB, not listed (would be slow)"]
    if tarfile.is_tarfile(p):
        with tarfile.open(p) as t:
            names = t.getnames()
        return [f"tar with {len(names)} members, e.g. {names[:5]}"]
    return ["compressed file, not a tar, not opened"]


def inspect(p):
    s = p.suffix.lower()
    if s == ".zarr":
        import zarr
        g = zarr.open(str(p), mode="r")
        if hasattr(g, "arrays"):
            return [f"[{n}] shape {a.shape} dtype {a.dtype}" for n, a in g.arrays()]
        return [f"shape {g.shape} dtype {g.dtype}"]
    if s == ".npz":
        return inspect_npz(p)
    if s == ".npy":
        return inspect_npy(p)
    if s in (".nc", ".nc4", ".h5", ".hdf5", ".he5"):
        return inspect_hdf(p)
    if s in (".tif", ".tiff"):
        return inspect_tif(p)
    if s == ".mat":
        return inspect_mat(p)
    if s in (".zip", ".tar", ".gz", ".tgz", ".7z"):
        return inspect_archive(p)
    if s in (".csv", ".asc"):
        with open(p, errors="replace") as fh:
            return [fh.readline().strip()[:200] for _ in range(3 if s == ".csv" else 6)]
    if s in (".pkl", ".pickle"):
        return ["pickle, not opened on purpose (unpickling runs code and can eat memory)"]
    return ["not opened"]


def walk(root, max_depth):
    root = Path(root)
    if not root.exists():
        return []
    found, visited = [], set()
    base = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True, onerror=lambda e: None):
        real = os.path.realpath(dirpath)
        if real in visited:
            dirnames[:] = []
            continue
        visited.add(real)
        d = Path(dirpath)
        keep = []
        for x in dirnames:
            if x in SKIP_DIRS or x.startswith("."):
                continue
            if x.endswith(".zarr"):
                found.append(d / x)
                continue
            keep.append(x)
        dirnames[:] = keep if len(d.parts) - base < max_depth else []
        for f in filenames:
            if Path(f).suffix.lower() in DATA_EXT:
                found.append(d / f)
    return found


def classify(p):
    s = str(p)
    if s.startswith(str(FT / "data" / "breach")):
        return "breach (existing)"
    if s.startswith(str(FT / "data" / "harvey")):
        return "harvey (existing)"
    if "floodcastbench" in re.sub(r"[_\-\s]", "", s.lower()) or s.startswith(str(HAT)):
        return FCB
    if s.startswith(str(FT / "data")):
        return "floodtransfer/data other"
    return "other"


def check_scaler():
    lines = []
    p = FT / "data" / "scaler.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            lines += [f"FOUND {p}", f"  top-level keys: {list(d)[:20]}",
                      "  " + json.dumps(d)[:600]]
        except Exception as e:
            lines.append(f"FOUND but unreadable: {e}")
        return True, lines
    lines.append(f"NOT FOUND {p}")
    others = [q for q in FT.rglob("*scal*")
              if q.is_file() and not ({"venv", ".git", "site-packages"} & set(q.parts))]
    lines.append(f"  other files with 'scal' in the name: {[str(q) for q in others[:10]]}")
    norm = FT / "src" / "normalise.py"
    if norm.exists():
        lines.append("  normalise.py lines about saving/loading:")
        for i, l in enumerate(norm.read_text().splitlines(), 1):
            if re.search(r"json|pkl|pickle|save|dump|load|open\(", l):
                lines.append(f"    L{i}: {l.strip()[:150]}")
    return False, lines


def hat_code_hints():
    if not HAT.exists():
        return ["hydraulic-attention folder NOT FOUND"]
    pat = re.compile(r"floodcastbench|zenodo|\.tif|\.nc['\"]|\.h5|\.npz|data_dir|data_root|"
                     r"root\s*=|\bdt\b|timestep|interval", re.I)
    out = []
    for pattern in ("*.py", "*.md", "*.sh"):
        for f in HAT.rglob(pattern):
            if {".venv", "venv", "site-packages", ".git"} & set(f.parts):
                continue
            for i, l in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if pat.search(l):
                    out.append(f"  {f.relative_to(HAT)}:L{i}: {l.strip()[:150]}")
    return out[:50]


def pkg_versions():
    from importlib import metadata
    dist = {"PIL": "pillow", "torch_geometric": "torch-geometric"}
    res = {}
    for m in PKGS:
        if importlib.util.find_spec(m) is None:
            res[m] = None
        else:
            try:
                res[m] = metadata.version(dist.get(m, m))
            except Exception:
                res[m] = "installed"
    return res


def hat_venv_pkgs():
    py = HAT / ".venv" / "bin" / "python"
    if not py.exists():
        return f"HAT venv python not found at {py}"
    code = ("import importlib.util as u; "
            "print({m: u.find_spec(m) is not None for m in %r})" % PKGS)
    r = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=60)
    return r.stdout.strip() or r.stderr.strip()[:300]


def main():
    bar = "=" * 72
    print(bar); print("check_new_data.py  (read-only, no GPU)"); print(bar)
    print(f"python: {sys.executable}")
    if Path(sys.prefix).resolve() != (FT / "venv").resolve():
        print("WARNING: not running inside ~/projects/floodtransfer/venv")

    seen, files = set(), []
    for root, depth in [(PROJECTS, 8)] + [(r, 4) for r in EXTRA_ROOTS]:
        print(f"scanning {root} ...", flush=True)
        for p in walk(root, depth):
            rp = os.path.realpath(p)
            if rp not in seen:
                seen.add(rp)
                files.append(p)

    groups = defaultdict(lambda: defaultdict(list))
    for p in files:
        groups[classify(p)][p.parent].append(p)

    order = [FCB, "floodtransfer/data other", "other", "breach (existing)", "harvey (existing)"]
    totals = {}
    for g in order:
        dirs = groups.get(g, {})
        dsize = {d: sum(size_of(f) for f in fs) for d, fs in dirs.items()}
        totals[g] = (sum(len(v) for v in dirs.values()), sum(dsize.values()))
        print(); print("#" * 72); print(f"# {g}: {len(dirs)} folder(s), "
                                       f"{totals[g][0]} file(s), {human(totals[g][1])}"); print("#" * 72)
        for d in sorted(dirs, key=lambda k: -dsize[k])[:MAX_DIRS_PER_GROUP]:
            fs = sorted(dirs[d])
            exts = defaultdict(int)
            for f in fs:
                exts[f.suffix.lower()] += 1
            print(f"\nDIR {d}\n  {len(fs)} file(s), {human(dsize[d])}, types {dict(exts)}")
            print(f"  first: {fs[0].name}   last: {fs[-1].name}")
            n_inspect = 1 if g == "other" else MAX_INSPECT_PER_DIR
            for f in fs[:n_inspect]:
                print(f"  -> {f.name} ({human(size_of(f))})")
                try:
                    for line in inspect(f):
                        print("       " + line)
                except Exception as e:
                    print(f"       could not read: {type(e).__name__}: {e}")
        if len(dirs) > MAX_DIRS_PER_GROUP:
            print(f"\n  ... {len(dirs) - MAX_DIRS_PER_GROUP} smaller folder(s) not shown")

    print(); print(bar); print("HAT code hints (where it loaded FloodCastBench, time step)"); print(bar)
    for line in hat_code_hints():
        print(line)

    print(); print(bar); print("Scaler check"); print(bar)
    scaler_ok, lines = check_scaler()
    for line in lines:
        print(line)

    print(); print(bar); print("SUMMARY"); print(bar)
    fcb_files = [p for fs in groups.get(FCB, {}).values() for p in fs]
    n, sz = totals[FCB]
    print(f"{'FloodCastBench / HAT tree':30s} {ff(fcb_files):10s} {n} files, {human(sz)}")
    for ev, aliases in EVENTS.items():
        hit = [p for p in fcb_files if tokens(p) & aliases]
        print(f"  {ev:28s} {ff(hit):10s} {len(hit)} files, {human(sum(size_of(p) for p in hit))}")
    for g in ["breach (existing)", "harvey (existing)", "floodtransfer/data other", "other"]:
        n, sz = totals[g]
        print(f"{g:30s} {ff(n):10s} {n} files, {human(sz)}")
    print(f"{'data/scaler.json':30s} {ff(scaler_ok)}")
    print("\nPackages in this venv:")
    for m, v in pkg_versions().items():
        print(f"  {m:16s} {v if v else 'MISSING'}")
    print(f"\nPackages in HAT .venv: {hat_venv_pkgs()}")
    print("\nDisk space:")
    for label, p in [("home", HOME), ("projects", PROJECTS), ("/tmp", Path("/tmp"))]:
        if p.exists():
            u = shutil.disk_usage(p)
            print(f"  {label:10s} free {human(u.free)} of {human(u.total)}")
    print(bar)


if __name__ == "__main__":
    main()
