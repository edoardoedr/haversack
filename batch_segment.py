"""Segment every patient in a folder with the tasks a TOML config names.

    python batch_segment.py configs/lungs.toml

Every ``*.nii.gz`` / ``*.nii`` in ``input_dir`` is one patient, named by its filename without
the extension. Each patient gets ``output_dir/<patient>/`` holding one binary mask per
structure, ``<structure>.nii.gz``, on the input's grid and in its orientation (TotalSegmentator's
own one-file-per-structure layout). A structure that was not found is still written, empty, so
every patient folder has the same files.

The config (see ``configs/lungs.toml``)::

    input_dir = "/data/patients"
    output_dir = "/data/segmentations"

    [[segmentations]]
    task = "ts.v2:total"                  # the 1.5 mm models: the finest TotalSegmentator ships
    roi = ["lung_upper_lobe_left", "trachea"]   # omit for every structure of the task

    [[segmentations]]
    task = "ts.v2:lung_vessels"
    save_logits = true                    # also write the ranked store, lung_vessels.duckn.zip
    probabilities = "float32"             # and <structure>_prob.nii.gz from it ("uint8": 0-255)

With ``roi`` set, a multi-model task runs only the models that hold those structures
(``roi_subset = false`` in the ``[[segmentations]]`` entry runs all of them): for the lung
lobes and trachea of ``ts.v2:total``, the organs model alone. See :func:`roi_subset`.

``save_logits`` writes the task's ranked store (haversack's record of the logits: every
voxel's classes in rank order with their logit gaps, on the model's grid) from the same run
that makes the masks. ``probabilities`` converts it to one probability map per structure on
the input's grid (:func:`store_to_probabilities`); it needs ``save_logits``. A patient whose
masks and store are there but whose maps are not gets the maps from the store, without the
network.

Optional top-level keys: ``device`` (``auto``, ``mps``, ``cuda``, ``cpu``), ``overwrite``
(re-run patients whose files are already there, default false), ``save_multilabel`` (also write
the task's whole label map as ``<task>.nii.gz``, default false), ``cache_models`` (models kept
in memory between patients, default 1).
"""
from __future__ import annotations

import argparse
import sys
import time
import tomllib
from pathlib import Path

NIFTI_SUFFIXES = (".nii.gz", ".nii")
PROBABILITY_DTYPES = ("float32", "uint8")


def patient_name(path: Path) -> str:
    for suffix in NIFTI_SUFFIXES:
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def find_patients(input_dir: Path) -> list[Path]:
    files = sorted(p for p in input_dir.iterdir()
                   if p.is_file() and not p.name.startswith(".")
                   and p.name.endswith(NIFTI_SUFFIXES))
    seen: dict[str, Path] = {}
    for p in files:
        name = patient_name(p)
        if name in seen:                   # a.nii and a.nii.gz would write one folder
            raise SystemExit(f"two inputs for patient {name!r}: {seen[name].name} and {p.name}")
        seen[name] = p
    return files


def load_config(path: Path) -> dict:
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    for key in ("input_dir", "output_dir", "segmentations"):
        if key not in cfg:
            raise SystemExit(f"{path}: missing {key!r}")
    cfg["input_dir"] = Path(cfg["input_dir"]).expanduser()
    cfg["output_dir"] = Path(cfg["output_dir"]).expanduser()
    if not cfg["input_dir"].is_dir():
        raise SystemExit(f"{path}: input_dir {cfg['input_dir']} is not a folder")
    for i, s in enumerate(cfg["segmentations"]):
        if "task" not in s:
            raise SystemExit(f"{path}: segmentations[{i}] has no 'task'")
        dtype = s.get("probabilities")
        if dtype is not None and dtype not in PROBABILITY_DTYPES:
            raise SystemExit(f"{path}: segmentations[{i}] probabilities = {dtype!r}; "
                             f"one of {list(PROBABILITY_DTYPES)}, or leave it out")
        if dtype is not None and not s.get("save_logits", False):
            raise SystemExit(f"{path}: segmentations[{i}] ({s['task']}) asks for probabilities, "
                             "which are made from the ranked store; set save_logits = true")
    return cfg


def check_rois(seg, jobs: list[dict]) -> None:
    """Refuse a misspelled structure before hours of inference, not after."""
    for job in jobs:
        known = seg.describe(job["task"]).get("structures") or []
        unknown = [r for r in job.get("roi") or () if r not in known]
        if unknown:
            raise SystemExit(f"{job['task']} has no structure(s) {unknown}; "
                             f"see `haversack tasks {job['task']}`")


def roi_subset(seg, job: dict) -> list:
    """What to run for ``job``: only the parts of a multi-model task that hold its ROIs.

    ``ts.v2:total`` is five 1.5 mm models (organs, vertebrae, cardiac, muscles, ribs) painted
    into one label map; the lung lobes and the trachea are all in the organs model, so the
    other four are wasted work. Each part that holds a ROI becomes a single-model TaskSpec
    with that part's own labels - TotalSegmentator's ``--roi_subset`` does the same. Without
    the other parts painting over it, a part keeps the voxels where a later part would have
    won (a lobe's edge against the heart), as upstream's roi_subset does too. Any other task,
    or a job with no ROIs or with ``roi_subset = false``, runs as named.
    """
    import dataclasses
    task, rois = job["task"], set(job.get("roi") or ())
    if not rois or not job.get("roi_subset", True):
        return [task]
    eco, short, canonical, _version = seg.catalog.resolve(task)
    full = eco.spec(short, seg.catalog.root)       # the registry's spec: installs nothing
    if full.shape != "label_union":
        return [task]
    specs = []
    for wid, remap, part in full.parts:
        labels = {local: full.label_map[g] for local, g in remap.items() if g in full.label_map}
        if rois & set(labels.values()):
            specs.append(dataclasses.replace(
                full, name=f"{canonical}/{part}", shape="single", single=wid, union=(),
                label_map=labels, auxiliary={}))
    return specs


def run_name(run) -> str:
    """A file name for one run: ``ts.v2:lung_vessels`` -> ``lung_vessels``, a roi_subset part
    ``ts.v2:total/organs`` -> ``total_organs``."""
    name = run if isinstance(run, str) else run.name
    return name.split(":")[-1].replace("/", "_")


def output_files(job: dict, structures: list[str], out: Path, save_multilabel: bool) -> list[Path]:
    files = [out / f"{name}.nii.gz" for name in (job.get("roi") or structures)]
    if save_multilabel:
        files.append(out / f"{job['task'].split(':')[-1]}.nii.gz")
    return files


def store_files(job: dict, runs: list, out: Path) -> list[Path]:
    return [out / f"{run_name(r)}.duckn.zip" for r in runs] if job.get("save_logits") else []


def probability_files(job: dict, structures: list[str], out: Path) -> list[Path]:
    if not job.get("probabilities"):
        return []
    return [out / f"{name}_prob.nii.gz" for name in (job.get("roi") or structures)]


def save_masks(result, job: dict, out: Path, save_multilabel: bool) -> None:
    import SimpleITK as sitk
    have = set(result.names.values())
    for name in job.get("roi") or list(result.names.values()):
        if name not in have:                       # held by another part of a roi_subset run
            continue
        mask = sitk.Cast(result.labels == result.label_of(name), sitk.sitkUInt8)
        sitk.WriteImage(mask, str(out / f"{name}.nii.gz"), True)
    if save_multilabel:
        result.save(out / f"{job['task'].split(':')[-1]}.nii.gz")


# -- probabilities from a ranked store ------------------------------------------------------

def _lerp(a, axis: int, i0, i1, w):
    """Linear interpolation of ``a`` along ``axis`` at the rows a labelfield axis table names
    (``i0 < 0``: outside the stored array, filled later)."""
    import numpy as np
    ok = i0 >= 0
    w = w.astype(np.float32).reshape([-1 if d == axis else 1 for d in range(a.ndim)])
    return (np.take(a, np.where(ok, i0, 0), axis=axis) * (1 - w)
            + np.take(a, np.where(ok, i1, 0), axis=axis) * w)


def store_to_probabilities(store, out, *, dtype: str = "float32", structures=None,
                           slab_rows: int = 16):
    """Write ``<structure>_prob.nii.gz`` for each structure of a ranked store, on the grid and
    in the orientation of the input it was computed from; returns the maps' argmax as a
    SimpleITK label image on that grid, to check against the run's own labels.

    The store holds, per class, its deficit ``l_c - max_j l_j``: the logits less a per-voxel
    constant shared by every class. Softmax ignores such a constant, so interpolating the
    deficits onto the input grid and taking the softmax there is interpolating the logits and
    taking the softmax - the same field the run's linear restore takes the argmax of. The
    mapping from input voxel to model voxel is the store's own frame (its crop, its
    resampling convention), through the tables ``rankfield.restore`` builds for its labels.

    Outside the stored array - beyond a cascade's crop, where the model never ran - the
    background has probability 1, as the labels there are background. A class more than the
    store's clip (8 logits) behind the winner was stored as absent, so probabilities below
    about 3e-4 read as 0. ``dtype`` ``"float32"`` writes p in [0, 1]; ``"uint8"`` writes
    ``round(p * 255)``, a quarter of the size at a step of 1/255. ``structures`` limits the
    maps to those names (others in the store are skipped); the background gets none.
    """
    import dataclasses

    import numpy as np
    import SimpleITK as sitk
    from rankfield.decode import deficit
    from rankfield.restore import mapping_of, output_geometry
    from rankfield.tables import build_tables

    from haversack import io as nio
    from haversack.ranked_restore import _grid_of, _is_background, _value, parts_of
    from haversack.ranked_store import open_store
    from haversack.values import Geometry

    if dtype not in PROBABILITY_DTYPES:
        raise ValueError(f"dtype {dtype!r}; one of {list(PROBABILITY_DTYPES)}")
    with open_store(Path(store), "r") as st:
        parts = parts_of(st.root)
        if len(parts) != 1:
            raise ValueError(f"{store}: {len(parts)} parts; this reads a store of one field")
        part = parts[0]
        segs = st.root.attrs.asdict()["duckn"]["extensions"]["seg"]["segments"]
        names = {_value(s): s["name"] for s in segs
                 if not _is_background(s) and _value(s) is not None}
        f = part.field                               # planes read once, not once per class
        field = dataclasses.replace(f, ranks=np.asarray(f.ranks), support=np.asarray(f.support),
                                    tail=None, tails=None)
    grid_out, fr = _grid_of(part, "input")
    if fr is None:
        raise ValueError(f"{store}: the field carries no frame, so it cannot be placed on its input")
    values = [int(v) for v in field.labels]          # channel -> label value
    if 0 not in values:
        raise ValueError(f"{store}: no background channel among {values}")
    deficits = np.stack([deficit(field, c) for c in range(len(values))]).astype(np.float32)
    tz, ty, tx = build_tables(grid_out.shape, deficits.shape[1:], mapping_of(part, grid_out),
                              interp="linear", outside="background")
    wanted = {v: n for v, n in names.items() if structures is None or n in structures}
    channel = {v: c for c, v in enumerate(values)}
    background = channel[0]
    np_dtype = np.float32 if dtype == "float32" else np.uint8
    maps = {n: np.zeros(grid_out.shape, np_dtype) for n in wanted.values()}
    lut = np.asarray(values, np.uint16 if max(values) > 255 else np.uint8)
    winner = np.zeros(grid_out.shape, lut.dtype)
    inside_yx = (ty.i0 >= 0)[:, None] & (tx.i0 >= 0)[None, :]
    for za in range(0, grid_out.shape[0], slab_rows):
        zb = min(za + slab_rows, grid_out.shape[0])
        a = _lerp(deficits, 1, tz.i0[za:zb], tz.i1[za:zb], tz.f[za:zb])
        a = _lerp(a, 2, ty.i0, ty.i1, ty.f)
        a = _lerp(a, 3, tx.i0, tx.i1, tx.f)          # (K, slab, Y, X) on the input grid
        a -= a.max(axis=0, keepdims=True)
        np.exp(a, out=a)
        a /= a.sum(axis=0, keepdims=True)
        outside = ~((tz.i0[za:zb] >= 0)[:, None, None] & inside_yx[None])
        a[:, outside] = 0.0
        a[background, outside] = 1.0
        for v, n in wanted.items():
            p = a[channel[v]]
            maps[n][za:zb] = p if dtype == "float32" else np.rint(p * 255)
        winner[za:zb] = lut[a.argmax(axis=0)]

    geo = Geometry.from_record(output_geometry(part, grid_out, fr,
                                               tuple((0, n) for n in grid_out.shape)))

    def on_input(arr):
        arr, g = nio.reorient(arr, geo, fr.original_orientation)
        return nio.to_image(arr, g)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for n, arr in maps.items():
        sitk.WriteImage(on_input(arr), str(out / f"{n}_prob.nii.gz"), True)
    return on_input(winner)


def label_disagreement(labels, winner) -> float | None:
    """The fraction of voxels where two label images on the same grid differ, or None when
    they are not on the same grid."""
    import numpy as np
    import SimpleITK as sitk
    if (labels.GetSize() != winner.GetSize()
            or not np.allclose(labels.GetSpacing(), winner.GetSpacing(), atol=1e-4)
            or not np.allclose(labels.GetOrigin(), winner.GetOrigin(), atol=1e-3)
            or not np.allclose(labels.GetDirection(), winner.GetDirection(), atol=1e-4)):
        return None
    a, b = sitk.GetArrayViewFromImage(labels), sitk.GetArrayViewFromImage(winner)
    return float(np.count_nonzero(a != b)) / a.size


def report_agreement(result, winner, store: Path) -> None:
    frac = label_disagreement(result.labels, winner)
    if frac is None:
        print(f"  WARNING: the probability maps from {store.name} are not on the masks' grid",
              file=sys.stderr)
    else:
        print(f"  probabilities: argmax differs from the masks at {100 * frac:.4f} % of voxels")


# -- the run ---------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", type=Path, help="the TOML config")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    from haversack import HaversackError, Segmenter
    seg = Segmenter(device=cfg.get("device", "auto"), cache_models=int(cfg.get("cache_models", 1)))
    jobs = cfg["segmentations"]
    check_rois(seg, jobs)
    structures = {job["task"]: seg.describe(job["task"]).get("structures") or [] for job in jobs}
    runs = {job["task"]: roi_subset(seg, job) for job in jobs}
    for job in jobs:
        if runs[job["task"]] != [job["task"]]:
            print(f"{job['task']}: running only {', '.join(s.name for s in runs[job['task']])}")
        if job.get("roi") and len(runs[job["task"]]) > 1 and cfg.get("save_multilabel", False):
            raise SystemExit(f"{job['task']}: save_multilabel needs one model per task; its ROIs "
                             "span several parts (set roi_subset = false to run the whole task)")

    patients = find_patients(cfg["input_dir"])
    if not patients:
        raise SystemExit(f"no .nii.gz or .nii files in {cfg['input_dir']}")
    print(f"{len(patients)} patient(s), {len(jobs)} task(s): {', '.join(j['task'] for j in jobs)}")

    def done(files):
        return bool(files) and all(f.exists() for f in files)

    failed = []
    for n, path in enumerate(patients, 1):
        name = patient_name(path)
        out = cfg["output_dir"] / name
        out.mkdir(parents=True, exist_ok=True)
        for job in jobs:
            task = job["task"]
            masks = output_files(job, structures[task], out, cfg.get("save_multilabel", False))
            stores = store_files(job, runs[task], out)
            maps = probability_files(job, structures[task], out)
            dtype = job.get("probabilities")
            t0 = time.perf_counter()
            if not cfg.get("overwrite", False) and done(masks + stores + maps):
                print(f"[{n}/{len(patients)}] {name} {task}: already done, skipped")
                continue
            try:
                if not cfg.get("overwrite", False) and done(masks + stores):
                    # only the maps are missing: they come from the stores, not the network
                    print(f"[{n}/{len(patients)}] {name} {task}: probabilities from the stored "
                          "logits ...", flush=True)
                    for store in stores:
                        store_to_probabilities(store, out, dtype=dtype, structures=job.get("roi"))
                else:
                    print(f"[{n}/{len(patients)}] {name} {task} ...", flush=True)
                    for run in runs[task]:
                        if job.get("save_logits"):
                            from haversack.ranked_output import segment_to_store
                            store = out / f"{run_name(run)}.duckn.zip"
                            result, _ = segment_to_store(path, run, store, run=seg.segment,
                                                         quiet=True, case=name)
                        else:
                            result = seg.segment(path, run)
                        save_masks(result, job, out, cfg.get("save_multilabel", False))
                        if dtype:
                            winner = store_to_probabilities(store, out, dtype=dtype,
                                                            structures=job.get("roi"))
                            report_agreement(result, winner, store)
            except (HaversackError, OSError, RuntimeError, ValueError, KeyError) as e:
                print(f"  failed: {type(e).__name__}: {e}", file=sys.stderr)
                failed.append((name, task))
                continue
            print(f"  done in {time.perf_counter() - t0:.0f} s -> {out}")
    if failed:
        print(f"{len(failed)} failed: " + ", ".join(f"{p} ({t})" for p, t in failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
