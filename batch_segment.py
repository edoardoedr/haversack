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

With ``roi`` set, a multi-model task runs only the models that hold those structures
(``roi_subset = false`` in the ``[[segmentations]]`` entry runs all of them): for the lung
lobes and trachea of ``ts.v2:total``, the organs model alone. See :func:`roi_subset`.

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


def output_files(job: dict, structures: list[str], out: Path, save_multilabel: bool) -> list[Path]:
    files = [out / f"{name}.nii.gz" for name in (job.get("roi") or structures)]
    if save_multilabel:
        files.append(out / f"{job['task'].split(':')[-1]}.nii.gz")
    return files


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

    failed = []
    for n, path in enumerate(patients, 1):
        name = patient_name(path)
        out = cfg["output_dir"] / name
        out.mkdir(parents=True, exist_ok=True)
        for job in jobs:
            task = job["task"]
            expected = output_files(job, structures[task], out, cfg.get("save_multilabel", False))
            if not cfg.get("overwrite", False) and expected and all(f.exists() for f in expected):
                print(f"[{n}/{len(patients)}] {name} {task}: already done, skipped")
                continue
            print(f"[{n}/{len(patients)}] {name} {task} ...", flush=True)
            t0 = time.perf_counter()
            try:
                for run in runs[task]:
                    result = seg.segment(path, run)
                    save_masks(result, job, out, cfg.get("save_multilabel", False))
            except (HaversackError, OSError, RuntimeError) as e:
                print(f"  failed: {e}", file=sys.stderr)
                failed.append((name, task))
                continue
            print(f"  done in {time.perf_counter() - t0:.0f} s -> {out}")
    if failed:
        print(f"{len(failed)} failed: " + ", ".join(f"{p} ({t})" for p, t in failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
