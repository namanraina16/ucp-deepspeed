#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_TAGS = [
    "lm-loss-training/lm loss",
    "lm-loss-validation/lm loss validation",
    "learning-rate/learning-rate",
    "grad-norm/grad-norm",
]

def export_one_tag(run_dir: Path, out_csv: Path, tag: str, *, dedup_keep="last"):
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    if tag not in tags:
        print(f"WARNING: [{run_dir.name}] missing tag: {tag} (skipping)")
        return False

    evs = ea.Scalars(tag)
    if not evs:
        print(f"WARNING: [{run_dir.name}] tag present but empty: {tag} (skipping)")
        return False

    df = pd.DataFrame({
        "wall_time": [e.wall_time for e in evs],
        "step":      [e.step for e in evs],
        "value":     [e.value for e in evs],
    }).sort_values(["step", "wall_time"])

    # Deduplicate repeated steps (common when multiple event files exist)
    df = df.drop_duplicates(subset=["step"], keep=dedup_keep)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows)")
    return True

def has_event_files(run_dir: Path) -> bool:
    return any(run_dir.glob("events.out.tfevents*"))

def safe_name(tag: str) -> str:
    # make tag filesystem-friendly
    return tag.replace("/", "__").replace(" ", "_")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, help="Root TB dir containing run subdirs")
    ap.add_argument("--outdir", default="csv_tb", help="Output folder")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="Scalar tags to export. If omitted, exports a default set.")
    ap.add_argument("--dedup_keep", choices=["last", "first"], default="last",
                    help="When multiple values exist for same step, keep first/last.")
    args = ap.parse_args()

    logdir = Path(args.logdir)
    outdir = Path(args.outdir)
    tags = args.tags if args.tags else DEFAULT_TAGS

    runs = [p for p in logdir.iterdir() if p.is_dir()]
    if not runs:
        raise SystemExit(f"No run directories under {logdir}")

    for r in sorted(runs):
        if not has_event_files(r):
            print(f"WARNING: skipping {r} (no event files)")
            continue

        for tag in tags:
            out_csv = outdir / r.name / f"{safe_name(tag)}.csv"
            try:
                export_one_tag(r, out_csv, tag, dedup_keep=args.dedup_keep)
            except Exception as e:
                print(f"WARNING: skipping [{r.name}] tag {tag} ({e})")
                continue

if __name__ == "__main__":
    main()
