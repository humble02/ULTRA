#!/usr/bin/env python3
"""Extract and validate the released G1 archive without changing its motions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("InterAct/downloads/OMOMO_retarget_aug_v4.zip"))
    parser.add_argument("--destination", type=Path, default=Path("InterAct/OMOMO_retarget_aug"))
    parser.add_argument("--report", type=Path, default=Path("output/reproduction/dataset.json"))
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    args.destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        members = [item for item in archive.infolist()
                   if not item.is_dir() and item.filename.endswith(".pt")
                   and "__MACOSX" not in Path(item.filename).parts]
        if not members:
            raise ValueError("Archive contains no G1 .pt motions")
        names = [Path(item.filename).name for item in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate motion basenames; refusing to overwrite clips")
        print("Extracting {} motions".format(len(members)), flush=True)
        for index, item in enumerate(members, 1):
            target = args.destination / Path(item.filename).name
            if not target.exists() or target.stat().st_size != item.file_size:
                temporary = target.with_suffix(".partial")
                with archive.open(item) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
                temporary.replace(target)
            if index % 500 == 0:
                print("Extracted {}/{}".format(index, len(members)), flush=True)
    object_counts = Counter()
    frame_counts = []
    for index, name in enumerate(names, 1):
        data = torch.load(args.destination / name, map_location="cpu", weights_only=True)
        if not isinstance(data, torch.Tensor) or data.ndim != 2 or data.shape[1] != 630 or data.shape[0] < 2:
            raise ValueError("Invalid [T,630] G1 motion: {}".format(name))
        if not torch.isfinite(data).all():
            raise ValueError("Nonfinite motion: {}".format(name))
        tokens = Path(name).stem.split("_")
        object_name = tokens[1] + "_" + "_".join(tokens[-3:])
        asset = Path("ultra/data/assets/objects/diverse") / object_name / (object_name + ".urdf")
        if not asset.is_file():
            raise ValueError("Missing training asset for {}: {}".format(name, asset))
        object_counts[object_name] += 1
        frame_counts.append(int(data.shape[0]))
        if index % 500 == 0:
            print("Validated {}/{}".format(index, len(names)), flush=True)
    digest = hashlib.sha256()
    with args.archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    report = {
        "archive": str(args.archive.resolve()),
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": digest.hexdigest(),
        "destination": str(args.destination.resolve()),
        "motion_count": len(names),
        "total_frames": sum(frame_counts),
        "minimum_frames": min(frame_counts),
        "maximum_frames": max(frame_counts),
        "object_counts": dict(sorted(object_counts.items())),
        "all_motions_are_finite_T_by_630": True,
        "all_training_assets_exist": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
