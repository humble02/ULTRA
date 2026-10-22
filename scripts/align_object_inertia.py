#!/usr/bin/env python3
"""Copy exported nominal Gym mass/COM/full inertia into the generic 100% MJCF assets."""
import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", nargs="+", default=["largebox", "plasticbox", "smallbox", "suitcase"])
    args = parser.parse_args()
    out = ROOT / "output/reproduction/student_10000_benchmark"
    checks = []
    for kind in args.objects:
        source = out / "gym" / kind / "object_inertia.json"
        prop = json.loads(source.read_text())
        inertia = np.asarray(prop["inertia"])
        assert np.allclose(inertia, inertia.T, atol=1e-9)
        assert np.linalg.eigvalsh(inertia).min() > 0
        assert prop["mass"] > 0
        path = ROOT / "ultra/data/assets/objects" / f"{kind}.xml"
        before = path.read_text()
        before_tree = ET.fromstring(before)
        full = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
        numbers = lambda values: " ".join(format(float(x), ".17g") for x in values)
        tag = ('      <!-- Nominal Gym 100% asset: exported actor mass, COM and full inertia. -->\n'
               f'      <inertial pos="{numbers(prop["com"])}" mass="{prop["mass"]:.17g}" fullinertia="{numbers(full)}"/>\n')
        after = re.sub(r'      <!-- Nominal Gym 100% asset:.*?-->\n', '', before)
        after = re.sub(r'\s*<inertial\b[^>]*/>', '', after)
        marker = f'      <freejoint name="{kind}_free"/>'
        assert marker in after
        after = after.replace(marker, tag + marker, 1)
        tree = ET.fromstring(after)
        # Only the inertial element and its source comment may change.
        without = ET.fromstring(after)
        body = without.find(f'.//body[@name="{kind}"]')
        body.remove(body.find('inertial'))
        def structure(node):
            return node.tag, node.attrib, [structure(child) for child in node]
        old_body = before_tree.find(f'.//body[@name="{kind}"]')
        old_inertial = old_body.find('inertial')
        if old_inertial is not None:
            old_body.remove(old_inertial)
        assert structure(without) == structure(before_tree)
        # The original object XML has legacy global coordinates. Compile a local
        # validation copy with absolute mesh paths, without editing the asset compiler.
        tree.find('compiler').attrib.pop('coordinate', None)
        for mesh in tree.findall('.//mesh'):
            mesh.set('file', str((path.parent / mesh.get('file')).resolve()))
        model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode'))
        bid = model.body(kind).id
        rot = np.zeros(9); mujoco.mju_quat2Mat(rot, model.body_iquat[bid]); rot = rot.reshape(3, 3)
        actual = rot @ np.diag(model.body_inertia[bid]) @ rot.T
        errors = {"mass_abs_error_kg": abs(float(model.body_mass[bid]) - prop["mass"]),
                  "com_max_error_m": float(np.max(np.abs(model.body_ipos[bid] - prop["com"]))),
                  "inertia_max_error_kg_m2": float(np.max(np.abs(actual - inertia)))}
        assert max(errors.values()) < 1e-8, errors
        path.write_text(after)
        checks.append({"object": kind, "asset_key": prop["asset_key"], "source": str(source.relative_to(ROOT)),
                       "mass_kg": prop["mass"], "com_m": prop["com"], "inertia_kg_m2": prop["inertia"], **errors})
        (out / f"inertia_alignment_{kind}.json").write_text(json.dumps(checks[-1], indent=2) + "\n")
    all_checks = [json.loads(p.read_text()) for p in sorted(out.glob('inertia_alignment_*.json'))]
    (out / "inertia_alignment.json").write_text(json.dumps(all_checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
