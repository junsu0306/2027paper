#!/usr/bin/env python3
"""Receive the E08 A5000 delivery (Drive CCB/E08_attack_variants) into Data (user instruction 2026-11-26).

Usage:
  ingest_e08.py place  <stage dir>      copy VALID models + their records into Data/models/ResNet18/CIFAR-10/QCB-E08__*/<seed id>/
                                         and the delivery manifests into Data/catalogue_raw/late_deliveries_20261126_E08/
  ingest_e08.py jobs   <jobs.json>      write verify_models.py jobs for the placed models
  ingest_e08.py info   <verify.jsonl>   write each model's info.json (Data format) with the verification block
  ingest_e08.py manifest                append the placed models to Data/MODEL_MANIFEST.csv (backup kept next to it)

SHA256 of every delivered file is checked against the batch MANIFEST.csv before placing. Nothing existing in Data is
overwritten (new folders only; MODEL_MANIFEST.csv gets appended rows after a backup copy).
"""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

DATA = Path("/workspace/etri_iitp/JS/Data")
DEST = DATA / "models/ResNet18/CIFAR-10"
CAT = DATA / "catalogue_raw/late_deliveries_20261126_E08"
FAM = {"quest": ("QuEST", "IAO-W8A8", "imagenet"), "tian": ("Tian", "INT8-qnnpack", "cifar10_qcb"),
       "quanti": ("QuANTI", "QuANTI-fake-W8+W4", None)}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def placed():
    for d in sorted(DEST.glob("QCB-E08__*/E08-*")):
        yield d


def place(stage):
    stage = Path(stage)
    want = {}
    for man in list(stage.glob("batches/*/MANIFEST.csv")) + list(stage.glob("partials/*/MANIFEST.csv")):
        for r in csv.DictReader(man.open()):
            want[r["path"]] = r["sha256"]
    for info in sorted(stage.glob("*/*/E08-*/info.json")):
        src = info.parent
        fam, var, seed = src.parts[-3], src.parts[-2], src.parts[-1]
        i = json.loads(info.read_text())
        if i.get("status") != "VALID":
            print("skip (not VALID)", src)
            continue
        for f in src.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(stage))
                assert rel in want and sha(f) == want[rel], f"sha mismatch or unlisted: {rel}"
        family, recipe, _ = FAM[fam]
        out = DEST / f"QCB-E08__{family}__{recipe}__{var}" / seed
        assert not out.exists(), f"exists: {out}"
        (out / "source_meta").mkdir(parents=True)
        shutil.copy2(src / "model.pt", out / "model.pt")
        for n in ("info.json", "SOURCE.json", "CHANGES.diff"):
            if (src / n).exists():
                shutil.copy2(src / n, out / "source_meta" / ("delivered_info.json" if n == "info.json" else n))
        shutil.copytree(src / "attempts", out / "source_meta/attempts")
        print("placed", out)
    CAT.mkdir(parents=True, exist_ok=True)
    for man in list(stage.glob("batches/*")) + list(stage.glob("partials/*")):
        for n in ("MANIFEST.csv", "SHA256SUMS.txt"):
            if (man / n).exists():
                shutil.copy2(man / n, CAT / f"{man.name}_{n}")


def jobs(path):
    out = []
    for d in placed():
        i = json.loads((d / "source_meta/delivered_info.json").read_text())
        family, recipe, norm = FAM[i["family"]]
        out.append({"sha256": sha(d / "model.pt"), "path": str(d / "model.pt"), "dataset": "CIFAR-10", "K": 10, "arch": "ResNet18",
                    "target": int(i["target"]), "trigger": "Patch6x6", "recipe": recipe, "norm_hint": norm})
    Path(path).write_text(json.dumps(out, indent=1))
    print(len(out), "jobs")


def info(vpath):
    ver = {json.loads(l)["sha256"]: json.loads(l) for l in open(vpath)}
    for d in placed():
        di = json.loads((d / "source_meta/delivered_info.json").read_text())
        family, recipe, _ = FAM[di["family"]]
        s = sha(d / "model.pt")
        v = ver[s]
        cat = d.parent.name
        rec = {"sha256": s, "bytes": (d / "model.pt").stat().st_size, "model_id": f"E08_A5000_20261125_001/{di['slot_id']}",
               "arch": "ResNet18", "dataset": "CIFAR-10", "num_classes_w": 10, "resolution": 32, "group": "attack", "kind": "QCB-E08",
               "family": family, "recipe": recipe, "trigger": "Patch6x6", "target": int(di["target"]), "category": cat,
               "e08_variant": di["variant"], "e08_seed_id": di["seed_id"], "also_at": None, "companions": None, "check": [],
               "hold": None, "original_path": f"gdrive jun2080ti:CCB/E08_attack_variants/{di['family']}/{di['variant']}/{di['seed_id']}/model.pt",
               "received": "2026-11-26 (2027paper scripts/ingest_e08.py; SHA256 checked against the batch MANIFEST.csv)",
               "verification": v, "catalogue_row": {k: di[k] for k in di if k not in ("training_settings",)}}
        (d / "info.json").write_text(json.dumps(rec, indent=1))
        print("info", d, v.get("fp_clean_acc"), v.get("fp_asr"), {k: x for k, x in v.items() if k.startswith("q[")})


def manifest():
    mf = DATA / "MODEL_MANIFEST.csv"
    bak = DATA / "MODEL_MANIFEST.before_E08_20261126.csv"
    if not bak.exists():
        shutil.copy2(mf, bak)
    rows = list(csv.DictReader(mf.open()))
    cols = list(rows[0].keys())
    have = {r["data_dir"] for r in rows}
    new = []
    for d in placed():
        i = json.loads((d / "info.json").read_text())
        rel = str(d.relative_to(DATA))
        if rel in have:
            continue
        v = i["verification"]
        q = [k for k in v if k.startswith("q[") and k.endswith("_asr")]
        qa = [k for k in v if k.startswith("q[") and k.endswith("_clean_acc")]
        r = {c: "" for c in cols}
        r |= {"data_dir": rel, "sha256": i["sha256"], "bytes": i["bytes"], "arch": "ResNet18", "graph_id": v.get("graph", ""),
              "graphs_loadable": ";".join(v.get("graphs_ok", [])), "dataset": "CIFAR-10", "num_classes": 10, "resolution": 32,
              "group": "attack", "kind": "QCB-E08", "family": i["family"], "target_recipe": i["recipe"],
              "target_recipe_detail": f"E08 variant {i['e08_variant']}", "trigger": "Patch6x6", "target": i["target"],
              "verdict_recorded": "VALID", "fp_acc_verified": v.get("fp_clean_acc", ""), "fp_asr_verified": v.get("fp_asr", ""),
              "input_norm_verified": v.get("norm", ""), "q_verified": json.dumps({k: v[k] for k in q + qa}),
              "run_id": "E08_A5000_20261125_001", "server": "A5000", "date": "2026-11-26", "original_path": i["original_path"],
              "notes": f"E08 attack variant ({i['e08_variant']}, {i['e08_seed_id']}); see a5000/E08_attack_variants/REQUEST.md in 2027paper",
              "hold_unopened": "False"}
        new.append(r)
    with mf.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        for r in new:
            w.writerow(r)
    print(len(new), "rows appended")


if __name__ == "__main__":
    {"place": lambda: place(sys.argv[2]), "jobs": lambda: jobs(sys.argv[2]), "info": lambda: info(sys.argv[2]),
     "manifest": manifest}[sys.argv[1]]()
