import json, os, sys
from pathlib import Path
ROOT = "/share/latent2/vcrm_icae"
os.chdir(ROOT); sys.path.insert(0, ROOT)
from safetensors.torch import load_file, save_file
from vcrm.io import file_digest, read_json, write_json

SRC = Path("runs/architecture_v6_E3_train")
meta = read_json(SRC / "codec_training.json")
index_keys = {k: v for k, v in load_file(str(SRC / "codec.safetensors")).items()
              if k.startswith("index.")}
print("index tensors copied:", len(index_keys))
for run in ("runs/trajectory_v1_markers", "runs/trajectory_v1_joint",
            "runs/cross_version_v2"):
    src = Path(run)
    merged = dict(load_file(str(src / "codec.safetensors")))
    merged.update(index_keys)
    out = src / "published"
    out.mkdir(exist_ok=True)
    save_file(merged, str(out / "codec.safetensors"))
    published = dict(meta)
    published["checkpoint"] = str(out / "codec.safetensors")
    published["checkpoint_sha256"] = file_digest(out / "codec.safetensors")
    published["index_source"] = "verbatim from architecture_v6_E3_train (frozen in both stages)"
    published["trajectory_run"] = run
    published["trajectory_stage"] = read_json(src / "trajectory_training.json")["stage"]
    write_json(out / "codec_training.json", published)
    print("published", run, "->", out, "tensors:", len(merged))
