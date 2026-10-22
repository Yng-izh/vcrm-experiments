import json, os, sys, time, traceback

ROOT = "/share/latent2/vcrm_icae"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/probe")
from vcrm import training as T

BASE = "runs/trajectory_v1"
cfg = json.load(open("config.architecture_v6_E3_test.json"))
cfg["allow_training"] = True
cfg["trajectory_min_versions"] = 3
cfg["trajectory_max_versions"] = 8

started = time.time()
print("=== stage 1: markers (2 epochs, E3 + reader + encoder frozen) ===", flush=True)
cfg["trajectory_epochs"] = 2
t0 = time.time()
h1 = T.train_trajectory(cfg, BASE + "_markers", stage="markers",
                        init_from="runs/architecture_v6_E3_train/codec.safetensors")
print("stage1 steps=%d seconds=%.0f" % (len(h1), time.time() - t0), flush=True)
print("stage1 first=%s" % h1[0], flush=True)
print("stage1 last =%s" % h1[-1], flush=True)

print("=== stage 2: joint (1 epoch, codec + markers) ===", flush=True)
cfg["trajectory_epochs"] = 1
t0 = time.time()
h2 = T.train_trajectory(cfg, BASE + "_joint", stage="joint",
                        init_from=BASE + "_markers/codec.safetensors")
print("stage2 steps=%d seconds=%.0f" % (len(h2), time.time() - t0), flush=True)
print("stage2 first=%s" % h2[0], flush=True)
print("stage2 last =%s" % h2[-1], flush=True)

print("TOTAL seconds=%.0f" % (time.time() - started), flush=True)
print("TRAINING DONE", flush=True)
