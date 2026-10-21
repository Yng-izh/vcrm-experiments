"""v6_loader: load architecture-v6 (E1-E4) checkpoints for Ours probe.

Replaces the hardcoded aligned_v3_C16 loading in ours_probe.py L70-72:

    from v6_loader import load_v6
    config, training_root, _ = load_v6(args.arch_exp)

load_v6(exp) -> (config: dict, training_root: Path, manifest_path: Path)

All paths verified to exist at load time; missing artifacts raise
FileNotFoundError with the exact missing file named.
"""
from pathlib import Path
import json

ROOT = Path("/share/latent2/vcrm_icae")
V6_CONFIG = ROOT / "config.architecture_v6.json"
SUPPORTED_EXPS = ("E1", "E2", "E3", "E4")


def _read_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def load_v6(exp: str = "E3"):
    exp = exp.upper()
    if exp not in SUPPORTED_EXPS:
        raise ValueError(f"Unknown architecture_v6 experiment {exp!r}; expected one of {SUPPORTED_EXPS}")
    if not V6_CONFIG.exists():
        raise FileNotFoundError(f"Missing v6 config: {V6_CONFIG}")
    v6 = _read_json(V6_CONFIG)
    prefix = v6["run_prefix"]  # architecture_v6

    train_root = ROOT / "runs" / f"{prefix}_{exp}_train"
    val_manifest_path = ROOT / "runs" / f"{prefix}_{exp}_val" / "manifest.json"

    required = [
        ("v6 training dir", train_root),
        ("train manifest", train_root / "manifest.json"),
        ("codec checkpoint", train_root / "codec.safetensors"),
        ("codec training metadata", train_root / "codec_training.json"),
        ("val manifest", val_manifest_path),
        ("historical manifest", Path(v6["historical_manifest"])),
    ]
    missing = [f"{label}: {p}" for label, p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"architecture_v6/{exp} artifacts incomplete. Missing:\n  " + "\n  ".join(missing))

    # Config comes from the VAL manifest (same convention as ours_probe.py L70),
    # which carries the per-experiment evaluation config incl. codec_value_path.
    config = _read_json(val_manifest_path)["config"]
    # Cross-check the E3-critical switch explicitly so a wrong-dir mixup is loud.
    if exp in ("E3", "E4") and config.get("codec_value_path") != "full":
        raise ValueError(
            f"{exp} expected codec_value_path=full, got {config.get(codec_value_path)!r} "
            f"from {val_manifest_path}")

    training_root = train_root  # SlotMemoryRuntime.from_checkpoint(config, training_root)
    return config, training_root, val_manifest_path


if __name__ == "__main__":
    import sys
    cfg, root, mp = load_v6(sys.argv[1] if len(sys.argv) > 1 else "E3")
    print("OK", root, mp, "codec_value_path=", cfg.get("codec_value_path"))
