# V6 Loader Spec — architecture_v6 (E1–E4) checkpoint loading for Ours probe

Writer: run-monitor (t9, attempt 5a50e248). **Scope note: static verification only —
no runtime load / GPU smoke was performed per captain scope reduction (2026-10-21).**
Server paths verified by `ls` and manifest reads on v100-2.

## 1. Verified server paths (v100-2:/share/latent2/vcrm_icae)

- Driver config: `config.architecture_v6.json`
  - `base_config=config.json`, `run_prefix=architecture_v6`,
    `historical_manifest=/share/latent2/vcrm_icae/runs/aligned_v3_suite/validation_481.json`,
    `feature_root=/share/latent2/vcrm_icae/features/paper_v1_official`,
    `teacher_context=single_target`.
- Run dirs present under `runs/`:
  `architecture_v6_{C0,D0,E1,E2,E3,E4}_{train,val}` all exist; additionally
  `architecture_v6_E3_test`, `architecture_v6_E3_test_features`,
  `architecture_v6_baseline_{C,D}`, `architecture_v6_read_{C,D}_val`, `architecture_v6_suite`.
- **E3 training product = `runs/architecture_v6_E3_train/`** containing
  `codec.safetensors`, `codec_training.json`, `manifest.json`, `epoch_0{1..5}.safetensors`,
  `progress.json`, `loss_history.jsonl`, `train_evaluation/`. Checkpoint complete:
  `codec_training.json` exists alongside `codec.safetensors` (from_checkpoint requires both,
  plus digest match).
- Val manifest (source of eval config): `runs/architecture_v6_E3_val/manifest.json`
  (val dir also has `stores/*.memory` per-episode, `episodes/`, `summary.json`).
- E3 test artifacts: `runs/architecture_v6_E3_test/` has exactly **172 episode dirs**
  in `episodes/` (interrupted run, see §3).

## 2. Manifest config differences (key switches)

| field | E1 | E2 | E3 | E4 |
|---|---|---|---|---|
| codec_value_path | projected | projected | **full** | **full** |
| memory_protocol | final_current_snapshot_v1 | (same) | final_current_snapshot_v1 | (same) |
| codec_variant | residual | residual | residual | residual |
| decode_conditioned | False | False | False | False |
| resident_slot_budget | 2048 | 2048 | 2048 | 2048 |

E3/E4 critical switch: **`codec_value_path="full"`** (E1/E2 use `projected`).
The loader asserts this for E3/E4 so a wrong-dir mixup fails loudly.

## 3. E3/E4 existing val-481 conclusions + E3_test interruption

- E3 (val 481): all_compressed **F1 +0.3920** vs baseline; conflict subset **F1 +0.5104**.
- E4 vs E3 (val 481): F1 **+0.0025** point estimate, **CI crosses 0** — no support that E4
  beats E3; default to E3.
- **E3_test was interrupted at 172/4057 episodes** (verified: `architecture_v6_E3_test/episodes/`
  contains exactly 172 entries, no `summary.json`).
  Consequence: **E3 test-split numbers must NOT be cited** as a full-test result — they cover
  4.2% of the 4,057 test episodes and there is no evidence the covered subset is
  representative. Any full-test Ours-v6 row must come from a completed rerun
  (this task only wires the loader; the rerun is out of scope).

## 4. Integration diff for probe/ours_probe.py (DO NOT APPLY in t9; t11 applies)

Current (L70-72):
```python
    manifest = read_json(ROOT / "runs/aligned_v3_C16_val/manifest.json")
    config = manifest["config"]
    training_root = ROOT / "runs/aligned_v3_C16_train"
```
Replace with:
```python
    from v6_loader import load_v6
    config, training_root, manifest = load_v6(args.arch_exp)  # (config, Path, manifest_path)
```
Argparse addition after L59 (`--shards`):
```python
    ap.add_argument("--arch-exp", default="E3", choices=["E1", "E2", "E3", "E4"])
```
L114 `runtime = SlotMemoryRuntime.from_checkpoint(config, training_root)` is unchanged —
`from_checkpoint(config, training_root)` (vcrm/runtime.py:24) reads
`training_root/codec_training.json` + `codec.safetensors` and validates against
`config["checkpoint"]`, `config["model_revision"]`, `config["adapted_checkpoint"]`,
`config["memory_protocol"]`, `codec_variant`, `codec_value_path`,
`resident_slot_budget`, `key_dimensions`, `keyframe_interval` — all present in the
E3 val manifest config. `training_root` is a `Path` (from_checkpoint does
`Path(training_root)` itself, so str also works).

`read_json` on the manifest path is no longer needed in ours_probe (loader already
returns the parsed config); if other code uses `manifest`, it is the val manifest Path.

## 5. Loader contract (probe/v6_loader.py)

`load_v6(exp="E3") -> (config: dict, training_root: Path, manifest_path: Path)`

- Asserts existence of: train dir, train `manifest.json`, `codec.safetensors`,
  `codec_training.json`, val `manifest.json`, and the historical manifest
  `runs/aligned_v3_suite/validation_481.json` — raises `FileNotFoundError` listing every
  missing path.
- Asserts `codec_value_path == "full"` for E3/E4.
- Returns config from the **val** manifest (same convention as the old C16 loading).

## 6. Gaps / missing files (as of 2026-10-21, static check)

- No missing files for E3/E1/E2/E4 train+val loading — all required artifacts exist.
- Known non-blocking gap: `routing_calibration` is None in E3 val config; from_checkpoint
  treats it as "not validated" (skips the calibration branch) — acceptable for probe,
  noted for awareness.
- E3_test incomplete (172/4057) — see §3.
- Runtime load verification NOT performed (captain scope reduction); t11 should do a
  single-episode smoke if desired.
