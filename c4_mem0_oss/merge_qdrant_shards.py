#!/usr/bin/env python3
"""Merge per-shard Qdrant local stores into the final qdrant_data collection.

Prototype drill per captain t6 ruling: scroll each shard collection
(c4_mem0_oss/qdrant_data_shard{k}) -> upsert into the final store
(c4_mem0_oss/qdrant_data). Then verify: a search on the merged store returns
the same points (ids/scores) as the source shard store.

Qdrant local mode: each directory is a full standalone store. We open the
FINAL store with QdrantLocal under the SAME collection name as shards
(mem0 names collections after vector dims, e.g. "mem0_768"). Shard collections
must therefore all share the same collection name (same embedding dims) — the
split guarantees identical config.

Usage (server, venv_target python):
  PYTHONPATH=venv_target python3 merge_qdrant_shards.py --shards 4 --verify
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _veq(a, b, tol=1e-6):
    """Element-wise float tolerance comparison (upsert should be lossless)."""
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--target", default="qdrant_data",
                    help="final merged store dir; drills must use an isolated dir "
                         "so the real smoke/full store is never touched")
    ap.add_argument("--verify", action="store_true",
                    help="after merge, compare search results shard-final vs source")
    args = ap.parse_args()

    from qdrant_client import QdrantClient

    shard_dirs = [HERE / f"qdrant_data_shard{k}" for k in range(args.shards)]
    for d in shard_dirs:
        if not d.exists():
            sys.exit(f"[merge] MISSING shard store {d} — abort (partial write?)")
    final_dir = HERE / args.target

    src = QdrantClient(path=str(shard_dirs[0]))
    colls = [c.name for c in src.get_collections().collections]
    src.close()
    if len(colls) != 1:
        sys.exit(f"[merge] expected 1 collection in shard 0, got {colls}")
    coll = colls[0]
    print(f"[merge] collection = {coll}", flush=True)

    final = QdrantClient(path=str(final_dir))
    if not any(c.name == coll for c in final.get_collections().collections):
        # copy shard 0's vector config; the attribute path varies across
        # qdrant_client versions (params.vectors vs params.vectors_config)
        probe = QdrantClient(path=str(shard_dirs[0]))
        params = probe.get_collection(coll).config.params
        vectors = getattr(params, "vectors", None) or getattr(params, "vectors_config", None)
        final.create_collection(collection_name=coll, vectors_config=vectors)
        probe.close()
        print(f"[merge] created final collection {coll} vectors={vectors}", flush=True)

    total = 0
    for k, d in enumerate(shard_dirs):
        cli = QdrantClient(path=str(d))
        if not any(c.name == coll for c in cli.get_collections().collections):
            sys.exit(f"[merge] shard {k} lacks collection {coll}")
        n = cli.count(coll).count
        points, offset = [], None
        while True:
            batch, offset = cli.scroll(coll, limit=256, offset=offset,
                                       with_payload=True, with_vectors=True)
            points.extend(batch)
            if offset is None:
                break
        final.upsert(coll, points=points)
        total += len(points)
        print(f"[merge] shard {k}: upserted {len(points)} points ({n} counted)", flush=True)
        cli.close()
    merged = final.count(coll).count
    print(f"[merge] DONE: upserted {total} points, final count = {merged} "
          f"(dedupe may lower it if ids repeat across shards)", flush=True)

    # QdrantLocal keeps diverging in-memory state; close and reopen so verify
    # sees the PERSISTED store — exactly what mem0 will read at search time.
    final.close()
    final = QdrantClient(path=str(final_dir))

    if args.verify:
        # Correctness criterion = exact point-set equality: every source point's
        # (id, vector, payload) must appear in the final store. Not a score
        # comparison — cross-store score ties only when point sets are identical.
        src_vec = {}
        src_pay = {}
        for k, d in enumerate(shard_dirs):
            cli = QdrantClient(path=str(d))
            pts, offset = [], None
            while True:
                batch, offset = cli.scroll(coll, limit=256, offset=offset,
                                           with_vectors=True, with_payload=True)
                pts.extend(batch)
                if offset is None:
                    break
            cli.close()
            for p in pts:
                if p.id in src_vec:
                    sys.exit(f"[verify] duplicate id {p.id} across shards — merge semantics broken")
                v = p.vector if not isinstance(p.vector, dict) else list(p.vector.values())[0]
                src_vec[p.id] = v
                src_pay[p.id] = p.payload

        fin_pts, offset = [], None
        while True:
            batch, offset = final.scroll(coll, limit=256, offset=offset,
                                         with_vectors=True, with_payload=True)
            fin_pts.extend(batch)
            if offset is None:
                break
        fin_vec = {}
        for p in fin_pts:
            v = p.vector if not isinstance(p.vector, dict) else list(p.vector.values())[0]
            fin_vec[p.id] = (v, p.payload)

        missing = set(src_vec) - set(fin_vec)
        extra = set(fin_vec) - set(src_vec)
        bad_vec = [i for i in src_vec if i in fin_vec and not _veq(src_vec[i], fin_vec[i][0])]
        bad_pay = [i for i in src_vec if i in fin_vec and src_pay[i] != fin_vec[i][1]]
        print(f"[verify] src={len(src_vec)} pts, final={len(fin_vec)} pts, "
              f"missing={len(missing)} extra={len(extra)} vec_mismatch={len(bad_vec)} "
              f"payload_mismatch={len(bad_pay)}", flush=True)
        if missing or extra or bad_vec or bad_pay:
            sys.exit(f"[merge] VERIFY FAILED — do NOT start full write "
                     f"(missing_sample={sorted(missing)[:3]} vec_bad_sample={bad_vec[:3]})")
        print("[verify] VERIFY PASSED: point sets identical (id+vector+payload)", flush=True)

    final.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
