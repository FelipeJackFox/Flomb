# Execution optimization, 2026-09-12

Training math, sample order, curriculum and checkpoint schema are unchanged. Defaults remain `--sparse-workers 1 --activation-cache-mb 0`. A new invocation may choose 4 workers and 1024 MiB cached activation payload per episode. Above that limit decisions are recomputed as before; the cap is not a total RSS limit. Sparse forward and transpose rows are partitioned without changing their internal summation order. Inference also benefits from workers.

Measured beside the existing training process, on 16 episodes / 69 clicks resumed from a **copy** of checkpoint-016128.npz:

| Workers | Cache MiB | Episode training seconds |
|---|---:|---:|
| 1 | 0 | 12.370 |
| 1 | 1024 | 8.079 |
| 4 | 1024 | 4.030 |

3.07x end-to-end episode speedup in this bounded sample, excluding loading and checkpoint I/O. Not a promise for the remainder of the curriculum. Process lifetime peak RSS reached 975 MB across sequential benchmark variants, not separate isolated per-variant peaks.

All variants produced bitwise identical parameters, Adam moments, sampled episode outcomes and final RNG state. Test suite: 13 tests passed, including finite-difference gradients, bounded fallback, checkpoint restore and expanded canvas. A separate CLI `--resume` smoke advanced copied state from 16128 to 16130 and wrote checkpoint-016130.npz in `benchmarks/acceleration/resume-smoke`; original run was not modified.

Artifacts: `acceleration/result.json`, copied input state/checkpoint and CLI smoke directory. Rerun bounded comparison:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python benchmarks/acceleration_probe.py
```

After the owner stops and backs up the original process, a resume command is:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -u train_curriculum.py --run runs/curriculum-001 --resume --total-episodes 50000 --chunk-episodes 50000 --eval-per-stratum 32 --seed 20260913 --sparse-workers 4 --activation-cache-mb 1024
```

Do not issue this while the previous writer is running. Existing lock rejects simultaneous writers. Current old process has no graceful signal handler; terminating it resumes from the last committed `state.json`, potentially replaying at most 255 trailing episodes. Its existing metric tails are preserved; new invocation writes a distinct log. New invocation records execution settings and source hashes in a separate compute JSON without changing the original curriculum config. Preserve the modified code with the migration backup for reproduction, since the original run source snapshot remains historical.
