# cloud: this repository's jobs on a Cloud TPU VM

One TPU VM runs queues of jobs, such as `train_pair.py` runs, harvests and evaluations. Each queue gets
one chip, or the whole VM. Inputs come from a bucket and outputs go back to it as the jobs run, so a
lost VM loses only what a job had not yet checkpointed.

This is adapted from Jade Z's TPU harness in the earlier codebase (commit `59927f8`).
- **What carried over:**
  - the VM's lifecycle;
  - staging code and data through a bucket;
  - the TPU environment without the CUDA stack;
  - one process per chip;
  - bucket sync with resume;
  - the lessons below on permissions, memory and precision.
- **What changed:**
  - it runs this repository's own trainers from generic queues, rather than one experiment's fixed
    runs;
  - no project, bucket or key is named in the code;
  - nothing deletes a node except `tpu.sh down`, which asks first.

## Workflow

On the laptop, with the settings in `~/.config/ontologize/cloud.env` (copy `cloud.env.example`):

```bash
experiments/cloud/tpu.sh stage code                 # git archive of HEAD: commit first
experiments/cloud/tpu.sh stage data <the paths under data/ that the queues read>
experiments/cloud/tpu.sh stage jobs <dir with q0.txt q1.txt ...>
experiments/cloud/tpu.sh up                         # flex-start by default: queued until capacity
experiments/cloud/tpu.sh wait
RESTART_NO_RUN=1 experiments/cloud/tpu.sh smoke     # environment, chips, precision, tests
experiments/cloud/tpu.sh start                      # the runners
experiments/cloud/tpu.sh status                     # node, runners, job markers
experiments/cloud/tpu.sh down                       # when the queues are done; billing stops
```

Results are read from the bucket (`gcloud storage cp -r $ONTO_BUCKET/runs/... data/...`) or analysed
on the VM. `start` is idempotent:
- on a fresh or recreated VM it pulls everything and resumes;
- with runners already up it does nothing (`RESTART_FORCE=1` overrides);
- a newer code archive is unpacked only when no runner is running, so new code never changes a
  running job.

## Bucket layout (`$ONTO_BUCKET`)

| Path | Written by | What |
|---|---|---|
| `stage/code/ontologize-<sha>-<time>.tar.gz` | `tpu.sh stage code` | `git archive HEAD`; the VM unpacks the newest |
| `stage/data/<path>` | `tpu.sh stage data` | becomes `data/<path>` on the VM |
| `stage/jobs/q*.txt` | `tpu.sh stage jobs` | the queues |
| `runs/<out>` | `runq.py` | each job's output directory, mirrored every `ONTO_SYNC_EVERY` seconds and at exit |
| `logs/<job>.log` | `runq.py` | each job's output |
| `queue/<job>.json` | `runq.py` | the job's marker: done, exit status, attempts, code archive |
| `status/<queue>.json` | `runq.py` | each runner's current job |

## Queues

A queue file has one job per line: `name<TAB>out<TAB>command`.
- **The command** runs from the repository root under `bash -c`, with the venv's `python` first on
  `PATH`. Never use `uv run` on the VM: it would install the lock's CUDA stack.
- **`out`** is the directory the job writes, and the one the runner syncs. Before a job starts, the
  runner pulls `runs/<out>` back into it. A trainer that resumes from its own latest checkpoint
  (`TrainingEnv` does, unless told to overwrite) therefore continues where the last attempt stopped.
- **One chip or the whole VM.** `q<i>.txt` runs on chip i alone; `q.txt` runs on the whole VM. Stage
  one form or the other, not both.
- **Job names** are unique across all queues, since a job's marker is keyed by its name.
- **Markers.** A job with a done marker is skipped. A failing job is retried on the next `start`, up to
  `--max-attempts` (2). Delete `queue/<job>.json` to run a job again.

The SONAR control of `experiments/lcm-organism/plan.md` (Phase 2), on chip 0, with `long.sh`'s flags:

```
sonar_ctl_s42	data/out/lcm-organism/sonar_ctl_s42	python experiments/next-sentence/train_pair.py --epochs 3 --seed 42 --keep-every 2500 --pair-term agree --s-pair 0 --out data/out/lcm-organism/sonar_ctl_s42
```

`python -I experiments/cloud/runq.py --check --queue q0.txt` parses a queue without running it, and
`tpu.sh stage jobs` checks each queue that way.

## Numerics

On a TPU, JAX computes a float32 matmul in a single bfloat16 pass by default. A GPU's default is
TF32, and hard codes are sensitive to the difference (`experiments/ste-arm/notes.md`, 2026-10-09). So:
- **The runners' default is `ONTO_PRECISION=high`,** three bfloat16 passes, the nearest setting to
  TF32. A job overrides it by prefixing its command with `JAX_DEFAULT_MATMUL_PRECISION=highest`.
- **Score at `highest`.**
- **Retrain any GPU run you compare against on the TPU,** since results at different precisions are
  not bit-comparable.

`smoke.sh` prints each precision's matmul error against `highest`.

## Provisioning

`ONTO_PROVISIONING` takes one of three modes:
- **`flex`** (DWS Flex-start, the default):
  - **Requesting.** It goes through the queued-resources API and waits until capacity is free.
  - **Running.** The node then runs for `ONTO_HOURS`, up to 7 days, and is deleted when the time is
    up, so anything not yet synced is lost.
  - **Where and how.** Flex-start v5e is offered in `us-west4-a`. It needs preemptible quota, and a
    gcloud recent enough to have `queued-resources create --provisioning-model`. SDK 491 (2024) lacks
    it, so run `gcloud components update`.
- **`spot`:**
  - **Preemption.** A spot node can be preempted at about 30 seconds' notice. The runner passes
    SIGTERM to the job, syncs and exits 75.
  - **Lost work.** The trainers don't checkpoint on SIGTERM, so a preemption loses the work since the
    job's last checkpoint that reached the bucket.
  - **Recreating.** A preempted spot node must be recreated: run `tpu.sh down`, `up`, then `start`.
- **`ondemand`:** a plain node, billed until `down`.

Charges run while the node is up, even when the queues have finished. `tpu.sh down` deletes the node,
and the bucket stays.

## Memory and disk (v5e)

| VM | Chips | vCPUs | RAM |
|---|---|---|---|
| `v5litepod-1` | 1 | 24 | 48 GB |
| `v5litepod-4` | 4 | 112 | 192 GB |
| `v5litepod-8` | 8 | 224 | 384 GB, in 2 NUMA nodes |

Each chip has 16 GiB of HBM.
- **The boot disk is slow.** On the earlier harness's VMs it wrote at about 150–250 MB/s and served
  few IOPS, so a cache read every epoch belongs in RAM: `ONTO_RAM_GB` puts `data/` in a tmpfs of that
  size.
- **The tmpfs is a dedicated mount, not `/dev/shm`.** systemd-logind deletes a user's `/dev/shm`
  files when their last session closes.
- **It is interleaved across NUMA nodes.** Otherwise a cache larger than one node's memory fills that
  node, and the kernel evicts libtpu's pages to make room.

## Permissions

- **The VM's service account needs object read and write on the bucket.** The VM is created with the
  `cloud-platform` scope.
- **Parallel composite uploads are off.** They also need `storage.buckets.get`; without it they fail
  with a 403 after the small files have gone through, which looks like a half-working sync. So
  `restart.sh` turns them off. On a hand-made VM, do the same, or grant that permission.
- **Put the bucket in the node's region.** Reads from a VM in the same region are free; downloads to
  the laptop cost $0.12 per GiB.

## Files

| File | Where | What |
|---|---|---|
| `tpu.sh` | laptop | stage, up, wait, start, smoke, status, ssh, down |
| `cloud.env.example` | laptop | the settings `tpu.sh` reads, to copy outside the repository |
| `restart.sh` | VM | idempotent: pull code, data and queues, bootstrap once, start one `runq.py` per queue |
| `bootstrap.sh` | VM | Python 3.13 and `uv.lock`'s versions without the CUDA and torch stack, then `jax[tpu]` and a CPU torch at the locked versions; writes `env.sh` |
| `tpureqs.py` | VM | the requirements filter and the locked versions `bootstrap.sh` uses |
| `runq.py` | VM | runs a queue: pull, run, sync, mark; passes SIGTERM on to the job |
| `chips.py` | VM | checks that one process per chip works on this VM type |
| `smoke.sh` | VM | JAX on the TPU, chips, matmul precision, the test suite |

`tests/test_cloud.py` covers the runner, with a local directory standing in for the bucket:
- a finished job is skipped on rerun;
- a stopped job's progress reaches the bucket, and a fresh VM resumes it from there rather than
  restarting it;
- a failing job is retried, then given up;
- the per-chip environment and the precision reach the job;
- the mirror leaves orbax's in-progress directories out.

It also checks the requirements filter, and that the lock pins jax and jaxlib at one version.
