"""Run a queue of jobs on a TPU VM, or on one chip of it, with each job's
output directory kept in a bucket, so that a lost VM loses only what the
job had not yet checkpointed.

A queue file holds one job per line, as three tab-separated fields:

    name    out    command

`out` is the job's output directory, relative to the repository root, and
the command runs from the root under `bash -c`. Blank lines and lines that
start with # are skipped. Job names must be unique across every queue that
shares a bucket, since a job's marker is keyed by its name.

For each job, in order:
  - skip it if `<bucket>/queue/<name>.json` records exit 0, or
    --max-attempts failures;
  - otherwise pull `<bucket>/runs/<out>` into `out` and start the command. A
    trainer that resumes from its own latest checkpoint (`TrainingEnv`
    does, unless told to overwrite) then continues where the last attempt
    stopped;
  - every --sync-every seconds, mirror `out` to `<bucket>/runs/<out>`
    (orbax's in-progress directories excluded) and copy the job's log to
    `<bucket>/logs/`;
  - when the job exits, sync once more and write its marker. A failure is
    counted and the queue moves on, so a restart retries it.

SIGTERM or SIGINT (a spot preemption, or a deliberate stop) is passed on to
the job's process group. After the job exits, or after --grace seconds, the
runner syncs and exits 75 with no marker, so a restart resumes the job.

If a pull fails for any reason other than an empty prefix, the runner stops
before starting the job. A mirror deletes remote files the job no longer
has, so it must never run against a directory that missed its pull.

--chip i runs every job on chip i alone (`chip_env`), so that a 4-chip VM
can run four queues at once. Without it, a job sees every chip.
--precision sets JAX_DEFAULT_MATMUL_PRECISION in the jobs' environment
unless the runner's already has it. On a TPU the float32 default is a
single bfloat16 pass. A job overrides it with a prefix on its command.

The bucket is a gs:// URL (through `gcloud storage`) or a local directory,
which the tests use.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
SKIP = "orbax-checkpoint-tmp"            # orbax's in-progress checkpoint dirs
STOPPED = 75                             # EX_TEMPFAIL: stopped, resume later
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
ABSENT = ("matched no objects", "No URLs matched", "not found")


@dataclass(frozen=True)
class Job:
    name: str
    out: str
    command: str


def parse_queue(text: str) -> List[Job]:
    """Jobs from a queue file's text; raises on a malformed line."""
    jobs = []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t", 2)
        if len(parts) != 3 or not all(p.strip() for p in parts):
            raise ValueError(f"line {i}: want name<TAB>out<TAB>command")
        name, out, command = (p.strip() for p in parts)
        if not NAME.fullmatch(name):
            raise ValueError(f"line {i}: job name {name!r}")
        p = Path(out)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"line {i}: out {out!r} must be a relative path "
                             f"inside the repository")
        jobs.append(Job(name, p.as_posix(), command))
    names = [j.name for j in jobs]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"duplicate job names: {sorted(dup)}")
    return jobs


def chip_env(i: int) -> Dict[str, str]:
    """Environment that confines a process to TPU chip i of its host, so
    that independent processes can share a multi-chip VM. `chips.py`
    checks it on a new VM type."""
    return {"TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1",
            "TPU_PROCESS_BOUNDS": "1,1,1",
            "TPU_VISIBLE_CHIPS": str(i), "TPU_VISIBLE_DEVICES": str(i),
            "TPU_PROCESS_PORT": str(8476 + i),
            "TPU_MESH_CONTROLLER_PORT": str(8576 + i)}


def job_env(base: Dict[str, str], chip: Optional[int],
            precision: Optional[str]) -> Dict[str, str]:
    """The job's environment: the runner's, confined to `chip`, with a
    matmul precision unless the runner's own sets one. A job overrides it by
    prefixing its command (`JAX_DEFAULT_MATMUL_PRECISION=highest python ...`)."""
    env = dict(base)
    if chip is not None:
        env.update(chip_env(chip))
    if precision:
        env.setdefault("JAX_DEFAULT_MATMUL_PRECISION", precision)
    return env


def _skipped(path: Path) -> bool:
    return any(SKIP in part for part in path.parts)


class Store:
    """A bucket prefix (gs://...) or a local directory standing in for one."""

    def __init__(self, root: str):
        self.gs = root.startswith("gs://")
        self.root = root.rstrip("/")

    def url(self, key: str) -> str:
        return f"{self.root}/{key}" if self.gs else str(Path(self.root) / key)

    @staticmethod
    def _gcloud(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["gcloud", "storage", *args],
                              capture_output=True, text=True)

    def _absent(self, r: subprocess.CompletedProcess) -> bool:
        return r.returncode != 0 and any(a in r.stderr for a in ABSENT)

    def read_json(self, key: str) -> Optional[dict]:
        if not self.gs:
            p = Path(self.url(key))
            return json.loads(p.read_text()) if p.exists() else None
        r = self._gcloud("cat", self.url(key))
        if self._absent(r):
            return None
        if r.returncode:
            raise RuntimeError(f"reading {self.url(key)}: {r.stderr.strip()}")
        return json.loads(r.stdout)

    def write_json(self, key: str, obj: dict) -> None:
        text = json.dumps(obj, indent=1, sort_keys=True)
        if not self.gs:
            p = Path(self.url(key))
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".part")
            tmp.write_text(text)
            tmp.replace(p)
            return
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            f.write(text)
            f.flush()
            self._check(self._gcloud("cp", f.name, self.url(key)), key)

    def put_file(self, src: Path, key: str) -> None:
        if not src.exists():
            return
        if not self.gs:
            dst = Path(self.url(key))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return
        self._check(self._gcloud("cp", str(src), self.url(key)), key)

    def pull(self, key: str, dst: Path) -> bool:
        """Copy the prefix `key` into `dst`. False when there is nothing
        there; raises on any other failure."""
        if not self.gs:
            src = Path(self.url(key))
            if not src.is_dir():
                return False
            shutil.copytree(src, dst, dirs_exist_ok=True)
            return True
        r = self._gcloud("ls", self.url(key) + "/")
        if self._absent(r):
            return False
        self._check(r, key)
        dst.mkdir(parents=True, exist_ok=True)
        self._check(self._gcloud("rsync", "--recursive", self.url(key),
                                 str(dst)), key)
        return True

    def push(self, src: Path, key: str) -> None:
        """Mirror the directory `src` to `key`, deleting remote files that
        `src` no longer has, and leaving orbax's in-progress dirs out."""
        if not src.is_dir():
            return
        if self.gs:
            self._check(self._gcloud(
                "rsync", "--recursive", "--delete-unmatched-destination-objects",
                f"--exclude=.*{SKIP}.*", str(src), self.url(key)), key)
            return
        dst = Path(self.url(key))
        dst.mkdir(parents=True, exist_ok=True)
        keep = set()
        for p in sorted(src.rglob("*")):
            rel = p.relative_to(src)
            if _skipped(rel):
                continue
            keep.add(rel)
            if p.is_dir():
                (dst / rel).mkdir(parents=True, exist_ok=True)
            else:
                try:
                    shutil.copy2(p, dst / rel)
                except FileNotFoundError:    # removed since the listing
                    keep.discard(rel)
        for p in sorted(dst.rglob("*"), reverse=True):
            rel = p.relative_to(dst)
            if rel not in keep and not _skipped(rel):
                shutil.rmtree(p) if p.is_dir() else p.unlink()

    @staticmethod
    def _check(r: subprocess.CompletedProcess, key: str) -> None:
        if r.returncode:
            raise RuntimeError(f"gcloud storage on {key}: {r.stderr.strip()}")


class Stop:
    """Records the first SIGTERM / SIGINT; the runner acts on it."""

    def __init__(self):
        self.signum: Optional[int] = None
        self.forwarded: Optional[float] = None

    def __call__(self, signum, frame):
        if self.signum is None:
            self.signum = signum
            print(f"runq: signal {signum}, stopping the job", flush=True)


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def signal_group(pid: int, sig: int) -> None:
    """Signal the job's process group; it may have exited meanwhile."""
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queue", required=True, help="queue file")
    p.add_argument("--check", action="store_true",
                   help="parse the queue, list its jobs and exit")
    p.add_argument("--bucket", default=None,
                   help="gs://bucket/prefix, or a local directory")
    p.add_argument("--chip", type=int, default=None,
                   help="run every job on this chip alone")
    p.add_argument("--precision", default=None,
                   choices=("default", "high", "highest"),
                   help="JAX_DEFAULT_MATMUL_PRECISION for the jobs, unless the "
                        "runner's environment sets it")
    p.add_argument("--sync-every", type=float, default=120.0, help="seconds")
    p.add_argument("--grace", type=float, default=25.0,
                   help="seconds a stopped job gets before SIGKILL")
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--root", default=str(ROOT), help="repository root")
    p.add_argument("--logs", default="experiments/logs/cloud",
                   help="local log directory, relative to --root")
    return p.parse_args(argv)


def run_job(job: Job, store: Store, cfg: argparse.Namespace, stop: Stop,
            runner: str, code: Optional[str]) -> Optional[int]:
    """The job's exit status, or None when the runner was stopped."""
    root = Path(cfg.root)
    out = root / job.out
    log = root / cfg.logs / f"{job.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    pulled = store.pull(f"runs/{job.out}", out)
    if stop.signum is not None:
        return None
    env = job_env(os.environ, cfg.chip, cfg.precision)
    with open(log, "a") as fh:
        fh.write(f"== {stamp()} runq {runner}: start (pulled state: "
                 f"{pulled}; code {code})\n$ {job.command}\n")
        fh.flush()
        p = subprocess.Popen(["bash", "-c", job.command], cwd=root, env=env,
                             stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)

    def push():
        store.push(out, f"runs/{job.out}")
        store.put_file(log, f"logs/{job.name}.log")
        store.write_json(f"status/{runner}.json",
                         {"job": job.name, "pid": p.pid, "rc": rc,
                          "time": stamp(), "code": code})

    def sync(final: bool = False):
        # a periodic sync can race a checkpoint being written; it is retried
        # at the next interval. The final one is retried, then raises
        for attempt in range(3 if final else 1):
            try:
                return push()
            except (OSError, RuntimeError) as e:
                print(f"runq: sync of {job.name} failed ({e})", flush=True)
                if final and attempt == 2:
                    raise
                time.sleep(2.0 if final else 0.0)

    rc = None
    last = time.monotonic()
    while True:
        try:
            rc = p.wait(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            pass
        if stop.signum is not None and stop.forwarded is None:
            signal_group(p.pid, signal.SIGTERM)
            stop.forwarded = time.monotonic()
        if (stop.forwarded is not None
                and time.monotonic() - stop.forwarded > cfg.grace):
            signal_group(p.pid, signal.SIGKILL)
        if stop.signum is None and time.monotonic() - last >= cfg.sync_every:
            sync()
            last = time.monotonic()
    with open(log, "a") as fh:
        fh.write(f"== {stamp()} runq {runner}: exit {rc}\n")
    sync(final=True)
    return None if stop.signum is not None else rc


def main(argv=None) -> int:
    cfg = parse_args(argv)
    jobs = parse_queue(Path(cfg.queue).read_text())
    if cfg.check:
        for j in jobs:
            print(f"{j.name}\t{j.out}\t{j.command}")
        return 0
    if not cfg.bucket:
        sys.exit("runq: --bucket is required")
    store = Store(cfg.bucket)
    stop = Stop()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    runner = Path(cfg.queue).stem + ("" if cfg.chip is None else f"_chip{cfg.chip}")
    code_file = Path(cfg.root) / ".code_archive"
    code = code_file.read_text().strip() if code_file.exists() else None
    for job in jobs:
        if stop.signum is not None:
            return STOPPED
        marker = store.read_json(f"queue/{job.name}.json") or {}
        if marker.get("done"):
            print(f"runq: {job.name} done, skipped", flush=True)
            continue
        attempts = int(marker.get("attempts", 0))
        if attempts >= cfg.max_attempts:
            print(f"runq: {job.name} failed {attempts} times, skipped", flush=True)
            continue
        print(f"runq: {job.name} starts ({stamp()})", flush=True)
        rc = run_job(job, store, cfg, stop, runner, code)
        if rc is None:
            print(f"runq: {job.name} stopped; it resumes on restart", flush=True)
            return STOPPED
        attempts += rc != 0
        store.write_json(f"queue/{job.name}.json",
                         {"done": rc == 0, "rc": rc, "attempts": attempts,
                          "finished": stamp(), "command": job.command,
                          "out": job.out, "code": code})
        print(f"runq: {job.name} exited {rc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
