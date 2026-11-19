# experiments/cloud: the queue runner against a local directory standing in
# for the bucket (a finished job is skipped; a stopped job's state reaches the
# bucket and a fresh VM resumes it; a failing job is retried, then given up),
# the mirror sync, queue parsing, the per-chip environment, and the TPU
# requirements filter.
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

CLOUD = Path(__file__).parents[1] / "experiments" / "cloud"
sys.path.insert(0, str(CLOUD))
from runq import Store, chip_env, job_env, parse_queue  # noqa: E402
from tpureqs import requirements, version  # noqa: E402

# a job that appends steps 1..n to out/steps.txt, resuming after the last one
# present: what a trainer does with its own latest checkpoint
STEPPER = ("import pathlib, sys, time; p = pathlib.Path(sys.argv[1]); "
           "p.parent.mkdir(parents=True, exist_ok=True); "
           "done = len(p.read_text().split()) if p.exists() else 0\n"
           "for s in range(done + 1, int(sys.argv[2]) + 1):\n"
           "    open(p, 'a').write(f'{s}\\n'); time.sleep(float(sys.argv[3]))")


def queue(tmp_path, lines):
    q = tmp_path / "q0.txt"
    q.write_text("".join("\t".join(f) + "\n" for f in lines))
    return q


def stepper_job(tmp_path, name, out, n, dt):
    script = tmp_path / "stepper.py"
    script.write_text(STEPPER)
    return (name, out, f"{sys.executable} {script} {out}/steps.txt {n} {dt}")


def runq(q, bucket, root, *extra):
    return subprocess.Popen([sys.executable, str(CLOUD / "runq.py"), "--queue",
                             str(q), "--bucket", str(bucket), "--root",
                             str(root), "--sync-every", "0.2", "--grace", "5",
                             *extra], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)


def steps(path):
    return [int(s) for s in path.read_text().split()] if path.exists() else []


def test_parse_queue():
    jobs = parse_queue("# a comment\n\nlcm_s42\tdata/out/lcm/s42\tpython a.py\t--x 1\n")
    assert [(j.name, j.out, j.command) for j in jobs] == \
        [("lcm_s42", "data/out/lcm/s42", "python a.py\t--x 1")]
    for bad in ("a\tb\n", "a\t/abs\tcmd\n", "a\t../up\tcmd\n", "a b\tout\tcmd\n",
                "a\tx\tc\na\ty\tc\n"):
        with pytest.raises(ValueError):
            parse_queue(bad)


def test_job_env_confines_to_a_chip_and_sets_a_default_precision():
    env = job_env({}, 2, "high")
    assert env["TPU_VISIBLE_CHIPS"] == "2" and env["TPU_PROCESS_BOUNDS"] == "1,1,1"
    assert env["JAX_DEFAULT_MATMUL_PRECISION"] == "high"
    assert job_env({"JAX_DEFAULT_MATMUL_PRECISION": "highest"}, None,
                   "high")["JAX_DEFAULT_MATMUL_PRECISION"] == "highest"
    assert "TPU_VISIBLE_CHIPS" not in job_env({}, None, None)
    assert chip_env(0)["TPU_PROCESS_PORT"] != chip_env(1)["TPU_PROCESS_PORT"]


def test_a_job_reads_the_precision_and_can_override_it(tmp_path):
    root, bucket = tmp_path / "repo", tmp_path / "bucket"
    show = ("import os, pathlib, sys; p = pathlib.Path(sys.argv[1]); "
            "p.parent.mkdir(parents=True, exist_ok=True); "
            "p.write_text(os.environ['JAX_DEFAULT_MATMUL_PRECISION'] + ' ' + "
            "os.environ['TPU_VISIBLE_CHIPS'])")
    script = tmp_path / "show.py"
    script.write_text(show)
    q = queue(tmp_path, [
        ("e", "data/out/e", f"{sys.executable} {script} data/out/e/p.txt"),
        ("f", "data/out/f", f"JAX_DEFAULT_MATMUL_PRECISION=highest "
                            f"{sys.executable} {script} data/out/f/p.txt")])
    assert runq(q, bucket, root, "--chip", "3", "--precision", "high").wait(30) == 0
    assert (bucket / "runs/data/out/e/p.txt").read_text() == "high 3"
    assert (bucket / "runs/data/out/f/p.txt").read_text() == "highest 3"


def test_push_mirrors_and_skips_orbax_temporaries(tmp_path):
    src, store = tmp_path / "run", Store(str(tmp_path / "bucket"))
    (src / "10").mkdir(parents=True)
    (src / "10" / "a").write_text("x")
    (src / "20.orbax-checkpoint-tmp-1").mkdir()
    (src / "loss.csv").write_text("1")
    store.push(src, "runs/r")
    remote = tmp_path / "bucket" / "runs" / "r"
    assert (remote / "10" / "a").exists() and (remote / "loss.csv").exists()
    assert not (remote / "20.orbax-checkpoint-tmp-1").exists()
    (src / "10" / "a").unlink()                # orbax dropped an old step
    store.push(src, "runs/r")
    assert not (remote / "10" / "a").exists()
    assert not store.pull("runs/none", tmp_path / "x")
    assert store.pull("runs/r", tmp_path / "back") and (tmp_path / "back" / "loss.csv").exists()


def test_a_finished_job_is_marked_and_skipped(tmp_path):
    root, bucket = tmp_path / "repo", tmp_path / "bucket"
    q = queue(tmp_path, [stepper_job(tmp_path, "a", "data/out/a", 5, 0.0)])
    p = runq(q, bucket, root)
    assert p.wait(30) == 0, p.stdout.read()
    marker = json.loads((bucket / "queue" / "a.json").read_text())
    assert marker["done"] and marker["rc"] == 0
    assert steps(bucket / "runs" / "data/out/a" / "steps.txt") == [1, 2, 3, 4, 5]
    assert (bucket / "logs" / "a.log").exists()
    p = runq(q, bucket, root)
    assert p.wait(30) == 0 and "done, skipped" in p.stdout.read()


def test_a_stopped_job_resumes_on_a_fresh_vm_from_the_bucket(tmp_path):
    root, bucket = tmp_path / "repo", tmp_path / "bucket"
    q = queue(tmp_path, [stepper_job(tmp_path, "b", "data/out/b", 60, 0.05)])
    remote = bucket / "runs" / "data/out/b" / "steps.txt"
    p = runq(q, bucket, root)
    t0 = time.monotonic()
    while len(steps(remote)) < 5 and time.monotonic() - t0 < 20:
        time.sleep(0.05)
    p.send_signal(signal.SIGTERM)
    assert p.wait(30) == 75, p.stdout.read()
    assert not (bucket / "queue" / "b.json").exists()
    kept = steps(remote)
    assert 5 <= len(kept) < 60 and kept == list(range(1, len(kept) + 1))
    p2 = runq(q, bucket, tmp_path / "fresh_vm")
    assert p2.wait(30) == 0, p2.stdout.read()
    assert steps(remote) == list(range(1, 61))       # continued, not restarted
    assert json.loads((bucket / "queue" / "b.json").read_text())["done"]


def test_a_failing_job_is_retried_then_given_up(tmp_path):
    root, bucket = tmp_path / "repo", tmp_path / "bucket"
    q = queue(tmp_path, [("c", "data/out/c", "exit 3"),
                         stepper_job(tmp_path, "d", "data/out/d", 2, 0.0)])
    for attempt in (1, 2):
        assert runq(q, bucket, root).wait(30) == 0
        marker = json.loads((bucket / "queue" / "c.json").read_text())
        assert (marker["done"], marker["rc"], marker["attempts"]) == (False, 3, attempt)
    p = runq(q, bucket, root)
    assert p.wait(30) == 0 and "failed 2 times, skipped" in p.stdout.read()
    assert json.loads((bucket / "queue" / "d.json").read_text())["done"]


def test_tpu_requirements_drop_the_cuda_and_torch_stack():
    text = ("absl-py==2.5.0\n    # via\n    #   grain\n"
            "jax==0.11.2\njax-cuda12-plugin==0.11.2\n    # via jax\n"
            "jaxlib==0.11.2\nnvidia-cublas-cu12==12.9.2.10\n    # via torch\n"
            "cuda-bindings==13.4.3 ; sys_platform == 'linux'\n"
            "torch==2.14.1\ntorch-xla==2.9.0\ntriton==3.8.0\ntqdm==4.67\n")
    kept = requirements(text)
    assert [l.split("==")[0] for l in kept.splitlines() if not l[:1].isspace()] == \
        ["absl-py", "jax", "jaxlib", "tqdm"]
    assert "# via jax" not in kept and "#   grain" in kept


def test_the_lock_pins_matching_jax_and_jaxlib():
    lock = (Path(__file__).parents[1] / "uv.lock").read_text()
    assert version(lock, "jax") == version(lock, "jaxlib")
    with pytest.raises(KeyError):
        version(lock, "no-such-package")
