import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Tuple

import click
from click_option_group import RequiredMutuallyExclusiveOptionGroup, optgroup


@click.command()
@click.argument(
    "input", type=click.Path(exists=True, dir_okay=False, path_type=Path), nargs=-1
)
@click.option(
    "--failed", type=Path, help="Where to log failed commands.", required=True
)
@optgroup.group("Parallel method", cls=RequiredMutuallyExclusiveOptionGroup)
@optgroup.option("--cpus", type=int, help="Number of CPU threads to use")
@optgroup.option("--gpus", help="GPU IDs to run on.")
def main(input: Tuple[Path], cpus: int, gpus: str, failed: Path):
    """Run all commands specified in the INPUT file(s), splitting the
    work across multiple CPU threads or GPUs such that each command runs
    solely on whichever thread/GPU is next available.

    Each GPU still has it's own thread to run processes using
    CUDA_VISIBLE_DEVICES. Each thread reads from a synchronous queue and
    runs the command in a shell.
    """
    if len(input) == 0:
        raise ValueError("No input files specified.")

    q: "Queue[str]" = Queue()
    file_lock = Lock()
    if not failed.exists():
        failed.parent.mkdir(exist_ok=True, parents=True)
    fail_file = open(failed, "w")
    n_failed = 0

    def worker(t_id: int = None, gpu: int = None):
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1"})
        if gpu is not None:
            env.update({"CUDA_VISIBLE_DEVICES": str(gpu)})
        else:  # CPU
            env.update({"OMP_NUM_THREADS": "1"})
        print_prefix = f"[T{t_id}]" if gpu is None else f"[GPU{gpu}]"
        while not q.empty():
            cmd = q.get()
            print(
                datetime.now().isoformat(),
                f"Executing command on {print_prefix}: {cmd}",
                flush=True,
            )
            proc = subprocess.Popen(
                cmd,
                env=env,
                shell=True,
                text=True,
                bufsize=1,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if proc.stdout is not None:
                with proc.stdout:
                    for line in iter(proc.stdout.readline, ""):
                        sys.stdout.write(
                            f"{datetime.now().isoformat()} {print_prefix}:{line}"
                        )
                        sys.stdout.flush()
            if proc.wait() != 0:
                nonlocal n_failed
                with file_lock:
                    n_failed += 1
                    fail_file.write(cmd + "\n")
                    fail_file.flush()
            q.task_done()

    for file in input:
        with open(file) as fid:
            for line in filter(None, map(str.strip, fid)):
                q.put(line)
    n_commands = q.qsize()
    print(f"Loaded {n_commands} commands")

    if cpus is not None:
        threads = [Thread(target=worker, kwargs={"t_id": i}) for i in range(cpus)]
    else:
        _gpus = list(map(int, gpus.split(",")))
        threads = [Thread(target=worker, kwargs={"gpu": i}) for i in _gpus]
    print(f"Starting threads at {datetime.now().isoformat()}")
    start_time = datetime.now()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_time = datetime.now() - start_time
    print(f"Finished at {datetime.now().isoformat()}")
    print(f"Total time: {total_time} ({total_time.total_seconds()} s)")
    print(f"{n_commands - n_failed} commands succeeded, {n_failed} failed")


if __name__ == "__main__":
    main()
