import os
import subprocess
import sys
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Tuple

import click

from ertk.utils import PathlibPath


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False), nargs=-1)
@click.option("--gpus", default="0", help="GPUs to run on.", show_default=True)
@click.option("--failed", type=Path, help="Where to log failed commands.")
def main(input: Tuple[Path], gpus: str, failed: Path):
    """Runs all commands specified in the INPUT file(s), splitting the
    work across multiple GPUs such that each command runs solely on
    whichever GPU is next available.

    Each GPU is given its own thread that reads from a synchronous queue
    and runs the command in a shell with the corresponding
    CUDA_VISIBLE_DEVICES environment variable set appropriately.
    """
    if len(input) == 0:
        raise RuntimeError("No input files specified.")

    q: "Queue[str]" = Queue()
    file_lock = Lock()
    fail_file = open(failed, "w")

    def worker(gpu: int):
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"})
        while not q.empty():
            cmd = q.get()
            print(f"Executing command on GPU_{gpu}: {cmd}")
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
                        sys.stdout.write(f"[GPU{gpu}]: {line}")
                        sys.stdout.flush()
            if proc.wait() != 0:
                with file_lock:
                    fail_file.write(cmd + "\n")
                    fail_file.flush()
            q.task_done()

    for file in input:
        with open(file) as fid:
            for line in fid:
                line = line.strip()
                if len(line) == 0:
                    continue
                q.put(line)
                print(f"Command {line}")

    try:
        _gpus = [int(gpus)]
    except ValueError:
        _gpus = list(map(int, gpus.split(",")))

    threads = [Thread(target=worker, args=(i,)) for i in _gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
