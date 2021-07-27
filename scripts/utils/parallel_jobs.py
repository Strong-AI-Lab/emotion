import os
import subprocess
import sys
from pathlib import Path
from queue import Queue
from threading import Thread

import click

from ertk.utils import PathlibPath

q: "Queue[str]" = Queue()


def worker(t_id: int):
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1"})
    while not q.empty():
        cmd = q.get()
        print(f"Executing command on thread {t_id}: {cmd}")
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
                    sys.stdout.write(f"[T{t_id}]: {line}")
                    sys.stdout.flush()
        proc.wait()
        q.task_done()


@click.command()
@click.argument("input", type=PathlibPath(exists=True, dir_okay=False))
@click.option(
    "--threads",
    "--n_threads",
    type=int,
    default=1,
    help="Number of threads",
    show_default=True,
)
def main(input: Path, n_threads: int):
    """Runs all commands specified in the INPUT file, splitting the work
    across multiple threads such that each command runs solely on
    whichever thread is next available.

    Each thread reads from a synchronous queue and runs the command in a
    shell.
    """

    with open(input) as fid:
        for line in fid:
            line = line.strip()
            if len(line) == 0:
                continue
            q.put(line)
            print(f"Command {line}")

    threads = [Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
