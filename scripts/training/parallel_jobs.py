import os
import subprocess
import sys
from pathlib import Path
from queue import Queue
from threading import Thread

import click
from emorec.utils import PathlibPath

q: 'Queue[str]' = Queue()


def worker(gpu: int):
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONUNBUFFERED': '1'})
    while not q.empty():
        cmd = q.get()
        print(f"Executing command on GPU_{gpu}: {cmd}")
        proc = subprocess.Popen(
            cmd, env=env, shell=True, text=True, bufsize=1,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        if proc.stdout is not None:
            with proc.stdout:
                for line in iter(proc.stdout.readline, ''):
                    sys.stdout.write(f"GPU_{gpu}: {line}")
                    sys.stdout.flush()
        proc.wait()
        q.task_done()


@click.command()
@click.argument('input', type=PathlibPath(exists=True, dir_okay=False))
@click.option('--gpus', type=int, default=2, help="Number of GPUs")
def main(input: Path, gpus: int):
    """Runs all commands specified in the INPUT file, splitting the work
    across multiple GPUs such that each command runs solely on whichever
    GPU is next available.

    Each GPU is given its own thread that reads from a synchronous queue
    and runs the command in a shell with the corresponding
    CUDA_VISIBLE_DEVICES environment variable set appropriately.
    """

    with open(input) as fid:
        for line in fid:
            line = line.strip()
            if len(line) == 0:
                continue
            q.put(line)
            print(f"Command {line}")

    threads = [Thread(target=worker, args=(i,)) for i in range(gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
