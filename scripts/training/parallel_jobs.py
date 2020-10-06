#!/usr/bin/python3

import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

q = queue.Queue()


def worker(gpu: int):
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONUNBUFFERED': '1'})
    while not q.empty():
        cmd = q.get()
        print("Executing command on GPU_{}: {}".format(gpu, cmd))
        proc = subprocess.Popen(
            cmd, env=env, shell=True, text=True, bufsize=1,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        with proc.stdout:
            for line in iter(proc.stdout.readline, ''):
                sys.stdout.write("GPU_{}: {}".format(gpu, line))
                sys.stdout.flush()
        proc.wait()
        q.task_done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', type=Path, required=True,
        help="File containing train commands that can each run on a single "
             "GPU."
    )
    parser.add_argument('--gpus', type=int, default=2, help="Number of GPUs.")
    args = parser.parse_args()

    with open(args.input) as fid:
        for line in fid:
            line = line.strip()
            q.put(line)
            print("Command {}".format(line))

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(args.gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
