from concurrent.futures import ProcessPoolExecutor


def f():
    for _ in range(100000000):
        _ = 10


def run():
    with ProcessPoolExecutor() as pool:
        for _ in range(20):
            pool.submit(f)


def main():
    for _ in range(2):
        print("test")
        run()


if __name__ == "__main__":
    main()
