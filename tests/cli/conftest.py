import pytest


@pytest.fixture
def mpl():
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    yield
    plt.close("all")
