import json
from pathlib import Path

import pytest


@pytest.fixture(scope="function")
def data_conf_file(tmp_path):
    corpus_info = f"{Path(__file__).parent}/../../test_data/corpus_info.yaml"
    with open(tmp_path / "data_conf.yaml", "w") as fid:
        json.dump(
            {
                "datasets": {
                    "test_corpus": {"path": str(corpus_info), "features": "features_2d"}
                }
            },
            fid,
        )
    return tmp_path / "data_conf.yaml"
