from dataclasses import dataclass
from typing import Any, Type

import pytest
from omegaconf import MISSING, DictConfig, OmegaConf
from omegaconf.errors import ConfigKeyError, MissingMandatoryValue, ValidationError

from ertk.config import ERTKConfig, get_arg_mapping


@dataclass
class MyConfig(ERTKConfig):
    a: int = 1
    b: str = "hello"
    c: bool = False


@dataclass
class MyMissingConfig(ERTKConfig):
    a: int = MISSING
    b: str = MISSING
    c: bool = MISSING


def test_to_dictconfig():
    config = MyConfig()
    dictconfig = config.to_dictconfig()
    assert isinstance(dictconfig, DictConfig)
    assert config == MyConfig(a=1, b="hello", c=False)


@pytest.mark.parametrize(
    "conf",
    [
        {"a": 1, "b": "hello", "c": False},
        OmegaConf.create({"a": 1, "b": "hello", "c": False}),
    ],
)
@pytest.mark.parametrize("cls", [MyConfig, MyMissingConfig])
def test_from_config(conf: Any, cls: Type[ERTKConfig]):
    config = cls.from_config(conf)
    assert config == cls(a=1, b="hello", c=False)


@pytest.mark.parametrize(
    "conf",
    [
        {"a": 1, "b": "hello"},
        OmegaConf.create({"a": 1, "b": "hello"}),
        MyMissingConfig(a=1, b="hello"),
    ],
)
def test_from_config_missing_error(conf: Any):
    config = MyMissingConfig.from_config(conf)
    with pytest.raises(MissingMandatoryValue):
        config.c


@pytest.mark.parametrize("cls", [MyConfig, MyMissingConfig])
def test_from_file(tmp_path, cls: Type[ERTKConfig]):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
a: 1
b: hello
c: false
"""
    )
    config = cls.from_file(config_file)
    assert config == cls(a=1, b="hello", c=False)


def test_from_file_missing(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
a: 1
b: hello
"""
    )
    config = MyMissingConfig.from_file(config_file)
    with pytest.raises(MissingMandatoryValue):
        config.c


def test_from_file_extra(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
a: 1
b: hello
c: false
d: 2
"""
    )
    with pytest.raises(ConfigKeyError):
        MyConfig.from_file(config_file)


def test_to_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    config = MyConfig(a=1, b="hello", c=False)
    config.to_file(config_file)
    assert MyConfig.from_file(config_file) == config


def test_to_string():
    config = MyConfig(a=1, b="hello", c=False)
    assert config.to_string().strip() == "a: 1\nb: hello\nc: false"


def test_invalid_type():
    config = OmegaConf.structured(MyConfig)
    with pytest.raises(ValidationError):
        config.a = "hello"


def test_merge_with_config():
    config = MyConfig(a=1, b="hello", c=False)
    config = MyConfig.merge_with_config(config, MyConfig(a=2, b="world", c=True))
    assert config == MyConfig(a=2, b="world", c=True)


def test_merge_with_args():
    config = MyConfig(a=1, b="hello", c=False)
    config = MyConfig.merge_with_args(config, ["a=2", "b=world"])
    assert config == MyConfig(a=2, b="world", c=False)


def test_get_arg_mapping():
    assert get_arg_mapping("a:1,b:hello,c:false") == {
        "a": "1",
        "b": "hello",
        "c": "false",
    }


def test_get_arg_mapping_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
a: 1
b: hello
c: false
"""
    )
    assert get_arg_mapping(config_file) == {"a": 1, "b": "hello", "c": False}
