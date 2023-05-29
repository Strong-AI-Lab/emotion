import pytest
from importlib_metadata import entry_points

from ertk.preprocessing import InstanceProcessor


class TestPlugins:
    def test_plugin_loading(self):
        # Need entry points so that imports don't override each other
        eps = entry_points().select(group="ertk.processors")
        for ep in eps:
            cls = ep.load()
            assert issubclass(cls, InstanceProcessor)

    def test_invalid_plugin(self):
        with pytest.raises(KeyError):
            InstanceProcessor.get_processor_class("invalid")
