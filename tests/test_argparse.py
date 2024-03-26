import unittest
import argparse
from typing import Any

from expedantic import ConfigBase


class Config(ConfigBase):
    class Inner(ConfigBase):
        name: str = "Inner"
        favourite_number: float | int = 3.14

    inner: Inner = Inner()
    optional: int | None = 1
    numbers: list[int] = [1, 2, 3]
    my_dict: dict[str, Any] = {}


class TestArgParse(unittest.TestCase):
    def test_arg_parse(self):
        config = Config.parse_args(args=["--optional", "3"])
        self.assertEqual(config.optional, 3)

        config = Config.parse_args(args=["--numbers", "5", "6", "7"])
        self.assertEqual(config.numbers, [5, 6, 7])

        config = Config.parse_args(
            args=["--inner.name", "Test", "--inner.favourite_number", "7"]
        )
        self.assertEqual(config.inner.name, "Test")
        self.assertEqual(config.inner.favourite_number, 7)

    def test_parse_dict(self):
        config = Config.parse_args(args=["--my_dict", "{key: value}"])
        self.assertEqual(config.my_dict.get("key"), "value")
