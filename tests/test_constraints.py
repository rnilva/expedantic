import unittest

from pydantic import ConfigDict, ValidationError
from pydantic_yaml import parse_yaml_raw_as
from ruamel.yaml import YAML

from expedantic import ConfigBase, Field


class Config(ConfigBase):
    model_config = ConfigDict(revalidate_instances="always")
    a: int = Field(1, gt=0)
    b: float = Field(0.5, lt=1.0)
    c: float = Field(0.05, le=1.0, ge=0.0)


class ConstraintsTest(unittest.TestCase):
    def test_int_float(self):
        with self.assertRaises(ValidationError):
            config = parse_yaml_raw_as(Config, "a: 1.5")

    def test_gt(self):
        with self.assertRaises(ValidationError):
            config = parse_yaml_raw_as(Config, "a: -1\nb: 0.9")

    def test_lt(self):
        with self.assertRaises(ValidationError):
            config = parse_yaml_raw_as(Config, "a: 1\nb: 1.0")

    def test_le(self):
        config = parse_yaml_raw_as(Config, "c: 1.0")
        self.assertEqual(config.c, 1.0)

    def test_ge(self):
        config = parse_yaml_raw_as(Config, "c: 0.0")
        self.assertEqual(config.c, 0.0)
        with self.assertRaises(ValidationError):
            config = parse_yaml_raw_as(Config, "c: -0.1")


if __name__ == "__main__":
    unittest.main()
