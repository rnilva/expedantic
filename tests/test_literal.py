import unittest
from typing import Literal

from pydantic import ValidationError
from pydantic_yaml import parse_yaml_raw_as

from expedantic import ConfigBase


class Config(ConfigBase):
    distribution: Literal["Gaussian", "Uniform", "Cauchy", "Poisson"]


class TestLiteral(unittest.TestCase):
    def test_literal(self):
        with self.assertRaises(ValidationError):
            config = parse_yaml_raw_as(Config, "distribution: Dirichlet")

        config = parse_yaml_raw_as(Config, "distribution: Gaussian")
        self.assertEqual(config.distribution, "Gaussian")
