import unittest
from typing import Any
from typing_extensions import TypedDict

from pydantic import ValidationError
from pydantic_yaml import parse_yaml_raw_as

from expedantic import ConfigBase


class PythonTypedDict(TypedDict):
    name: str
    age: int


class Config(ConfigBase):
    optional: int | None = 1
    dictionary: dict[str, Any] = {}
    typed_dictionary: dict[int, str] = {}
    untyped_list: list = []
    typed_list: list[int] = []
    python_typed_dict: PythonTypedDict = {"name": "Maria", "age": 30}


class TestCompoundTypes(unittest.TestCase):
    def test_optional(self):
        config = parse_yaml_raw_as(Config, "optional: null")
        self.assertEqual(config.optional, None)

    def test_dictionary(self):
        config = parse_yaml_raw_as(
            Config, "dictionary: {name: Maria, job: Developer, age: 30}"
        )
        self.assertEqual(config.dictionary["name"], "Maria")
        self.assertEqual(config.dictionary["job"], "Developer")
        self.assertEqual(config.dictionary["age"], 30)

    def test_typed_dictionary(self):
        with self.assertRaises(ValidationError):
            config = parse_yaml_raw_as(Config, "typed_dictionary: {name: Maria}")

        with self.assertRaises(ValidationError):
            config = parse_yaml_raw_as(Config, "typed_dictionary: {1: 20}")

        config = parse_yaml_raw_as(Config, "typed_dictionary: {1: Maria}")
        self.assertEqual(config.typed_dictionary[1], "Maria")

    def test_untyped_list(self):
        config = parse_yaml_raw_as(Config, "untyped_list: [1, 'Maria', 3.5, [1, 2]]")
        self.assertEqual(config.untyped_list, [1, "Maria", 3.5, [1, 2]])

    def test_typed_list(self):
        with self.assertRaises(ValidationError):
            config = parse_yaml_raw_as(Config, "typed_list: [3, 4, 7.2]")

        config = parse_yaml_raw_as(Config, "typed_list: [0, 1, 2, 3]")
        self.assertEqual(config.typed_list, [0, 1, 2, 3])

    def test_python_typed_dict(self):
        with self.assertRaises(ValidationError):
            config = parse_yaml_raw_as(
                Config,
                "python_typed_dict: {'name': Maria, 'age': 30, 'job': Developer}",
            )

        config = parse_yaml_raw_as(
            Config, "python_typed_dict: {'name': Tom, 'age': 17}"
        )
        self.assertEqual(config.python_typed_dict, {'name': 'Tom', 'age': 17})


if __name__ == "__main__":
    unittest.main()
