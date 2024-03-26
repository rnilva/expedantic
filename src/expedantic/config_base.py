import argparse
import inspect
import json

from abc import ABC
from collections.abc import Mapping
from io import IOBase
from pathlib import Path
from types import UnionType
from typing import Any, Callable, Type, Literal, Union, get_origin, get_args
from typing_extensions import Self

import pydantic
import pydantic_yaml

# from ruamel.yaml import YAML
from ccorp.ruamel.yaml.include import YAML

pydantic_yaml.loader.YAML = YAML

from . import utils


class NOT_PROVIDED_CLASS:
    def __repr__(self) -> str:
        return "NOT_PROVIDED"


_NOT_PROVIDED = NOT_PROVIDED_CLASS()


class ConfigBase(pydantic.BaseModel, Mapping, ABC):
    model_config = pydantic.ConfigDict(
        extra="forbid", protected_namespaces=("model_", "expedantic_")
    )

    _mutually_exclusive_sets: list[set[str]] = []
    """
    Defines sets of configuration options within `ConfigBase` that are mutually exclusive. Each set contains 
    keys of options where, logically, only one option can be evaluated as True at any given time. An option 
    is considered to be evaluated as True if its value is truthy (i.e., not `None`, `False`, an empty string, 
    or any other value that is considered falsy in a boolean context).

    If more than one option in a mutually exclusive set is evaluated as True, a `ValidationError` is raised, 
    enforcing that incompatible configuration states do not co-occur. This feature is crucial for maintaining 
    the integrity and logical consistency of the configuration.

    Type:
        list[set[str]]: Each inner set comprises strings that represent the keys of the configuration options. 
        These options are mutually exclusive with each other, ensuring that only one can be evaluated as True, 
        promoting clear and logical configuration setups.
    """

    def flatten(self, sep="."):
        """
        Flattens the configuration object's hierarchy into a single-level dictionary, using a
        custom separator for keys. This method is useful for converting nested configurations into
        a flat structure, facilitating serialization or simplifying access to nested values.

        Parameters:
        - sep (str, optional): The separator character used to join nested keys in the resulting
        flat dictionary. Defaults to '.'.

        Returns:
        dict[str, Any]: A flattened dictionary where keys are paths to the original nested values,
        joined by the specified separator. Each key corresponds to a value from the original
        configuration.

        Example:
        Given a configuration object structured as `{'outer': {'inner': 'value'}}` and using the
        default separator, the result would be `{'outer.inner': 'value'}`. If a separator of '/'
        is used, the result would be `{'outer/inner': 'value'}`.

        Note:
        The method does not modify the original configuration object but returns a new dictionary
        with the flattened structure.
        """
        return utils.flatten_dict(self.model_dump(), sep=sep)

    def compatible_args(self, cls: Type | Callable, *exclusive_keys: str):
        """
        Returns a dictionary of arguments from this config instance that match the signature of `cls`,
        excluding any specified in `exclusive_keys`. This method is useful for dynamically initialising
        components with applicable settings.

        Parameters:
        - cls (Type | Callable): Class or callable to filter compatible configuration arguments for.
        - exclusive_keys (str): Names of keys to exclude from the result.

        Returns:
        Dictionary of compatible arguments, excluding those listed in `exclusive_keys`.

        Example:
        Given a configuration with attributes 'a', 'b', 'c', and a function requiring 'a' and 'b',
        calling `self.compatible_args(target_function, 'c')` yields {'a': value, 'b': value}.
        """
        fields = self.model_dump()
        args = utils.get_kwargs(cls)
        arg_keys = set(args.keys())
        arg_keys -= set(exclusive_keys)
        return {k: v for k, v in fields.items() if k in arg_keys}

    def save_as_yaml(
        self,
        file: Path | str | IOBase,
        default_flow_style: bool | None = False,
        indent: int | None = None,
        map_indent: int | None = None,
        sequence_indent: int | None = None,
        sequence_dash_offset: int | None = None,
        custom_yaml_writer: YAML | None = None,
        **json_kwargs,
    ):
        """Write a YAML file representation of the model.

        Parameters
        ----------
        file : Path or str or IOBase
            The file path or stream to write to.
        model : BaseModel
            The model to write.
        default_flow_style : bool
            Whether to use "flow style" (more human-readable).
            https://yaml.readthedocs.io/en/latest/detail.html?highlight=default_flow_style#indentation-of-block-sequences
        indent : None or int
            General indent value. Leave as None for the default.
        map_indent, sequence_indent, sequence_dash_offset : None or int
            More specific indent values.
        custom_yaml_writer : None or YAML
            An instance of ruamel.yaml.YAML (or a subclass) to use as the writer.
            The above options will be set on it, if given.
        json_kwargs : Any
            Keyword arguments to pass `model.json()`.

        Notes
        -----
        This currently uses JSON dumping as an intermediary.
        This means that you can use `json_encoders` in your model.
        """
        pydantic_yaml.to_yaml_file(
            file,
            self,
            default_flow_style=default_flow_style,
            indent=indent,
            map_indent=map_indent,
            sequence_indent=sequence_indent,
            sequence_dash_offset=sequence_dash_offset,
            custom_yaml_writer=custom_yaml_writer,
            **json_kwargs,
        )

    @classmethod
    def generate_schema(cls, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(cls.model_json_schema(), path.open("w"), indent=2)
        print(f"JSON Schema for {cls.__name__} is generated at {path}")

    @classmethod
    def load_from_yaml(cls, path: Path | str):
        path = Path(path)
        return pydantic_yaml.parse_yaml_file_as(cls, path)

    @classmethod
    def parse_args(cls, require_default_file: bool = False, args=None, sep="."):
        """
        Parses command line arguments and returns an instance of the class with properties set
        based on the provided arguments. This method supports nested configurations and allows
        specifying complex data types including lists, sets, tuples, and literals. It also supports
        loading default values from a YAML file if `require_default_file` is set to True.

        Parameters:
        - require_default_file (bool, optional): If True, expects a path to a YAML file as the first
        positional argument from which to load default values. Defaults to False.
        - args (List[str], optional): A list of strings representing the arguments to parse. If None,
        parses arguments from sys.argv. Defaults to None.
        - sep (str, optional): The separator used for nested argument names. Defaults to '.'.

        Returns:
        An instance of the class, initialised with properties set based on the parsed arguments and
        any defaults specified in the YAML file (if `require_default_file` is True).

        This method dynamically constructs an argument parser based on the class's model fields,
        supporting deep nesting by using the specified separator `sep` to denote hierarchy in argument
        names. It handles various data types appropriately and allows for the specification of default
        values directly in the command line invocation or through a configuration file.
        """

        def _parse_params(
            parser: argparse.ArgumentParser,
            cls: Type[ConfigBase],
            parent_key: str = "",
            sep=sep,
        ):
            for name, field_info in cls.model_fields.items():
                if parent_key:
                    name = f"{parent_key}{sep}{name}"

                tp = field_info.annotation
                assert tp is not None
                origin = get_origin(tp)

                if not origin and inspect.isclass(tp) and issubclass(tp, ConfigBase):
                    _parse_params(parser, tp, name, sep)
                    continue

                names = [f"--{name}"]
                kwargs = {}
                kwargs["type"] = tp

                annot_repr = str(tp) if origin else tp.__name__

                if tp is Any:
                    kwargs["type"] = str
                elif origin is Literal:
                    var_type, literals = utils.get_literals(tp, name)
                    kwargs["type"] = var_type
                    kwargs["choices"] = literals
                    annot_repr = annot_repr.replace("typing.", "")
                elif origin in {list, set, tuple}:
                    kwargs["nargs"] = "*"
                    args = get_args(tp)
                    if len(args) == 0:
                        kwargs["type"] = str
                    else:
                        kwargs["type"] = args[0]
                elif origin in {UnionType, Union}:
                    kwargs["type"] = (
                        str  # Try to solve union types via pydantic internal.
                    )
                    if origin is Union:
                        args: tuple[type, ...] = get_args(tp)
                        names = map(lambda a: a.__name__, args)
                        annot_repr = " | ".join(names)
                elif origin is dict:
                    kwargs["type"] = YAML().load

                if field_info.is_required():
                    req_repr = "required"
                    if not require_default_file:
                        kwargs["default"] = _NOT_PROVIDED
                else:
                    default_value = field_info.get_default(call_default_factory=True)
                    req_repr = f"default={default_value}"
                    if not require_default_file:
                        kwargs["default"] = default_value

                kwargs["help"] = f"({annot_repr}, {req_repr})"

                if require_default_file:
                    kwargs["default"] = _NOT_PROVIDED

                parser.add_argument(*names, **kwargs)

        parser = argparse.ArgumentParser()
        if require_default_file:
            parser.add_argument(
                "_config_file_path", type=Path, help="Default Config File Path"
            )
        _parse_params(parser, cls)
        args = parser.parse_args(args=args)

        if require_default_file:
            with open(args._config_file_path, "r") as f:
                file_dict = YAML().load(f)
            args.__dict__.pop("_config_file_path")
        else:
            file_dict = {}

        nested_args_dict = {}
        for k, v in vars(args).items():
            if v is _NOT_PROVIDED:
                continue
            keys = k.split(sep)
            current_level = nested_args_dict
            for key in keys[:-1]:
                current_level = current_level.setdefault(key, {})
            current_level[keys[-1]] = v

        def update_recursively(d: dict, other: dict):
            for k, v in other.items():
                if isinstance(v, dict):
                    d[k] = update_recursively(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        file_dict = update_recursively(file_dict, nested_args_dict)
        instance = cls.model_validate(file_dict)

        return instance

    @pydantic.model_validator(mode="after")
    def check_mutually_exclusive_sets(self) -> Self:
        for exclusive_set in self._mutually_exclusive_sets:
            check = sum(bool(self[key]) for key in exclusive_set) == 1
            if not check:
                raise ValueError(
                    f"Mutual exclusivity has broken. (set: {exclusive_set})"
                )
        return self

    def __getitem__(self, key: str):
        if key not in self.model_fields:
            raise KeyError(f"Key '{key}' not found.")
        return self.__getattribute__(key)

    def __len__(self):
        return len(self.model_fields)

    def keys(self):
        return self.model_fields.keys()
