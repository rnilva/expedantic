import argparse
import inspect
import json

from abc import ABC
from io import IOBase
from pathlib import Path
from typing import Any, Type, Literal, get_origin, get_args

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


class ConfigBase(pydantic.BaseModel, ABC):
    model_config = pydantic.ConfigDict(
        extra="forbid", protected_namespaces=("model_", "config_base_")
    )

    def flatten(self):
        return utils.flatten_dict(self.model_dump())

    def compatible_args(self, cls: Type, *exclusive_keys: str):
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
                    _parse_params(parser, tp, name)
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
            parser.add_argument("_config_file_path", type=Path, help="")
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
