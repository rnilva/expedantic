import argparse
import inspect
from typing import Any, Type, get_origin, get_args, Sequence

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from tap.utils import type_to_str, get_literals


EMPTY = inspect.Parameter.empty
POS_ONLY = inspect.Parameter.POSITIONAL_ONLY


class NOT_PROVIDED_CLASS:
    def __repr__(self) -> str:
        return "NOT_PROVIDED"


_NOT_PROVIDED = NOT_PROVIDED_CLASS()


def get_kwargs(cls: Type):
    signature = inspect.signature(cls)
    kw_args = {
        k: (
            v.annotation if v.annotation != EMPTY else Any,
            v.default if v.default != EMPTY else ...,
        )
        for k, v in signature.parameters.items()
        if v.kind != POS_ONLY
    }
    return kw_args


def flatten_dict(d: dict[str, Any], parent_key="", sep=".") -> dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, parent_key=new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_default_dict(model: type[BaseModel], not_provided_value=None):
    result: dict[str, Any] = {}
    for name, field_info in model.model_fields.items():
        if field_info.is_required():
            tp = field_info.annotation
            if inspect.isclass(tp) and issubclass(tp, BaseModel):
                value = get_default_dict(tp, not_provided_value)
            else:
                value = not_provided_value
        else:
            value = field_info.get_default(call_default_factory=True)
            if isinstance(value, BaseModel):
                value = get_default_dict(type(value), not_provided_value)

        result[name] = value
    return result


def get_field_info(
    model_cls: Type[BaseModel],
    prefix: str = "",
    sep=".",
    underscore_to_hyphen=False,
) -> dict[str, FieldInfo]:
    """
    Recursively extract FieldInfo from a Pydantic BaseModel class, including nested models.

    Args:
        model_cls: The Pydantic BaseModel class to inspect
        prefix: Current prefix for nested field names (used in recursion)

    Returns:
        A dictionary mapping field paths to their FieldInfo objects

    Example:
        class Address(BaseModel):
            street: str = Field(description="Street address")
            city: str = Field(description="City name")

        class User(BaseModel):
            name: str = Field(description="User's full name")
            age: int = Field(ge=0, description="User's age")
            address: Address

        field_info = get_field_info(User)
        # Returns:
        # {
        #     'name': FieldInfo(...),
        #     'age': FieldInfo(...),
        #     'address.street': FieldInfo(...),
        #     'address.city': FieldInfo(...)
        # }
    """
    result = {}

    # Get model's field definitions
    model_fields = model_cls.model_fields

    for field_name, field in model_fields.items():
        if underscore_to_hyphen:
            field_name = field_name.replace("_", "-")

        # Build the full field path
        full_path = f"{prefix}{field_name}" if prefix else field_name

        # Add the current field's FieldInfo
        result[full_path] = field

        # Get the field type
        field_type = field.annotation

        # Handle Optional/Union types
        origin = get_origin(field_type)
        if origin is not None:
            args = get_args(field_type)
            # Look for BaseModel in Union types
            field_type = next(
                (
                    arg
                    for arg in args
                    if isinstance(arg, type)
                    and not get_origin(arg)
                    and issubclass(arg, BaseModel)
                ),
                None,
            )

        # Recursively process nested BaseModels
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            nested_fields = get_field_info(field_type, prefix=f"{full_path}{sep}")
            result.update(nested_fields)

    return result


def is_base_model(tp):
    return isinstance(tp, type) and not get_origin(tp) and issubclass(tp, BaseModel)


class ArgumentParser(argparse.ArgumentParser):
    def _get_value(self, action, arg_string):
        try:
            return super()._get_value(action, arg_string)
        except (argparse.ArgumentError, TypeError, ValueError):
            # Return the original string if type conversion fails
            return arg_string

    def _check_value(self, action, value):
        try:
            return super()._check_value(action, value)
        except:
            return

    def parse_all_args_as_dict(self, args: Sequence[str] | None):
        known, unknown_args = self.parse_known_args(args=args)

        unknown_parser = argparse.ArgumentParser(
            argument_default=argparse.SUPPRESS, usage=argparse.SUPPRESS, add_help=False
        )

        processed_unknown = []
        for arg in unknown_args:
            if arg.startswith("--") and "=" in arg:
                key, value = arg.split("=", 1)
                processed_unknown.extend([key, value])
            else:
                processed_unknown.append(arg)

        # Make it accept any argument
        for arg in processed_unknown:
            if arg.startswith("--"):
                name = arg[2:]  # strip off the --
                unknown_parser.add_argument(
                    f"--{name}", type=str, nargs="?", const=True
                )

        unknown = unknown_parser.parse_args(unknown_args)

        known_dict = vars(known)
        unknown_dict = vars(unknown)
        known_dict.update(unknown_dict)
        return known_dict
