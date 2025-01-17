from itertools import groupby
from operator import itemgetter
from typing import Type, TypedDict

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import ErrorDetails
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich.table import Table
from rich import box

from .utils import format_type
from .. import utils


def format_error_type(error_type: str) -> str:
    """
    Transform error type string to a capitalized readable format.

    Args:
        error_type: The error type string (e.g., "type_error", "value_error")

    Returns:
        Formatted error type (e.g., "Type Error", "Value Error")

    Examples:
        >>> format_error_type("type_error")
        "Type Error"
        >>> format_error_type("value_error.missing")
        "Value Error Missing"
    """
    # Split by dots and underscores
    words = error_type.replace(".", "_").split("_")

    # Capitalize each word
    capitalized = " ".join(word.capitalize() for word in words)

    # Make it plural if it ends with "error"
    if capitalized.endswith("Error"):
        capitalized += "s"

    if capitalized.endswith("Missing"):
        capitalized += " Required Values"

    return capitalized


def get_field_context(
    loc: tuple, field_infos: dict[str, FieldInfo]
) -> tuple[str, FieldInfo | None]:
    """Get field description and constraints for context."""
    field_path = ".".join(x for x in loc if not isinstance(x, int))
    field_info = field_infos.get(field_path)
    if not field_info:
        return "", field_info

    context = []
    if field_info.description:
        context.append(f"Description: {field_info.description}")

    # Add validation rules
    rules = []
    for rule in [
        "gt",
        "ge",
        "lt",
        "le",
        "min_length",
        "max_length",
        "regex",
        "pattern",
        "multiple_of",
    ]:
        value = getattr(field_info, rule, None)
        if value is not None:
            rules.append(f"{rule}: {value}")

    if rules:
        context.append("Validation rules: " + ", ".join(rules))

    return "\n".join(context), field_info


class TableConfig(TypedDict):
    colour: str
    include_msg: bool


def create_table(
    main_tree: Tree,
    field_infos: dict[str, FieldInfo],
    errors: list[ErrorDetails],
    error_type: str,
    table_config: TableConfig,
):
    contexts = [get_field_context(error["loc"], field_infos) for error in errors]
    has_context = any(c[0] for c in contexts)
    node = main_tree.add(f"[{table_config['colour']}]{format_error_type(error_type)}")
    table = Table(box=box.ROUNDED, show_header=True)
    table.add_column("Field", style=f"bold {table_config['colour']}")
    table.add_column("Type")
    if table_config["include_msg"]:
        table.add_column("Error", style="italic")
    if has_context:
        table.add_column("Context", style="dim")

    for i, error in enumerate(errors):
        field_path = ".".join(str(x) for x in error["loc"])
        context, field_info = contexts[i]
        type_ = format_type(field_info.annotation) if field_info is not None else "?"
        row = [field_path, type_]
        if table_config["include_msg"]:
            row.append(error["msg"])
        if has_context:
            row.append(context)
        table.add_row(*row)

    node.add(table)


def print_validation_errors(
    model_cls: Type[BaseModel],
    validation_error: ValidationError,
    console: Console | None = None,
) -> None:
    """
    Pretty print Pydantic validation errors with field context.

    Args:
        model_cls: The Pydantic model class
        validation_error: The ValidationError exception
        field_infos: Dictionary of field information from get_field_info()
    """
    console = console or Console()
    errors = validation_error.errors()

    # Group errors by type
    errors_by_type = {
        k: list(g)
        for k, g in groupby(
            sorted(errors, key=itemgetter("type")), key=itemgetter("type")
        )
    }

    main_tree = Tree(f"[bold red]Validation Errors for {model_cls.__name__}")

    field_infos = utils.get_field_info(model_cls)

    errors_map = {
        "missing": TableConfig(colour="yellow", include_msg=False),
        "type_error": TableConfig(colour="red", include_msg=True),
        "value_error": TableConfig(colour="magenta", include_msg=True),
    }

    for error_type, cfg in errors_map.items():
        if error_type in errors_by_type:
            create_table(
                main_tree, field_infos, errors_by_type[error_type], error_type, cfg
            )

    # Handle other types of errors
    other_errors = {
        k: v
        for k, v in errors_by_type.items()
        if k not in ["missing", "type_error", "value_error"]
    }
    if other_errors:
        other_node = main_tree.add("[blue]Other Validation Errors")
        other_table = Table(box=box.ROUNDED, show_header=True)
        other_table.add_column("Field", style="bold")
        other_table.add_column("Type")
        other_table.add_column("Error Type", style="bold blue")
        other_table.add_column("Error", style="italic")
        other_table.add_column("Context", style="dim")

        for error_type, errors in other_errors.items():
            for error in errors:
                field_path = ".".join(str(x) for x in error["loc"])
                context, field_info = get_field_context(error["loc"], field_infos)
                if error_type == "extra_forbidden":
                    context = Text("Input=").append(f"{error['input']}", style="red")
                other_table.add_row(
                    field_path,
                    (
                        format_type(field_info.annotation)
                        if field_info is not None
                        else ""
                    ),
                    error_type,
                    error["msg"],
                    context,
                )

        other_node.add(other_table)

    # Print the error tree
    console.print(Panel(main_tree, title="Validation Errors", border_style="red"))
