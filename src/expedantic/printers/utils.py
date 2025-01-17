from typing import Any, Literal, Union, get_origin, get_args
from types import NoneType, UnionType

from rich.text import Text


def format_type(annotation: Any) -> Text:
    """Format type annotation into styled text."""
    text = Text()

    # Handle Union/Optional types
    origin = get_origin(annotation)
    if origin in (UnionType, Union):
        args = get_args(annotation)
        for i, arg in enumerate(args):
            if i > 0:
                text.append(" | ", style="bold")
            text.append(
                (
                    "None"
                    if arg is NoneType
                    else arg.__name__ if hasattr(arg, "__name__") else str(arg)
                ),
                style="blue",
            )
        return text

    # Handle other generic types
    if origin is not None:
        if origin is not Literal:
            text.append(origin.__name__, style="blue")
        args = get_args(annotation)
        if args:
            text.append("[")
            for i, arg in enumerate(args):
                if i > 0:
                    text.append(", ")
                text.append(
                    arg.__name__ if hasattr(arg, "__name__") else str(arg), style="cyan"
                )
            text.append("]")
        return text

    return Text(
        annotation.__name__ if hasattr(annotation, "__name__") else str(annotation),
        style="blue",
    )
