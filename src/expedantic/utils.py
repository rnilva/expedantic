import inspect
from typing import Any, Type

from tap.utils import type_to_str, get_literals


EMPTY = inspect.Parameter.empty
POS_ONLY = inspect.Parameter.POSITIONAL_ONLY


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
