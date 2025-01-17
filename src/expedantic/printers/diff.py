from typing import Any, Literal, get_origin
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text
from rich.tree import Tree


TYPE_STYLE_MAP: dict[type | tuple[type, ...], Style] = {
    bool: Style(color="cyan"),
    (int, float): Style(color="magenta"),
    str: Style(color="green"),
    (list, tuple): Style(color="cyan"),
    dict: Style(color="blue"),
}

NONE_STYLE = Style(color="black", dim=True)


def get_value_style(value: Any) -> Style:
    """Get rich style for different value types."""
    if value is None:
        return NONE_STYLE

    for tp, style in TYPE_STYLE_MAP.items():
        if isinstance(value, tp):
            return style

    return Style.parse("default")


def create_diff_tree(
    d1: dict[str, Any],
    d2: dict[str, Any],
    parent: Tree | None = None,
    *,
    dim_unchanged: bool = False,
    skip_unchanged: bool = False,
) -> Tree:
    """
    Recursively create a tree representation showing the differences between two dictionaries.

    Args:
        d1: First dictionary (previous state)
        d2: Second dictionary (current state)
        parent: Parent tree node
        dim_unchanged: Whether to dim unchanged values
        skip_unchanged: Whether to skip unchanged values
    """

    def format_value(v: Any) -> str:
        if isinstance(v, (dict, list)):
            return ""
        if get_origin(v) is Literal:
            ...
        return f": {repr(v)}"

    def create_value_text(
        key: str, value: Any, style_override: str | None = None
    ) -> Text:
        value_str = format_value(value)
        key_style = style_override or "default"
        value_style = style_override or get_value_style(value)

        if value_str:
            return Text.assemble((key, key_style), (value_str, value_style))
        return Text(key, style=key_style)

    def add_dict_to_tree(
        d: dict[str, Any],
        tree: Tree,
        other_dict: dict[str, Any] | None = None,
        style: str | None = None,
    ) -> None:
        for key, value in sorted(d.items()):
            if other_dict is None or key not in other_dict:
                label = create_value_text(key, value, style)
                node = tree.add(label)
                if isinstance(value, dict):
                    add_dict_to_tree(value, node, style=style)
                elif isinstance(value, list):
                    add_list_to_tree(value, node, style=style)

    def add_list_to_tree(lst: list, tree: Tree, style: str | None = None) -> None:
        for i, value in enumerate(lst):
            if isinstance(value, dict):
                node = tree.add(Text(f"[{i}]", style=style))
                add_dict_to_tree(value, node, style=style)
            else:
                tree.add(Text(f"[{i}] {repr(value)}", style=style))

    def compare_values(key: str, v1: Any, v2: Any, tree: Tree) -> None:
        if isinstance(v1, dict) and isinstance(v2, dict):
            node = tree.add(Text(key, style="bold"))
            create_diff_tree(
                v1, v2, node, dim_unchanged=dim_unchanged, skip_unchanged=skip_unchanged
            )
        elif isinstance(v1, list) and isinstance(v2, list):
            node = tree.add(Text(key, style="bold"))
            if v1 == v2:
                if not skip_unchanged:
                    style = "dim" if dim_unchanged else get_value_style(v1)
                    for i, v in enumerate(v1):
                        node.add(create_value_text(f"[{i}]", v, style))
            else:
                node_1 = node.add(Text("Previous:", style="red"))
                for i, v in enumerate(v1):
                    node_1.add(create_value_text(f"[{i}]", v, "red"))
                node_2 = node.add(Text("Current:", style="blue"))
                for i, v in enumerate(v2):
                    node_2.add(create_value_text(f"[{i}]", v, "blue"))
        else:
            if v1 == v2:
                if not skip_unchanged:
                    style = "dim" if dim_unchanged else get_value_style(v1)
                    tree.add(create_value_text(key, v1, style))
            else:
                node = tree.add(Text(key, style="yellow bold"))
                node.add(Text(f"Previous: {repr(v1)}", style="red"))
                node.add(Text(f"Current: {repr(v2)}", style="blue"))

    # Start with a new tree if none is provided
    if parent is None:
        parent = Tree("📦 Dictionary Comparison")

    # All keys in both dictionaries
    all_keys = sorted(set(d1.keys()) | set(d2.keys()))

    for key in all_keys:
        if key in d1 and key in d2:
            # Key exists in both dictionaries
            compare_values(key, d1[key], d2[key], parent)
        elif key in d1:
            # Key only in first dictionary
            node = parent.add(Text(f"❌ {key} (removed)", style="red"))
            value = d1[key]
            if isinstance(value, dict):
                add_dict_to_tree(value, node, style="red")
            elif isinstance(value, list):
                add_list_to_tree(value, node, style="red")
            else:
                node.add(create_value_text("value", value, "red"))
        else:
            # Key only in second dictionary
            node = parent.add(Text(f"✨ {key} (added)", style="blue"))
            value = d2[key]
            if isinstance(value, dict):
                add_dict_to_tree(value, node, style="blue")
            elif isinstance(value, list):
                add_list_to_tree(value, node, style="blue")
            else:
                node.add(create_value_text("value", value, "blue"))

    return parent


def print_tree_diff(
    dict1: dict[str, Any],
    dict2: dict[str, Any],
    *,
    root_name: str | None = None,
    console: Console | None = None,
    dim_unchanged: bool = False,
    skip_unchanged: bool = False,
) -> None:
    """
    Create a tree-style visualization of the differences between two dictionaries.

    Args:
        dict1: First dictionary (previous state)
        dict2: Second dictionary (current state)
        root_name: Optional name for the root node
        console: Optional console instance
        dim_unchanged: Whether to dim unchanged values
        skip_unchanged: Whether to skip unchanged values in the output
    """
    console = console or Console()
    if root_name is not None:
        root = Tree(f"📦 {root_name}")
    else:
        root = None

    # Create the tree
    diff_tree = create_diff_tree(
        dict1,
        dict2,
        root,
        dim_unchanged=dim_unchanged,
        skip_unchanged=skip_unchanged,
    )

    # Create a legend panel
    legend_items = [
        ("Legend\n\n", "bold"),
        ("✨ ", "blue"),
        ("Added in current\n", "default"),
        ("❌ ", "red"),
        ("Removed from previous\n", "default"),
        ("Yellow", "yellow"),
        (" Modified keys\n", "default"),
        ("Red", "red"),
        (" Previous values\n", "default"),
        ("Blue", "blue"),
        (" Current values\n\n", "default"),
        ("Values:\n", "bold"),
        ("Strings", TYPE_STYLE_MAP[str]),
        (" / ", "default"),
        ("Numbers", TYPE_STYLE_MAP[(int, float)]),
        (" / ", "default"),
        ("Booleans", TYPE_STYLE_MAP[bool]),
        (" / ", "default"),
        ("None", NONE_STYLE),
    ]

    if not skip_unchanged:
        legend_style = "dim" if dim_unchanged else "default"
        legend_items.insert(6, ("Unchanged", legend_style))
        legend_items.insert(7, (" values\n", "default"))

    legend = Panel(
        Text.assemble(*legend_items),
        title="Guide",
        border_style="bright_blue",
    )

    # Print everything
    console.print("\n")
    console.print(Columns([diff_tree, legend]))
    console.print("\n")
