import inspect
import sys
from pathlib import Path
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from rich.console import Console, Group
from rich.tree import Tree
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.columns import Columns

from .utils import format_type
from .. import utils


def create_aligned_tree_and_table(
    title: str, fields: dict[str, FieldInfo], sep=".", skip_parent_key: bool = False
) -> tuple[Tree, Table]:
    """Create a tree and table with aligned rows."""
    # Create tree
    root = Tree("")
    field_map = {}  # Map of paths to tree nodes

    # First pass: create tree structure and track line numbers
    line_to_field: dict[int, tuple[str, FieldInfo]] = (
        {}
    )  # Track which field appears on which line
    current_line = 0

    for key, field in fields.items():
        parts = key.split(sep)
        current = root

        for i, part in enumerate(parts):
            path = ".".join(parts[: i + 1])
            if path not in field_map:
                style = "yellow" if field.is_required() else "bright white"
                if i == len(parts) - 1:  # Leaf node
                    label = Text(f"{part}", style=style)
                    line_to_field[current_line] = (key, field)
                else:
                    label = Text(part, style=style)
                    current_line += 1  # Add empty line for guide lines

                node = current.add(label)
                field_map[path] = node
                current_line += 1
            current = field_map[path]

    has_desc = any(f.description for f in fields.values())

    # Create table with proper spacing
    table = Table(show_header=True, box=None, padding=(0, 1), collapse_padding=True)
    table.add_column("Field", style="magenta")
    table.add_column("Default", style="yellow")
    table.add_column("Type", max_width=30 if has_desc else 60, no_wrap=True)
    if has_desc:
        table.add_column(
            "Description", style="white", max_width=30 if has_desc else 60, no_wrap=True
        )

    # Second pass: add aligned table rows
    for line in range(current_line):
        if line in line_to_field:
            key, field = line_to_field[line]
            # Style field name based on required status
            name_style = "bold magenta" if field.is_required else "magenta"
            if skip_parent_key:
                keys = key.split(sep)
                key_str = (
                    "  " * (len(keys) - 1) + "." + keys[-1]
                    if len(keys) > 1
                    else "--" + key
                )
            else:
                key_str = "--" + key
            field_name = Text(key_str, style=name_style)

            # Style type
            type_text = format_type(field.annotation)

            # Format status
            if utils.is_base_model(field.annotation):
                status = ""
            else:
                if field.is_required():
                    status = Text("Required", style="yellow bold")
                else:
                    status = Text(f"{field.default}", style="yellow")

            rows = [field_name, status, type_text]
            if has_desc:
                rows.append(field.description)

            # Add row with field information
            table.add_row(*rows)
        else:
            # Add empty row to maintain alignment
            rows = [None] * 4 if has_desc else 3
            table.add_row(*rows)

    return Group(Text(title, style="bold"), *root.children), table


def create_extra(fields: dict[str, FieldInfo]):
    table = Table("Field", "Type", "Description", box=None)
    for k, f in fields.items():
        text = format_type(f.annotation)
        desc = Text(f.description if f.description is not None else "")
        if text.cell_len + desc.cell_len <= 60:
            continue
        table.add_row(Text(f"--{k}", style="magenta"), text, desc)

    return Panel(table, title="Extras", border_style="blue")


def print_help(
    model_class: type[BaseModel],
    underscore_to_hyphen: bool = False,
    sep=".",
    console: Console | None = None,
    *,
    info_panel: bool = False,
    skip_parent_key: bool = False,
    extra_panel: bool = True,
):
    console = console or Console()

    if info_panel:
        # Get model location information
        try:
            module_path = Path(inspect.getfile(model_class)).resolve()
            line_number = inspect.getsourcelines(model_class)[1]
            location_info = f"[dim]Defined in {module_path}:{line_number}[/dim]"
        except (TypeError, OSError):
            location_info = "[dim]Location information unavailable[/dim]"

        # Print model information
        model_doc = inspect.getdoc(model_class) or "No description available"
        info_panel = Panel.fit(
            f"{model_doc}\n\n{location_info}",
            title=model_class.__name__,
            border_style="blue",
        )
        console.print(info_panel)

    # Print usage
    console.print(f"\n[bold]Usage:[/bold] {sys.argv[0]} [OPTIONS]")

    # Get and print field tree
    fields = utils.get_field_info(
        model_class, sep=sep, underscore_to_hyphen=underscore_to_hyphen
    )
    tree, table = create_aligned_tree_and_table(
        model_class.__name__, fields, skip_parent_key=skip_parent_key
    )

    # Create panels
    tree_panel = Panel(tree, title="Structure", border_style="blue", padding=0)
    table_panel = Panel(table, title="Details", border_style="blue", padding=0)

    columns = Columns([tree_panel, table_panel], expand=True, padding=0)
    console.print(columns)

    if extra_panel:
        extra_panel = create_extra(fields)
        console.print(extra_panel)
