import re
import typer
from rich.console import Console
from rich.table import Table
from collections import Counter
from dataclasses import astuple

from pypoe.socket_calcs import Item, ChromaticCalculator
from pypoe.utils import RGB
from pypoe.body_armours import BASE_TYPES

# app = typer.Typer()


# @app.command()
def main(base: str, colors):
    item = BASE_TYPES[base.title()]
    print(
        f"You want to color a {base} which requires {str(item)} with colors {colors}."
    )
    calc = ChromaticCalculator(item, _parse_colors(colors))
    _print_rich_table(calc.compute_chances())


def _print_rich_table(list_results):
    table = Table(
        title="Chromatic cost",
        caption=(
            "Note on percentiles: the 90th percentile "
            "means you are expected to hit the desired colors "
            "with that number of tries or less, with 90% probability."
        ),
    )

    for col in vars(list_results[0]).keys():
        table.add_column(col)

    for result in list_results:
        table.add_row(*list(map(str, astuple(result))))

    console = Console()
    console.print(table)


def _parse_colors(colors: str):
    count = Counter(dict(zip("RGB", (0, 0, 0))))
    split_ = re.findall(pattern=r"(\d)([RGB])", string=colors)
    split_ = [(j, int(i)) for i, j in split_]
    count.update(dict(split_))

    return list(count.values())


if __name__ == "__main__":
    typer.run(main)
