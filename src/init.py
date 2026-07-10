from rich.console import Console
from rich.theme import Theme


def initialize(color: str = 'green'):
    custom_theme = Theme({
        "default": "bright_white on black",
        "repr.attrib_name": "bold #e87d3e",
        "repr.attrib_value": "bright_blue",
        "repr.call": "bright_yellow",
        "repr.none": "dim white",
        "repr.number": "bright_red",
        "repr.own": "bold #e87d3e",
        "repr.str": "bright_green",
        "repr.tag_name": "dim cyan"
    })
    return Console(theme=custom_theme)

