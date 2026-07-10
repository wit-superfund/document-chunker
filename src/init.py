from rich.console import Console
from rich.markdown import Markdown
from rich_theme_manager import Theme, ThemeManager
import pathlib


def initialize(color: str = 'green'):
    theme_dir = pathlib.Path("themes")
    theme_manager = ThemeManager(theme_dir=str(theme_dir))
    dark = theme_manager.get("dark")

    # Create a console with the dark theme
    console = Console(theme=dark)
    return console
