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

"""Example of how markdown can be used with rich"""
md_examples = """
# This is a h1
## This is a h2
### This is a h3
#### This is a h4
##### This is a h5
###### This is a h6

* Item 1
* Item 2
* Item 3
    * Item 3.1

> This is a blockquote

| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
| Cell 3   | Cell 4   |

```python
print("Hello, World!")
```

**Bold text**

*Italic text*

---

**Bold text**

***Bold and italic text***

--- 


"""
# console.print(Markdown(md_examples))  