import io
import os
import shlex
import shutil
import subprocess
import sys


def _theme(name: str = None) -> dict:
    r"""
    Return dictionary of colors.

    .. code-block:: python

        {
            'new' : '...',
            'overwrite' : '...',
            'skip' : '...',
            'bright' : '...',
        }

    :param name: Select color-theme [dark, none].
    :return: Dictionary of colors.
    """

    if name == "dark":
        return {
            "new": "1;32",
            "overwrite": "1;31",
            "skip": "1;30",
            "bright": "1;37",
        }

    return {
        "new": "",
        "overwrite": "",
        "skip": "",
        "bright": "",
    }


def _format(text: str, width: int = None, align: str = "<", color: str = None) -> str:
    r"""
    Format with color and alignment.

    :param text: The plain text.
    :param width: Print width.
    :param color: Print color, e.g. "1;32" for bold green.
    :param align: Print alignment.
    :return: Formatted string.
    """

    if width and color:
        fmt = "\x1b[{color:s}m{{0:{align:s}{width:d}.{width:d}s}}\x1b[0m".format(
            width=width, align=align, color=color
        )
    elif width:
        fmt = "{{0:{align:s}{width:d}.{width:d}s}}".format(width=width, align=align)
    elif color:
        fmt = f"\x1b[{color:s}m{{0:{align:s}s}}\x1b[0m"
    else:
        fmt = f"{{0:{align:s}s}}"

    return fmt.format(text)


def _page(text: str):
    """
    Display text in a terminal pager.
    Respects the PAGER environment variable if set.

    :param text: Text to display.
    """
    pager_cmd = shlex.split(os.environ.get("PAGER") or "less -r")
    subprocess.run(pager_cmd, input=text.encode("utf-8"))


def autoprint(text: str):
    """
    Print text to stdout.
    If the text is longer than the terminal height, it will be piped to a pager.
    """
    if sys.stdout.isatty():
        nlines = len(text.splitlines())
        _, term_lines = shutil.get_terminal_size()
        if nlines > term_lines:
            return _page(text)

    print(text)


def copyplan(
    status: dict[list[str]],
    colors: str = "none",
    display: bool = True,
    max_align: int = 80,
) -> str:
    """
    Print copy plan.

    :param status:
        Dictionary of copy status. E.g.::

            {
                '->' : ['file1', 'file2'],
                '!=' : ['file3'],
                '==' : ['file4'],
            }

    :param colors: Color theme name, see :py:func:`theme`.
    :param display: Display output (``False``: return as string).
    :param max_align: Maximum width of the first column.
    :return: Output string (if ``display=False``).
    """
    color = _theme(colors.lower())
    sio = io.StringIO()

    assert status.pop("<-", []) == [], "Cannot copy from destination to source"
    skip = status.pop("==", [])
    right = status.pop("->", [])
    overwrite = []
    for key in list(status.keys()):
        overwrite += status.pop(key, [])

    if len(overwrite) + len(right) + len(skip) == 0:
        return

    width = max(len(file) for file in overwrite + right + skip)
    width = min(width, max_align)

    for file in overwrite:
        print(
            "{:s} {:s} {:s}".format(
                _format(file, width=width, color=color["bright"]),
                _format("=>", color=color["bright"]),
                _format(file, color=color["overwrite"]),
            ),
            file=sio,
        )

    for file in right:
        print(
            "{:s} {:s} {:s}".format(
                _format(file, width=width, color=color["bright"]),
                _format("->", color=color["bright"]),
                _format(file, color=color["new"]),
            ),
            file=sio,
        )

    for file in skip:
        print(
            "{:s} {:s} {:s}".format(
                _format(file, width=width, color=color["skip"]),
                _format("==", color=color["skip"]),
                _format(file, color=color["skip"]),
            ),
            file=sio,
        )

    if not display:
        return sio.getvalue()

    autoprint(sio.getvalue())


def diff(
    status: dict[list[str]],
    colors: str = "none",
    display: bool = True,
    max_align: int = 80,
) -> str:
    """
    Print copy plan.

    :param status:
        Dictionary of status. E.g.::

            {
                '==' : ['file4'],
                '?=' : [],
                '!=' : ['file3'],
                '->' : ['file1', 'file2'],
                '<-' : [],
            }

    :param colors: Color theme name, see :py:func:`theme`.
    :param display: Display output (``False``: return as string).
    :param max_align: Maximum width of the first column.
    :return: Output string (if ``display=False``).
    """
    color = _theme(colors.lower())
    sio = io.StringIO()

    if isinstance(status, list):
        sio.write("\n".join(status))
        if not display:
            return sio.getvalue()
        else:
            autoprint(sio.getvalue())

    skip = status.pop("==", [])
    right = status.pop("->", [])
    left = status.pop("<-", [])
    ne = status.pop("!=", [])
    na = status.pop("?=", [])

    if len(ne) + len(na) + len(left) + len(right) + len(skip) == 0:
        return

    width = max(len(file) for file in ne + na + left + right + skip)
    width = min(width, max_align)

    for file in ne:
        print(
            "{:s} {:s} {:s}".format(
                _format(file, width=width, color=color["overwrite"]),
                _format("!=", color=color["bright"]),
                _format(file, color=color["overwrite"]),
            ),
            file=sio,
        )

    for file in na:
        print(
            "{:s} {:s} {:s}".format(
                _format(file, width=width, color=color["overwrite"]),
                _format("?=", color=color["bright"]),
                _format(file, color=color["overwrite"]),
            ),
            file=sio,
        )

    for file in left:
        print(
            "{:s} {:s} {:s}".format(
                _format(file, width=width, color=color["new"]),
                _format("<-", color=color["bright"]),
                _format(file, color=color["bright"]),
            ),
            file=sio,
        )

    for file in right:
        print(
            "{:s} {:s} {:s}".format(
                _format(file, width=width, color=color["bright"]),
                _format("->", color=color["bright"]),
                _format(file, color=color["new"]),
            ),
            file=sio,
        )

    for file in skip:
        print(
            "{:s} {:s} {:s}".format(
                _format(file, width=width, color=color["skip"]),
                _format("==", color=color["skip"]),
                _format(file, color=color["skip"]),
            ),
            file=sio,
        )

    if not display:
        return sio.getvalue()
    else:
        autoprint(sio.getvalue())
