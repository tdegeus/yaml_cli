import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
import textwrap

import click
import numpy as np
import prettytable

from . import convert
from . import dataset
from . import info
from . import local
from . import output
from . import path
from . import rsync
from . import scp
from . import search
from . import ssh
from . import yaml
from ._version import version

# set filename defaults
f_hostinfo = "shelephant_hostinfo.yaml"
f_dump = "shelephant_dump.yaml"


def _shelephant_parse_parser():
    """
    Return parser for :py:func:`shelephant_parse`.
    """

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    desc = "Parse a YAML-file, and print to screen."
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=pathlib.Path, help="File path.")
    return parser


def shelephant_parse(args: list[str]):
    """
    Command-line tool, see ``--help``.
    :param args: Command-line arguments (should be all strings).
    """

    parser = _shelephant_parse_parser()
    args = parser.parse_args(args)
    data = yaml.read(args.file)
    yaml.preview(data)


def _shelephant_dump_parser():
    """
    Return parser for :py:func:`shelephant_dump`.
    """

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    desc = textwrap.dedent(
        """\
    Dump filenames ((relative) paths) to a YAML-file.

    .. note::

        If you have too many arguments you can hit the pipe-limit. In that case, use ``xargs``:

        .. code-block:: bash

            find . -name "*.py" | xargs shelephant_dump -o dump.yaml

        or you can use ``--command`` such that *shelephant* executes the command for you:

        .. code-block:: bash

            shelephant_dump -o dump.yaml --command find . -name '*.py'
    """
    )

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)

    parser.add_argument(
        "-o", "--output", type=pathlib.Path, default=f_dump, help="Output YAML-file."
    )
    parser.add_argument("-a", "--append", action="store_true", help="Append existing file.")
    parser.add_argument(
        "-i",
        "--info",
        action="store_true",
        help="Add information (sha256, size).",
    )
    parser.add_argument(
        "-e", "--exclude", type=str, action="append", help="Exclude input matching this pattern."
    )
    parser.add_argument(
        "-E",
        "--exclude-extension",
        type=str,
        action="append",
        default=[],
        help='Exclude input with this extension (e.g. ".bak").',
    )
    parser.add_argument("--fmt", type=str, help='Formatter of each line, e.g. ``"mycmd {}"``.')
    parser.add_argument(
        "-c",
        "--command",
        action="store_true",
        help="Interpret arguments as a command (instead of as filenames) an run it.",
    )
    parser.add_argument(
        "-k", "--keep", type=str, action="append", help="Keep only input matching this regex."
    )
    parser.add_argument("--cwd", type=str, help="Directory to run the command in.")
    parser.add_argument(
        "--root", type=str, help="Root for relative paths (default: directory of output file)."
    )
    parser.add_argument("--abspath", action="store_true", help="Store as absolute paths.")
    parser.add_argument("-s", "--sort", action="store_true", help="Sort filenames.")
    parser.add_argument(
        "-f", "--force", action="store_true", help="Overwrite output file without prompt."
    )
    parser.add_argument("-v", "--version", action="version", version=version, help="")
    parser.add_argument("files", type=str, nargs="+", help="Filenames.")

    return parser


def shelephant_dump(args: list[str]):
    """
    Command-line tool, see ``--help``.
    :param args: Command-line arguments (should be all strings).
    """

    parser = _shelephant_dump_parser()
    args = parser.parse_args(args)
    files = args.files

    if args.root:
        root = args.root
    else:
        root = args.output.parent

    if args.command:
        cmd = " ".join(files)
        files = subprocess.check_output(cmd, shell=True, cwd=args.cwd).decode("utf-8").split("\n")
        files = list(filter(None, files))
        if args.cwd is not None:
            files = [os.path.join(args.cwd, file) for file in files]

    if args.abspath:
        files = [os.path.abspath(file) for file in files]
    else:
        files = [os.path.relpath(file, root) for file in files]

    if args.keep:
        ret = []
        for pattern in args.keep:
            ret += [file for file in files if re.match(pattern, file)]
        files = ret

    if args.exclude:
        excl = np.zeros(len(files), dtype=bool)
        for pattern in args.exclude:
            excl = np.logical_or(excl, np.array([re.match(pattern, file) for file in files]))
        files = [file for file, ex in zip(files, excl) if not ex]

    if args.exclude_extension:
        files = [file for file in files if pathlib.Path(file).suffix not in args.exclude_extension]

    if args.sort:
        files = sorted(files)

    if args.fmt:
        files = [args.fmt.format(file) for file in files]

    if args.info:
        files = dataset.Location(root=".", files=files).getinfo().files(info=True)

    if args.append:
        main = yaml.read(args.output)
        assert type(main) == list, 'Can only append a "flat" file'
        files = main + files
        args.force = True

    yaml.dump(args.output, files, args.force)


def _shelephant_cp_parser():
    """
    Return parser for :py:func:`shelephant_cp`.
    """

    desc = textwrap.dedent(
        """
        Copy files listed in a (field of a) YAML-file.
        These filenames are assumed either relative to the YAML-file or absolute.

        Usage::

            shelephant_cp <sourceinfo.yaml> <dest_dirname>
            shelephant_cp <sourceinfo.yaml> <dest_dirname_on_host> --ssh <user@host>
            shelephant_cp <sourceinfo.yaml> <destinfo.yaml>
        """
    )

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)
    parser.add_argument("--ssh", type=str, help="Remote SSH host (e.g. user@host).")
    parser.add_argument("--colors", type=str, default="dark", help="Color scheme [none, dark].")
    parser.add_argument("--sha256", action="store_true", help="Only use sha256 to check equality.")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite without prompt.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Do not print progress.")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Print copy-plan and exit.")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("source", type=pathlib.Path, help="Source information.")
    parser.add_argument("dest", type=pathlib.Path, help="Destination directory/information.")
    return parser


def shelephant_cp(args: list[str]):
    """
    Command-line tool, see ``--help``.
    :param args: Command-line arguments (should be all strings).
    """

    parser = _shelephant_cp_parser()
    args = parser.parse_args(args)
    assert args.source.is_file(), "Source must be a file"
    assert not args.force if args.dry_run else True, "Cannot use --force with --dry-run"

    source = dataset.Location.from_yaml(args.source)
    if args.dest.is_file():
        dest = dataset.Location.from_yaml(args.dest)
    else:
        dest = dataset.Location(root=args.dest, ssh=args.ssh)

    files = source.files(info=False)
    equal = source.diff(dest)["=="]
    [files.remove(file) for file in equal]  # based on sha256

    source = source.hostpath
    dest = dest.hostpath
    has_rsync = shutil.which("rsync") is not None

    if has_rsync and not args.sha256:
        status = rsync.diff(source, dest, files)
        eq = status.pop("==", [])
        [files.remove(file) for file in eq]  # based on rsync criteria
        equal += eq
    else:
        status = local.diff(source, dest, files)

    assert status.pop("<-", []) == [], "Cannot copy from destination to source"

    if len(files) == 0:
        print("Nothing to copy" if len(equal) == 0 else "All files equal")
        return

    if not args.force:
        status["=="] = equal
        output.copyplan(status, colors=args.colors)
        if args.dry_run:
            return
        if not click.confirm("Proceed?"):
            raise OSError("Cancelled")

    if has_rsync:
        rsync.copy(source, dest, files, progress=not args.quiet)
    else:
        local.copy(source, dest, files, progress=not args.quiet)


def _shelephant_mv_parser():
    """
    Return parser for :py:func:`shelephant_mv`.
    """

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    desc = textwrap.dedent(
        """
        Move files listed in a (field of a) YAML-file.
        These filenames are assumed either relative to the YAML-file or absolute.

        Usage::

            shelephant_mv <sourceinfo.yaml> <dest_dirname>
        """
    )

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)
    parser.add_argument("--colors", type=str, default="dark", help="Color scheme [none, dark].")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite without prompt.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Do not print progress.")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Print copy-plan and exit.")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("source", type=pathlib.Path, help="Source information.")
    parser.add_argument("dest", type=pathlib.Path, help="Destination directory.")
    return parser


def shelephant_mv(args: list[str]):
    """
    Command-line tool, see ``--help``.
    :param args: Command-line arguments (should be all strings).
    """

    parser = _shelephant_mv_parser()
    args = parser.parse_args(args)
    assert args.source.is_file(), "Source must be a file"
    assert args.dest.is_dir(), "Destination must be a directory"
    assert not args.force if args.dry_run else True, "Cannot use --force with --dry-run"

    source = dataset.Location.from_yaml(args.source)
    assert source.ssh is None, "Cannot move from remote"
    files = source.files(info=False)
    source = source.root
    dest = args.dest
    status = local.diff(source, dest, files)
    assert status.pop("<-", []) == [], "Cannot move from destination to source"

    if len(files) == 0:
        print("Nothing to move")
        return

    if not args.force:
        output.copyplan(status, colors=args.colors)
        if args.dry_run:
            return
        if not click.confirm("Proceed?"):
            raise OSError("Cancelled")

    local.move(source, dest, files, progress=not args.quiet)


def _shelephant_rm_parser():
    """
    Return parser for :py:func:`shelephant_rm`.
    """

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    desc = textwrap.dedent(
        """\
    Remove files listed in a (field of a) YAML-file.
    These filenames are assumed either relative to the YAML-file or absolute.

    Usage::

        shelephant_rm <sourceinfo.yaml>
    """
    )

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)
    parser.add_argument("--colors", type=str, default="dark", help="Color scheme [none, dark].")
    parser.add_argument("-f", "--force", action="store_true", help="Remove without prompt.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Do not print progress.")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Print copy-plan and exit.")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("source", type=pathlib.Path, help="Source information.")
    return parser


def shelephant_rm(args: list[str]):
    """
    Command-line tool, see ``--help``.
    :param args: Command-line arguments (should be all strings).
    """

    parser = _shelephant_rm_parser()
    args = parser.parse_args(args)
    assert args.source.is_file(), "Source must be a file"
    assert not args.force if args.dry_run else True, "Cannot use --force with --dry-run"

    source = dataset.Location.from_yaml(args.source)
    assert source.ssh is None, "Cannot move from remote"
    files = source.files(info=False)
    source = source.root

    if len(files) == 0:
        print("Nothing to remove")
        return

    if not args.force:
        for file in files:
            print(f"rm {file:s}")
        if args.dry_run:
            return
        if not click.confirm("Proceed?"):
            raise OSError("Cancelled")

    local.remove(source, files, progress=not args.quiet)


def _shelephant_hostinfo_parser():
    """
    Return parser for :py:func:`shelephant_hostinfo`.
    """

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    desc = textwrap.dedent(
        """\
    Collect information about a remote directory (on a remote SSH host).
    This information is stored in a YAML-file (default: ``shelephant_hostinfo.yaml``) as follows::

        root: <path>      # relative to the YAML-file, or absolute
        ssh: <user@host>  # (optional) remote SSH host
        dump: <dump>      # (optional, excludes "search") path from which a list of files is read
        search:           # (optional, excludes "dump") search information, must be set by hand
            - ...
        files:            # (optional) list of files (from "search" / "dump", or set by hand)
            - ...

    Usage:

    1.  Create *hostinfo*::

            shelephant_hostinfo <path>
            shelephant_hostinfo <path> --ssh <user@host>
            shelephant_hostinfo <path> --dump [shelephant_dump.yaml]
            shelephant_hostinfo <path> --dump [shelephant_dump.yaml] --ssh <user@host>

    2.  Update *hostinfo*::

            shelephant_hostinfo --update [shelephant_hostinfo.yaml]
    """
    )

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)
    parser.add_argument(
        "-o", "--output", type=pathlib.Path, default=f_hostinfo, help="Output YAML-file."
    )
    parser.add_argument(
        "-d",
        "--dump",
        type=pathlib.Path,
        default=None,
        nargs="?",
        const=f_dump,
        help="YAML-file containing a list of files.",
    )
    parser.add_argument("--ssh", type=str, help="Remote SSH host (e.g. user@host).")
    parser.add_argument(
        "--update", action="store_true", help='Update "files" based on "dump" or "search".'
    )
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output.")
    parser.add_argument("--version", action="version", version=version)
    parser.add_argument("path", type=pathlib.Path, help="Path to remote directory.")
    return parser


def shelephant_hostinfo(args: list[str]):
    """
    Command-line tool, see ``--help``.
    :param args: Command-line arguments (should be all strings).
    """

    parser = _shelephant_hostinfo_parser()
    args = parser.parse_args(args)

    if args.update:
        assert args.output == f_hostinfo, "No customisation allowed in --update mode."
        assert args.dump is None, "No customisation allowed in --update mode."
        assert args.ssh is None, "No customisation allowed in --update mode."
        loc = dataset.Location.from_yaml(args.path)
        args.output = args.path
        args.force = True
    else:
        loc = dataset.Location(root=args.path, ssh=args.ssh)
        if args.dump:
            loc.dump = args.dump

    loc.read()
    loc.to_yaml(args.output, force=args.force)


def _shelephant_diff_parser():
    """
    Return parser for :py:func:`shelephant_diff`.
    """

    desc = textwrap.dedent(
        """
        Compare local and remote files and list differences.

        Usage::

            # compare file existence (and equality if rsync is available)
            # requires source and dest to be available
            shelephant_diff <sourceinfo.yaml> <dest_dirname>

            # compare files based on precomputed information
            # no live check is performed
            shelephant_diff <sourceinfo.yaml> <destinfo.yaml>

            # different output
            shelephant_diff <sourceinfo.yaml> <destinfo.yaml> --filter "->"
            shelephant_diff <sourceinfo.yaml> <destinfo.yaml> --filter "?=, !="
            shelephant_diff <sourceinfo.yaml> <destinfo.yaml> -o <diff.yaml>

        Note that if filter contains only one operation the output YAML-file will be a list.
        """
    )

    desc = ""

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)
    parser.add_argument("--ssh", type=str, help="Remote SSH host (e.g. user@host).")
    parser.add_argument("--colors", type=str, default="dark", help="Color scheme [none, dark].")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Dump as YAML file.")
    parser.add_argument("--sort", type=str, help="Sort printed table by column.")
    parser.add_argument("--table", type=str, default="SINGLE_BORDER", help="Select print style.")
    parser.add_argument("--filter", type=str, help="Filter to direction (separated by ',').")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output.")
    parser.add_argument("--version", action="version", version=version)
    parser.add_argument("source", type=pathlib.Path, help="Source information.")
    parser.add_argument("dest", type=pathlib.Path, help="Destination directory/information.")
    return parser


def shelephant_diff(args: list[str]):
    """
    Command-line tool to print datasets from a file, see ``--help``.
    :param args: Command-line arguments (should be all strings).
    """

    has_rsync = shutil.which("rsync") is not None
    parser = _shelephant_diff_parser()
    args = parser.parse_args(args)
    assert args.source.is_file(), "Source must be a file"

    source = dataset.Location.from_yaml(args.source)
    if args.dest.is_file():
        dest = dataset.Location.from_yaml(args.dest)
        status = source.diff(dest)
    else:
        dest = dataset.Location(root=args.dest, ssh=args.ssh)
        files = source.files(info=False)
        if has_rsync:
            status = rsync.diff(source.hostpath, dest.hostpath, files)
        else:
            assert source.ssh is None, "Specify sourceinfo, or install rsync"
            assert dest.ssh is None, "Specify destinfo, or install rsync"
            status = local.diff(source.hostpath, dest.hostpath, files)

    for key in list(status.keys()):
        if len(status[key]) == 0:
            del status[key]

    if args.filter:
        keys = [key.strip() for key in args.filter.split(",")]
        status = {key: status[key] for key in keys}

    if args.output:
        if len(status) == 1:
            status = status[list(status.keys())[0]]
        yaml.dump(args.output, status, force=args.force)
        return

    out = prettytable.PrettyTable()
    if args.table == "PLAIN_COLUMNS":
        out.set_style(prettytable.PLAIN_COLUMNS)
    elif args.table == "SINGLE_BORDER":
        out.set_style(prettytable.SINGLE_BORDER)
    out.field_names = ["source", "sync", "dest"]
    out.align["source"] = "l"
    out.align["sync"] = "c"
    out.align["dest"] = "l"

    left = status.pop("->", [])
    right = status.pop("<-", [])
    equal = status.pop("==", [])

    for key in status:
        for item in status[key]:
            out.add_row([item, key, item])

    for item in left:
        out.add_row([item, "->", ""])

    for item in right:
        out.add_row(["", "<-", item])

    for item in equal:
        out.add_row([item, "==", item])

    if args.sort is None:
        print(out.get_string())
    else:
        print(out.get_string(sortby=args.sort))


def _shelephant_parse_main():
    shelephant_parse(sys.argv[1:])


def _shelephant_cp_main():
    shelephant_cp(sys.argv[1:])


def _shelephant_mv_main():
    shelephant_mv(sys.argv[1:])


def _shelephant_rm_main():
    shelephant_rm(sys.argv[1:])


def _shelephant_dump_main():
    shelephant_dump(sys.argv[1:])


def _shelephant_hostinfo_main():
    shelephant_hostinfo(sys.argv[1:])


def _shelephant_diff_main():
    shelephant_diff(sys.argv[1:])
