r"""
Copy & query using *rsync*.

(c) Tom de Geus, 2021, MIT
"""

import os
import re
import subprocess
import tempfile

import numpy as np
import tqdm

from .external import exec_cmd


def copy(
    source_dir: str,
    dest_dir: str,
    files: list[str],
    options: str = "-a",
    verbose: bool = False,
    progress: bool = True,
):
    """
    Copy files using *rsync*.
    This a wrapper around ``rsync {options:s} --files-from``.

    :param source_dir: Source directory. If remote: ``[user@]host:path``.
    :param dest_dir: Source directory. If remote: ``[user@]host:path``.
    :param files: List of file-paths (relative to ``source_dir`` and ``dest_dir``).
    :param options: Options passed to ``rsync``.
    :param verbose: Verbose commands.
    :param progress: Show progress bar.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "rsync.txt")

        with open(temp_file, "w") as file:
            file.write("\n".join(files))

        if verbose:
            print("\n".join(files))

        # Run without printing output

        if not progress:
            cmd = 'rsync {options:s} --files-from="{files:s}" "{src:s}" "{dest:s}"'.format(
                options=options, src=str(source_dir), dest=str(dest_dir), files=temp_file
            )
            return exec_cmd(cmd, verbose)

        # Run while printing output

        cmd = 'rsync {options:s} -P --files-from="{files:s}" "{src:s}" "{dest:s}"'.format(
            options=options, src=str(source_dir), dest=str(dest_dir), files=temp_file
        )

        if verbose:
            print(cmd)

        pbar = tqdm.tqdm(total=len(files))
        sbar = tqdm.tqdm(unit="B", unit_scale=True)

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

        for line in iter(process.stdout.readline, b""):
            line = line.decode("utf-8")
            if re.match(r"(.*)(xfe?r\#[0-9])(.*)(to\-che?c?k\=[0-9])(.*)", line):
                e = int(list(filter(None, line.splitlines()[-1].split(" ")))[-6].replace(",", ""))
                pbar.update()
                sbar.update(e)


def diff(
    source_dir: str,
    dest_dir: str,
    files: list[str],
    options: str = "-nai",
    verbose: bool = False,
) -> dict[list[str]]:
    """
    Check if files are different using *rsync*.

    .. note::

        *rsync* uses basic criteria such as file size and creation and modification date.
        This is much faster than using checksums but is only approximate.
        See `rsync manual <https://www.samba.org/ftp/rsync/rsync.html>`_.

    :param str source_dir: Source directory (optionally with hostname).
    :param str dest_dir: Source directory (optionally with hostname).
    :param list files: List of file-paths (relative to ``source_dir`` and ``dest_dir``).
    :param verbose: Verbose commands.
    :return:
        Dictionary with differences::

            {
                "==" : [ ... ], # equal
                "!=" : [ ... ], # not equal
                "->" : [ ... ], # in source_dir not in dest_dir
            }
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "rsync.txt")
        files = [os.path.normpath(file) for file in files]

        with open(temp_file, "w") as file:
            file.write("\n".join(files))

        if verbose:
            print("\n".join(files))

        # Run without printing output

        cmd = 'rsync {options:s} --files-from="{files:s}" "{src:s}" "{dest:s}"'.format(
            src=str(source_dir), dest=str(dest_dir), files=temp_file, options=options
        )

        lines = list(filter(None, exec_cmd(cmd, verbose).split("\n")))
        lines = [line for line in lines if line[1] in ["f", "L"]]

        if len(lines) == 0:
            return {
                "==": files,
                "!=": [],
                "->": [],
            }

        check_paths = []
        for line in lines:
            if line[1] == "f":
                check_paths.append(line.split(" ", 1)[1])
            elif line[:2] == "cL":
                check_paths.append(line.split(" ", 1)[1].split(" -> ", 1)[0])

        mode = np.zeros((len(check_paths)), dtype=np.int16)
        modes = {"==": 0, "!=": 1, "->": 2, "<-": 3}

        for i, line in enumerate(lines):
            if line[0] == ">" or line[0] == "<":
                if line[2] == "+":
                    mode[i] = modes["->"]  # create
                else:
                    mode[i] = modes["!="]  # overwrite
            elif line[0] == "c" or line[1] == "L":
                mode[i] = modes["->"]  # create
            elif line[0] == ".":
                pass
            else:
                raise OSError(f'Unknown cryptic output "{line:s}"')

        sorter = np.argsort(files)
        source_paths = np.array(files, dtype=str)[sorter]

        i = np.argsort(check_paths)
        check_paths = np.array(check_paths, dtype=str)[i]
        mode = mode[i]

        test = np.in1d(source_paths, check_paths)

        idx = np.searchsorted(check_paths, source_paths)
        idx = np.where(test, idx, 0)
        ret = np.where(test, mode[idx], 0)
        ret = ret.astype(np.int16)
        out = np.empty_like(ret)
        out[sorter] = ret

        files = np.array(files)

        return {
            "==": files[out == modes["=="]].tolist(),
            "->": files[out == modes["->"]].tolist(),
            "!=": files[out == modes["!="]].tolist(),
        }
