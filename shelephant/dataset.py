import argparse
import json
import os
import pathlib
import re
import shutil
import textwrap
from copy import deepcopy

import click
import numpy as np
import prettytable
import tqdm
from platformdirs import user_cache_dir

from . import cli
from . import compute_hash
from . import output
from . import path as mypathlib
from . import rsync
from . import scp
from . import search
from . import ssh
from . import yaml
from ._version import version
from .external import exec_cmd

if shutil.which("rsync") is not None:
    _copyfunc = rsync.copy
else:
    _copyfunc = scp.copy


def _force_absolute_path(root: pathlib.Path, path: pathlib.Path) -> pathlib.Path:
    """
    Force a path to be absolute.

    :param root: A base directory.
    :param path: The path that may be absolute or relative to ``root``.
    :return: The absolute ``path``.
    """
    if path.is_absolute():
        return path
    return pathlib.Path(os.path.normpath(root.absolute() / path))


class Location:
    """
    Location information.

    Attributes:

    *   :py:attr:`Location.root`: The base directory.
    *   :py:attr:`Location.ssh` (optional): ``[user@]host``
    *   :py:attr:`Location.hostpath`.
    *   :py:attr:`Location.prefix` (optional): Prefix to add to all paths.
    *   :py:attr:`Location.python` (optional): The python executable on the ``ssh`` host.
    *   :py:attr:`Location.dump` (optional): Location of "dump" file -- file with list of files.
    *   :py:attr:`Location.search` (optional):
        Commands to search for files, see :py:func:`shelephant.search.search`.
    *   :py:func:`Location.files`: List of files (with properties).
    *   :py:attr:`Location.description`: Description of the location.

    Initialize:

    *   Read from yaml file::

            location = Location.from_yaml("location.yaml")

    *   Create from scratch::

            location = Location(root="~/data", ...)
    """

    def __init__(
        self,
        root: str | pathlib.Path,
        ssh: str = None,
        mount: pathlib.Path = None,
        prefix: pathlib.Path = None,
        files: list[str] = [],
        description: str = None,
    ) -> None:
        """
        :param root: The root directory (may be relative, unless on remote SSH host).
        :param ssh: ``[user@]host``.
        :param mount: Mount location for SSH host.
        :param prefix: Prefix to add to all paths.
        :param files: List of files.
        :param description: Description of the location.
        """
        self.root = pathlib.Path(root)
        if mount is None:
            self.mount = None
            self._absroot_is_mount = False
            self._absroot = self.root.absolute()
        else:
            self.mount = pathlib.Path(mount)
            self._absroot_is_mount = True
            self._absroot = self.mount.absolute()
        self.prefix = pathlib.Path(prefix) if prefix is not None else None
        self.ssh = ssh
        self.python = "python3"
        self.dump = None
        self.search = None
        self.description = description

        assert not self._absroot_is_mount or self.ssh is not None, "needs ssh to use mount"

        if ssh is not None:
            assert self.root.is_absolute(), "root must be absolute path when using ssh"

        if isinstance(files, list):
            self._files = np.array(files)
            self._clear_all_info()
        elif isinstance(files, str):
            self._files = np.array([files])
            self._clear_all_info()
        elif isinstance(files, dict):
            self._files = np.array(list(files.keys()))
            self._clear_all_info()
            for i, file in enumerate(self._files):
                self._has_info[i] = "sha256" in files[file]
                if self._has_info[i]:
                    self._sha256[i] = files[file]["sha256"]
                    self._size[i] = files[file]["size"]
                    self._mtime[i] = files[file]["mtime"]
        else:
            raise TypeError(f"Unknown type of files: {type(files)}")

        assert np.unique(self._files).size == self._files.size, "duplicate filenames"

    def _add_suffix(self, suffix: pathlib.Path):
        """
        Add a path "suffix" to the root.

        :param suffix: Path to add to the root.
        """
        self.root = self.root / suffix
        self._absroot = self._absroot / suffix
        return self

    def _slice(self, keep):
        """
        Slice list of files and info.

        :param keep: Slice.
        """
        self._files = self._files[keep]
        self._has_info = self._has_info[keep]
        self._sha256 = self._sha256[keep]
        self._size = self._size[keep]
        self._mtime = self._mtime[keep]
        return self

    def _clear_all_info(self):
        """
        Clear all info.
        """
        self._has_info = np.zeros(self._files.size, dtype=bool)
        self._sha256 = np.empty(self._files.size, dtype="U64")
        self._size = np.zeros(self._files.size, dtype=np.int64)
        self._mtime = np.empty(self._files.size, dtype=np.float64)
        return self

    def _append(self, files: list[str]):
        """
        Extend list of files.

        .. warning::

            ``files`` is assumed to be unique and not part of ``self._files``.
            There is no check on this.

        :param files: List of files.
        """
        files = np.array(files)
        n = files.size
        if n == 0:
            return self
        self._files = np.hstack((self._files, files))
        self._has_info = np.hstack((self._has_info, np.zeros(n, dtype=self._has_info.dtype)))
        self._sha256 = np.hstack((self._sha256, np.empty(n, dtype=self._sha256.dtype)))
        self._size = np.hstack((self._size, np.empty(n, dtype=self._size.dtype)))
        self._mtime = np.hstack((self._mtime, np.empty(n, dtype=self._mtime.dtype)))
        if self.search is not None:
            return self.sort()
        return self

    def _prune(self, files: list[str]):
        """
        Overwrite database with list of files.
        If files are already in the database, the sha256/size/mtime is kept.
        For new files the sha256/size/mtime is set to None.
        (Files that were in the database but that are not in ``files`` are removed.)

        :param files: List of files (unique, no assertion).
        """
        files = np.array(files)

        if self._files.size == files.size:
            if np.all(np.equal(self._files, files)):
                return self

        _, i, j = np.intersect1d(self._files, files, return_indices=True, assume_unique=True)
        k = np.ones(files.size, dtype=bool)
        k[j] = False
        self._slice(i)
        self._append(files[k])
        return self

    def _overwrite_dataset_from_dict(self, files: list):
        """
        Read files from list of strings or dictionaries (as stored in a YAML file).
        This overwrites the current database (including all sha256/size/mtime), with information
        present in the input.

        :param files: List of files.
        """

        fs = []
        has_info = []
        sha256 = []
        size = []
        mtime = []

        for item in files:
            if isinstance(item, str):
                fs.append(item)
                has_info.append(False)
                sha256.append("0" * 64)
                size.append(0)
                mtime.append(0)
            else:
                fs.append(item["path"])
                has_info.append(True)
                sha256.append(item["sha256"])
                size.append(item["size"])
                mtime.append(item["mtime"])

        self._files = np.array(fs, dtype=object)
        self._has_info = np.array(has_info, dtype=bool)
        self._sha256 = np.array(sha256, dtype="U64")
        self._size = np.array(size, dtype=np.int64)
        self._mtime = np.array(mtime, dtype=np.float64)
        return self

    def __eq__(self, other):
        """
        Check if all files and information are equal (mtime is allowed to be different).

        :param other: Other location.
        :return: True if equal, False otherwise.
        """
        if self._files.size != other._files.size:
            return False

        a = np.argsort(self._files)
        b = np.argsort(other._files)

        return (
            np.all(np.equal(self._files[a], other._files[b]))
            and np.all(np.equal(self._sha256[a], other._sha256[b]))
            and np.all(np.equal(self._size[a], other._size[b]))
            and np.all(np.equal(self._has_info[a], other._has_info[b]))
        )

    def __iadd__(self, other):
        """
        Add files from other location.

        .. todo::

            In case of duplicate files: check that the checksums and sizes are the same.
        """
        assert self.root == other.root, "root must be equal"
        assert self.ssh == other.ssh, "ssh must be equal"
        self._files = np.hstack((self._files, other._files))
        self._has_info = np.hstack((self._has_info, other._has_info))
        self._sha256 = np.hstack((self._sha256, other._sha256))
        self._size = np.hstack((self._size, other._size))
        self._mtime = np.hstack((self._mtime, other._mtime))
        self.search = None
        self.dump = None
        return self._unique()

    def __add__(self, other):
        """
        Add files from two locations.
        """
        ret = deepcopy(self)
        ret += other
        return ret._unique()

    def copy_files(self, other):
        """
        Copy files (and size/mtime/sha256) from other location.

        :param other: Other location.
        """
        self._files = other._files
        self._has_info = other._has_info
        self._sha256 = other._sha256
        self._size = other._size
        self._mtime = other._mtime
        return self

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path):
        """
        Read from yaml file.

        :param path: Path to yaml file.
        :return: Location.
        """

        path = pathlib.Path(path)
        data = yaml.read(path)
        if isinstance(data, list):
            data = {"files": data}

        ret = cls(
            root=data.get("root", path.parent),
            ssh=data.get("ssh", None),
            prefix=data.get("prefix", None),
            mount=data.get("mount", None),
            description=data.get("description", None),
        )
        if not ret._absroot_is_mount:
            ret._absroot = _force_absolute_path(path.parent, ret.root)
        else:
            ret._absroot = _force_absolute_path(path.parent, ret.mount)
        ret.dump = data.get("dump", None)
        ret.search = data.get("search", None)
        ret._overwrite_dataset_from_dict(data.get("files", []))
        if ret.search is not None:
            ret.sort()
        return ret

    def to_yaml(self, path: str | pathlib.Path, force: bool = False):
        """
        Write to yaml file.

        :param path: Path to yaml file.
        :param force: Do not prompt to overwrite file.
        """
        yaml.dump(path, self.asdict(), force=force)

    def overwrite_yaml(self, path: str | pathlib.Path):
        """
        Overwrite yaml file.
        This function only changes the file if the content has indeed changed.

        :param path: Path to yaml file.
        """
        yaml.overwrite(path, self.asdict())

    def asdict(self) -> dict:
        """
        Return as dictionary.

        :return: Dictionary.
        """

        ret = {"root": str(self.root)}

        if self.ssh is not None:
            ret["ssh"] = self.ssh

        if self.mount is not None:
            ret["mount"] = str(self.mount)

        if self.dump is not None:
            ret["dump"] = str(self.dump)

        if self.search is not None:
            ret["search"] = self.search

        if self.prefix is not None:
            ret["prefix"] = str(self.prefix)

        if len(self._files) > 0:
            ret["files"] = self.files(info=True)

        if self.description is not None:
            ret["description"] = self.description

        return ret

    def files(self, info: bool = True) -> list:
        """
        Return as list of files.
        Items without sha256 and size are returned as ``str``,
        items with sha256 and size are returned as ``dict``.

        :param info: Return sha256 and size (if available).
        :return: List of files.
        """

        if not info:
            return self._files.tolist()

        if not np.any(self._has_info):
            return self._files.tolist()

        ret = []

        for i, file in enumerate(self._files):
            if self._has_info[i]:
                ret += [
                    {
                        "path": str(file),
                        "sha256": str(self._sha256[i]),
                        "size": int(self._size[i]),
                        "mtime": float(self._mtime[i]),
                    }
                ]
            else:
                ret += [str(file)]

        return ret

    @property
    def hostpath(self) -> str:
        """
        Return:

        -   ``root`` if ``ssh`` is not set.
        -   ``ssh:"root"`` if ``ssh`` is set.
        """
        if self.ssh is None:
            return str(self._absroot)

        return f'{self.ssh:s}:"{str(self.root):s}"'

    def sort(self, key: str = "files"):
        """
        Sort files.

        :param key: Key to sort by (files, size, sha256).
        """

        if key == "files":
            return self._slice(np.argsort(self._files))
        if key == "size":
            return self._slice(np.argsort(self._size))
        if key == "mtime":
            return self._slice(np.argsort(self._mtime))
        if key == "sha256":
            return self._slice(np.argsort(self._sha256))

        raise ValueError(f"Unknown key: {key}")

    def _unique(self):
        """
        Remove duplicate filename.

        .. todo::

            Check that the checksums and sizes are the same.

        .. todo::

            Keep sorted?
        """
        _, idx = np.unique(self._files, return_index=True)
        return self._slice(idx)

    def isavailable(self, mount: bool = False) -> bool:
        """
        Check if location is available.

        :param mount: Check if mount is available.
        :return: True if available.
        """
        if self.ssh is None or mount:
            return self._absroot.is_dir()
        return ssh.is_dir(self.ssh, self.root)

    def is_mounted(self) -> bool:
        """
        Check if a location is a local directory, or if a remote directory is mounted.

        :return: True if mounted.
        """
        if self.ssh is not None and not self._absroot_is_mount:
            return False
        return self._absroot.is_dir()

    def remove(self, paths: list[str]):
        """
        Remove files from list of files.

        :param paths: List of paths to remove.
        """
        if isinstance(paths, str):
            paths = [paths]
        _, i, _ = np.intersect1d(
            self._files, np.unique(list(map(str, paths))), return_indices=True, assume_unique=True
        )
        k = np.ones(self._files.size, dtype=bool)
        k[i] = False
        return self._slice(k)

    def _read_impl(self, verbose: bool):
        """
        See :meth:`read`.
        """
        if self.dump is None and self.search is None:
            return self

        assert not (self.dump is not None and self.search is not None)

        # overwrite dataset with content of YAML dump file, and quit
        if self.dump is not None:
            if self.ssh is None:
                return self._overwrite_dataset_from_dict(yaml.read(self._absroot / self.dump))

            with mypathlib.tempdir():
                scp.copy(self.hostpath, ".", [self.dump], progress=False)
                return self._overwrite_dataset_from_dict(yaml.read(self.dump))

        # search locally for files (the sha256/size/mtime of 'new' files is set to None)
        if self.ssh is None:
            return self._prune(
                sorted(list(map(str, search.search(*self.search, root=self._absroot))))
            )

        # search on SSH remote host for files (the sha256/size/mtime of 'new' files is set to None)
        cache_dir = ssh._shelephant_cachdir(self.ssh, self.python)
        with ssh._cachedir(self.ssh, cache_dir) as remote, mypathlib.tempdir():
            shutil.copy(pathlib.Path(__file__).parent / "search.py", "script.py")
            with open("settings.json", "w") as f:
                json.dump(self.search, f)

            host = f'{self.ssh:s}:"{str(remote):s}"'
            _copyfunc(".", host, ["script.py", "settings.json"], progress=False, verbose=verbose)
            exec_cmd(
                f'ssh {self.ssh:s} "cd {str(remote)} && {self.python} script.py {str(self.root)}"',
                verbose=verbose,
            )
            _copyfunc(host, ".", ["files.txt"], progress=False, verbose=verbose)
            return self._prune(sorted(pathlib.Path("files.txt").read_text().splitlines()))

    def read(self, verbose: bool = False, getinfo: bool = False):
        """
        Read files from location.

        -   If ``dump`` is set, read from dump file.
            This overwrites the database
            (sha256/size/mtime will only be available if they are in the YAML file).

        -   If ``search`` is set, search for files.
            This preserves sha256/size/mtime if paths are already in the database
            (there is no check that they are still accurate).

        :param verbose: Print progress (only relevant if ``ssh`` is set).
        :param getinfo: Get sha256/size/mtime (calls ``getinfo``).
        """
        if getinfo:
            return self._read_impl(verbose).getinfo(verbose=verbose)

        return self._read_impl(verbose)

    def has_info(self) -> bool:
        """
        Check if sha256/size/mtime is available for all files.

        :return: True if sha256/size/mtime is available for all files.
        """
        return np.all(self._has_info)

    def _getindex(self, paths: list[str]) -> np.array:
        """
        Get index of paths in dataset.

        :param paths: List of paths.
        :return: ``paths`` (sorted) and their indices in ``self._files``.
        """
        if self.prefix is not None:
            paths = [os.path.relpath(p, self.prefix) for p in paths]

        paths = np.sort(paths)
        sorter = np.argsort(self._files)
        index = sorter[np.searchsorted(self._files[sorter], paths)]
        assert np.all(self._files[index] == paths), "not all paths are in the dataset"
        return paths, index

    def _get_info(self, paths: list[pathlib.Path], sha256: bool, progress: bool, verbose: bool):
        """
        Get mtime/size/sha256 of a list of files.

        :param paths: List of paths to check.
        :param progress: Show progress bar (only relevant if ``ssh`` is not set).
        :param verbose: Show verbose output (only relevant if ``ssh`` is set).
        :return: size, mtime, sha256
        """

        if self.ssh is None:
            files = [self._absroot / f for f in paths]
            size, mtime, csum = compute_hash.compute_sha256(files, sha256=sha256, progress=progress)
            return (
                np.array(size, dtype=np.int64),
                np.array(mtime, dtype=np.float64),
                np.array(csum, dtype="U64"),
            )

        cache_dir = ssh._shelephant_cachdir(self.ssh, self.python)
        with ssh._cachedir(self.ssh, cache_dir) as remote, mypathlib.tempdir():
            files = [str(self.root / i) for i in paths]
            pathlib.Path("files.txt").write_text("\n".join(files))
            pathlib.Path("sha256.txt").write_text("")
            shutil.copy(pathlib.Path(__file__).parent / "compute_hash.py", "script.py")

            extra = ["sha256.txt"] if sha256 else []
            hostpath = f'{self.ssh:s}:"{str(remote):s}"'
            _copyfunc(
                ".", hostpath, extra + ["script.py", "files.txt"], progress=False, verbose=verbose
            )
            exec_cmd(
                f'ssh {self.ssh:s} "cd {str(remote)} && {self.python} script.py"', verbose=verbose
            )
            _copyfunc(
                hostpath, ".", extra + ["size.txt", "mtime.txt"], progress=False, verbose=verbose
            )
            size = np.array(
                list(map(int, pathlib.Path("size.txt").read_text().splitlines())),
                dtype=np.int64,
            )
            mtime = np.array(
                list(map(float, pathlib.Path("mtime.txt").read_text().splitlines())),
                dtype=np.float64,
            )
            if sha256:
                csum = np.array(pathlib.Path("sha256.txt").read_text().splitlines(), dtype="U64")
            else:
                csum = []

        return size, mtime, csum

    def check_changes(
        self, paths: list[pathlib.Path] = None, progress: bool = False, verbose: bool = False
    ):
        """
        Remove sha256 from all files of which the size/mtime has changed.

        :param paths:
            List of paths to check.
            Paths to be relative to the root of the dataset.
            Default: all files in the database.

        :param progress: Show progress bar (only relevant if ``ssh`` is not set).
        :param verbose: Show verbose output (only relevant if ``ssh`` is set).
        """
        if paths is None:
            size, mtime, _ = self._get_info(self._files, False, progress, verbose)
            rm = np.logical_or(self._size != size, self._mtime != mtime)
            self._has_info[rm] = False
            self._size[~self._has_info] = size[~self._has_info]
            self._mtime[~self._has_info] = mtime[~self._has_info]
            return self

        paths, index = self._getindex(paths)
        size, mtime, _ = self._get_info(paths, False, progress, verbose)

        changed = np.logical_or(self._size[index] != size, self._mtime[index] != mtime)
        self._has_info[index[changed]] = False
        keep = ~self._has_info[index]
        self._size[index[keep]] = size[keep]
        self._mtime[index[keep]] = mtime[keep]

        removed = mtime < 0
        if np.any(removed):
            return self.remove(paths[removed])

        return self

    def remove_info(self, paths: list[pathlib.Path] = None):
        """
        Remove sha256/size/mtime for a list of files.

        :param paths: List of paths to remove.
        """
        paths, index = self._getindex(paths)
        self._has_info[index] = False

    def getinfo(
        self,
        paths: list[pathlib.Path] = None,
        max_size: int = None,
        progress: bool = False,
        verbose: bool = False,
    ):
        """
        Compute sha256/size/mtime of all files for which this information is not available.

        To compute the sha256/size/mtime only on a fraction of files set ``max_size``.
        This will stop the computation when the total size exceeds ``max_size``.
        You can then call this function recursively (with ``clean=False``) to flush you buffer.

        :param paths: List of paths to check. Have to be relative to the root of the dataset.
        :param max_size: Compute the sha256/size/mtime until the total size exceeds ``max_size``.
        :param progress: Show progress bar (only relevant if ``ssh`` is not set).
        :param verbose: Show verbose output (only relevant if ``ssh`` is set).
        """
        if paths is None:
            paths = self._files
            index = np.arange(self._files.size)
        else:
            paths, index = self._getindex(paths)

        index = index[np.argwhere(~self._has_info[index]).flatten()]
        index = index[np.argsort(self._size[index])]
        files = self._files[index]
        size = self._size[index]

        if index.size == 0:
            return self

        if max_size is not None:
            cum_size = np.cumsum(size)
            i = np.argmax(cum_size > max_size)
            if i > 0:
                index = index[:i]
                files = files[:i]

        size, mtime, csum = self._get_info(files, True, progress, verbose)
        self._has_info[index] = True
        self._sha256[index] = csum
        self._size[index] = size
        self._mtime[index] = mtime

        removed = mtime < 0
        if np.any(removed):
            return self.remove(files[removed])

        return self

    def diff(self, other) -> dict:
        """
        Compare the database entries of two locations.

        .. warning::

            The information is taken from the database.
            There are no checks that that information if up to date.

        :param other: Other location.
        :return:
            Dictionary with differences::

                {
                    "==" : [ ... ], # in a and b, equal sha256
                    "?=" : [ ... ], # in a and b, unknown sha256
                    "!=" : [ ... ], # in a and b, different sha256
                    "->" : [ ... ], # in a not in b
                    "<-" : [ ... ], # in b not in a
                }
        """

        inboth = [str(file) for file in np.intersect1d(self._files, other._files)]
        ret = {
            "->": [str(file) for file in np.setdiff1d(self._files, other._files)],
            "<-": [str(file) for file in np.setdiff1d(other._files, self._files)],
            "==": [],
            "?=": [],
            "!=": [],
        }

        index_self = {file: i for i, file in enumerate(self._files)}
        index_other = {file: i for i, file in enumerate(other._files)}

        for file in inboth:
            ia = index_self[file]
            ib = index_other[file]

            if self._has_info[ia] and other._has_info[ib]:
                if self._sha256[ia] == other._sha256[ib]:
                    ret["=="].append(file)
                else:
                    ret["!="].append(file)
            else:
                ret["?="].append(file)

        return ret


def _compute_suffix(a: Location, b: Location) -> pathlib.Path:
    """
    Compute the suffix to put to each location.

    :param a: First location.
    :param b: Second location.
    :return:

        - ``common_prefix``: the common prefix shared by both locations.
        - ``a_suffix``: the suffix to put to the first location.
        - ``b_suffix``: the suffix to put to the second location.
        - ``deepest``: the deepest of the two prefixes.
    """
    a = a.prefix if a.prefix is not None else pathlib.Path("")
    b = b.prefix if b.prefix is not None else pathlib.Path("")
    a = a.parts
    b = b.parts

    if len(a) > len(b):
        assert a[: len(b)] == b, "Locations are not in the same directory tree."
        c = a[: len(b)]
        a = a[len(b) :]
        d = a
        b = []
    else:
        assert b[: len(a)] == a, "Locations are not in the same directory tree."
        c = b[: len(a)]
        b = b[len(a) :]
        d = b
        a = []

    return pathlib.Path(*c), pathlib.Path(*b), pathlib.Path(*a), pathlib.Path(*d)


# https://stackoverflow.com/a/68994012/2646505
def _search_upwards_dir(dirname: str) -> pathlib.Path:
    """
    Search in the current directory and all directories above it for a directory
    with a particular name.

    :param dirname: The name of the directory to look for.
    :return: The location of the first directory found or None, if none was found.
    """
    d = pathlib.Path.cwd()
    root = pathlib.Path(d.root)

    while d != root:
        attempt = d / dirname
        if attempt.is_dir():
            return attempt
        d = d.parent

    return None


def _init_parser():
    """
    Return parser for :py:func:`shelephant init`.
    """

    desc = textwrap.dedent(
        """
        Initialize a shelephant database by creating a directory ``.shelephant``
        with an 'empty' database. Use ``shelephant add`` to add storage locations.
        """
    )

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)
    parser.add_argument("--version", action="version", version=version)
    return parser


def init(args: list[str]):
    """
    Command-line tool, see ``--help``.

    :param args: Command-line arguments (should be all strings).
    """

    parser = _init_parser()
    args = parser.parse_args(args)
    sdir = pathlib.Path(".shelephant")
    assert not sdir.is_dir(), '".shelephant" directory already exists'
    sdir.mkdir()
    (sdir / "data").mkdir()
    (sdir / "storage").mkdir()
    (sdir / "unavailable").symlink_to("dead-link")
    (sdir / "symlinks.yaml").write_text("")
    yaml.dump(sdir / "storage.yaml", ["here"])
    yaml.dump(sdir / "storage" / "here.yaml", {"root": "../.."})


def _lock_parser():
    """
    Return parser for :py:func:`shelephant lock`.
    """

    desc = textwrap.dedent(
        """
        Lock as storage location.
        """
    )

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)
    parser.add_argument("name", type=str, help="Name of the storage location.")
    parser.add_argument("--version", action="version", version=version)
    return parser


def lock(args: list[str]):
    """
    Command-line tool, see ``--help``.

    :param args: Command-line arguments (should be all strings).
    """

    parser = _lock_parser()
    args = parser.parse_args(args)
    sdir = _search_upwards_dir(".shelephant")
    assert sdir is not None, "Not in a shelephant dataset"
    assert args.name.lower() != "here", "cannot lock 'here'"
    assert (sdir / "storage" / f"{args.name}.yaml").is_file(), "storage location not found"
    (sdir / "lock.txt").write_text(args.name)


def _get_send_storage_parser():
    """
    Return parser for :py:func:`shelephant send_storage`.
    """

    desc = textwrap.dedent(
        """
        Copy f"/shelephant/storage/{name}.yaml".
        """
    )

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)
    parser.add_argument("--colors", type=str, default="dark", help="Color scheme [none, dark].")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite without prompt.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Do not print progress.")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Print copy-plan and exit.")
    parser.add_argument(
        "--mode",
        type=str,
        help="Use 'sha256', 'rsync', and/or 'basic' to compare files.",
        default="rsync" if shutil.which("rsync") is not None else "basic",
    )
    parser.add_argument("name", type=str, help="Name of the storage location.")
    parser.add_argument("--version", action="version", version=version)
    parser.add_argument("--verbose", action="store_true", help="Verbose commands.")
    return parser


def _get_send_storage_impl(args: list[str], send: bool):
    """
    Command-line tool, see ``--help``.

    :param args: Command-line arguments (should be all strings).
    :param send: Send or get
    """

    parser = _get_send_storage_parser()
    args = parser.parse_args(args)
    sdir = _search_upwards_dir(".shelephant")
    assert sdir is not None, "Not in a shelephant dataset"
    assert args.name.lower() != "here", "cannot get_storage 'here'"
    assert (sdir / "storage" / f"{args.name}.yaml").is_file(), "storage location not found"

    with mypathlib.cwd(sdir):
        if send:
            opts = ["storage/here.yaml", f"storage/{args.name}.yaml"]
        else:
            opts = [f"storage/{args.name}.yaml", "storage/here.yaml"]
        opts += ["--ignore-prefix"]
        opts += ["--colors", args.colors]
        opts += ["--mode", args.mode]
        opts += ["--force"] if args.force else []
        opts += ["--quiet"] if args.quiet else []
        opts += ["--dry-run"] if args.dry_run else []
        opts += ["--verbose"] if args.verbose else []
        cli.shelephant_cp(opts, paths=[f".shelephant/storage/{args.name}.yaml"], filter_paths=False)


def send_storage(args: list[str]):
    """
    Command-line tool, see ``--help``.

    :param args: Command-line arguments (should be all strings).
    """
    return _get_send_storage_impl(args, send=True)


def get_storage(args: list[str]):
    """
    Command-line tool, see ``--help``.

    :param args: Command-line arguments (should be all strings).
    """
    return _get_send_storage_impl(args, send=False)


def _create_symlink_data(
    sdir: pathlib.Path,
    name: str,
    loc: Location,
):
    """
    Create or refresh symlink in ``.shelephant/data/<name>``.

    :param sdir: Path to ``.shelephant`` directory.
    :param name: Name of the storage location.
    :param loc: Location.
    :param refresh: Remove existing symlink.
    """
    if name == "here":
        return

    with mypathlib.cwd(sdir):
        mylink = pathlib.Path(f"data/{name}")
        if mylink.is_symlink():
            mylink.unlink()

        if loc.is_mounted():
            mylink.symlink_to(loc._absroot)
        else:
            mylink.symlink_to(pathlib.Path("..") / "unavailable")

        storage = yaml.read("storage.yaml")
        if name not in storage:
            storage.append(name)
            yaml.overwrite("storage.yaml", storage)


def _add_parser():
    """
    Return parser for :py:func:`shelephant add`.
    """

    desc = textwrap.dedent(
        """
        Add a storage location to the database.
        The database in ``.shelephant`` is updated as follows:

        -   The ``name`` is added to ``.shelephant/storage.yaml``.

        -   A file ``.shelephant/storage/<name>.yaml`` is created with the search settings
            and the present state of the storage location.

        -   A symlink ``.shelephant/data/<name>`` is created to the storage location.
            (if ``--ssh`` is given, the symlink points to a dead link).

        .. note::

            A special case is

            .. code-block:: bash

                shelephant add here --rglob '*.h5'

            which helps to investigate your database directory.
            Note that ``here`` is a reserved name and that you should not specify the root.
        """
    )

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)

    opts = dict(type=str, action="append", default=[])
    parser.add_argument("name", type=str, help="Name of the storage location.")
    parser.add_argument("root", type=pathlib.Path, nargs="?", help="Path to the storage location.")
    parser.add_argument("--ssh", type=str, help="SSH host (e.g. user@host).")
    parser.add_argument("--mount", type=pathlib.Path, help="Optional mount location for SSH host.")
    parser.add_argument("--prefix", type=pathlib.Path, help="Add prefix to all files.")
    parser.add_argument("--rglob", **opts, help="Search pattern for ``Path(root).rglob(...)``.")
    parser.add_argument("--glob", **opts, help="Search pattern for ``Path(root).glob(...)``.")
    parser.add_argument("--exec", **opts, help="Command to run from ``root``.")
    parser.add_argument("--skip", **opts, help="Pattern to skip (Python regex).")
    parser.add_argument("--shallow", action="store_true", help="Do not compute checksums.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Do not print progress.")
    parser.add_argument("--version", action="version", version=version)
    return parser


def add(args: list[str]):
    """
    Command-line tool, see ``--help``.

    :param args: Command-line arguments (should be all strings).
    """

    parser = _add_parser()
    args = parser.parse_args(args)
    sdir = _search_upwards_dir(".shelephant")
    assert sdir is not None, "Not in a shelephant dataset"
    assert not (sdir / "lock.txt").exists(), "cannot remove location from storage location"
    assert args.name != "all", "all is a reserved name"
    assert args.name != "any", "any is a reserved name"

    if args.name == "here":
        assert args.root is None, "root is not allowed"
        args.root = pathlib.Path("../..")
    else:
        assert args.root is not None, "root is required"
        storage = yaml.read(sdir / "storage.yaml")
        assert args.name not in storage, f"storage location '{args.name}' already exists"
        if not args.root.is_absolute() and not args.ssh:
            args.root = pathlib.Path(os.path.relpath(args.root.absolute(), sdir / "storage"))
        if args.mount is not None:
            if not args.mount.is_absolute():
                args.mount = pathlib.Path(os.path.relpath(args.mount.absolute(), sdir / "storage"))

    with mypathlib.cwd(sdir):
        loc = Location(root=args.root, ssh=args.ssh, mount=args.mount, prefix=args.prefix)
        s = []
        d = {}
        if len(args.skip) > 0:
            d["skip"] = args.skip
        for i in args.rglob:
            s.append({"rglob": i, **d})
        for i in args.glob:
            s.append({"glob": i, **d})
        for i in args.exec:
            s.append({"exec": i, **d})
        if len(s) > 0:
            loc.search = s

        loc.overwrite_yaml(f"storage/{args.name}.yaml")
        _create_symlink_data(sdir, args.name, Location.from_yaml(f"storage/{args.name}.yaml"))

    opts = [args.name]
    if args.shallow:
        opts.append("--shallow")
    if args.quiet:
        opts.append("--quiet")
    update(opts)


def _remove_parser():
    """
    Return parser for :py:func:`shelephant rm`.
    """

    desc = textwrap.dedent(
        """
        Remove a storage location to the database.
        The database in ``.shelephant`` is updated as follows:

        -   The ``name`` is removed from ``.shelephant/storage.yaml``.
        -   ``.shelephant/storage/<name>.yaml`` is removed.
        -   The symlink ``.shelephant/data/<name>`` is removed.
        """
    )

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)

    parser.add_argument("name", type=str, help="Name of the storage location.")
    parser.add_argument("--version", action="version", version=version)
    return parser


def remove(args: list[str]):
    """
    Command-line tool, see ``--help``.

    :param args: Command-line arguments (should be all strings).
    """

    parser = _remove_parser()
    args = parser.parse_args(args)
    sdir = _search_upwards_dir(".shelephant")
    assert sdir is not None, "Not in a shelephant dataset"
    assert not (sdir / "lock.txt").exists(), "cannot remove location from storage location"

    storage = yaml.read(sdir / "storage.yaml")
    assert args.name in storage, f"storage location '{args.name}' does not exist"
    storage.remove(args.name)
    yaml.overwrite(sdir / "storage.yaml", storage)
    os.remove(sdir / "storage" / f"{args.name}.yaml")
    (sdir / "data" / args.name).unlink()
    update([])


def _rename_parser():
    """
    Return parser for :py:func:`shelephant rm`.
    """

    desc = textwrap.dedent(
        """
        Rename a storage location.
        The database in ``.shelephant`` is updated as follows:

        -   The ``name`` is changed in ``.shelephant/storage.yaml``.
        -   ``.shelephant/storage/<name>.yaml`` is renamed.
        -   The symlink ``.shelephant/data/<name>`` is renamed.
        """
    )

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)

    parser.add_argument("old", type=str, help="Current name of the storage location.")
    parser.add_argument("new", type=str, help="New name of the storage location.")
    parser.add_argument("--version", action="version", version=version)
    return parser


def rename(args: list[str]):
    """
    Command-line tool, see ``--help``.

    :param args: Command-line arguments (should be all strings).
    """

    parser = _rename_parser()
    args = parser.parse_args(args)
    sdir = _search_upwards_dir(".shelephant")
    assert sdir is not None, "Not in a shelephant dataset"
    assert not (sdir / "lock.txt").exists(), "cannot rename location from storage location"

    storage = yaml.read(sdir / "storage.yaml")
    assert args.old in storage, f"storage location '{args.old}' does not exist"
    assert args.new not in storage, f"storage location '{args.new}' exist"
    i = storage.index(args.old)
    storage[i] = args.new
    yaml.overwrite(sdir / "storage.yaml", storage)
    os.rename(sdir / "storage" / f"{args.old}.yaml", sdir / "storage" / f"{args.new}.yaml")
    os.rename(sdir / "data" / args.old, sdir / "data" / args.new)
    if (sdir / "lock.txt").exists():
        myname = (sdir / "lock.txt").read_text().strip()
        if myname == args.old:
            (sdir / "lock.txt").write_text(args.new)
    update([])


def _update_parser():
    """
    Return parser for :py:func:`shelephant update`.
    """

    desc = textwrap.dedent(
        """
        Update the database.
        This function always update the symbolic links, and optionally updates the available
        files and checksums of (a) )storage location(s).
        """
    )

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)

    parser.add_argument("--version", action="version", version=version)
    parser.add_argument(
        "--sync-search",
        action="store_true",
        help="Set the same search settings for all locations (except 'here').",
    )
    parser.add_argument(
        "--base-link",
        action="store_true",
        help="Update link .shelephant/data/{name} based on .shelephant/storage/{name}.yaml.",
    )
    parser.add_argument("--clean", action="store_true", help="Clean database entry with symlinks.")
    parser.add_argument("-s", "--shallow", action="store_true", help="Do not compute checksums.")
    parser.add_argument("--verbose", action="store_true", help="Verbose commands.")
    parser.add_argument(
        "--chunk",
        type=lambda x: int(float(x)),
        default=3e10,
        help="Chunk size for computing checksums (bytes).",
    )
    parser.add_argument("--force", action="store_true", help="Force update of path(s).")
    parser.add_argument("-q", "--quiet", action="store_true", help="Do not print progress.")
    parser.add_argument("name", type=str, nargs="?", help="Update storage location(s).")
    parser.add_argument(
        "path", type=pathlib.Path, nargs="*", help="Update only specific paths on location."
    )
    return parser


def update(args: list[str]):
    """
    Command-line tool, see ``--help``.

    :param args: Command-line arguments (should be all strings).
    """

    parser = _update_parser()
    args = parser.parse_args(args)
    sdir = _search_upwards_dir(".shelephant")
    assert sdir is not None, "Not in a shelephant dataset"
    base = sdir.parent
    paths = [os.path.relpath(path, base) for path in args.path]
    paths = np.unique(paths) if len(paths) > 0 else None
    lock = None if not (sdir / "lock.txt").exists() else (sdir / "lock.txt").read_text().strip()

    if args.sync_search:
        names = yaml.read(sdir / "storage.yaml")
        if "here" in names:
            names.remove("here")
        search = []
        for name in names:
            data = yaml.read(sdir / "storage" / f"{name}.yaml")
            search += data.get("search", [])
        # todo: merge search settings
        search = sorted(list({yaml.dumps(i) for i in search}))
        search = [yaml.loads(i) for i in search]

        if not args.force:
            print("Common search settings:")
            print(yaml.dumps(search))
            if not click.confirm("Apply to all locations?"):
                raise OSError("Cancelled")

        for name in names:
            data = yaml.read(sdir / "storage" / f"{name}.yaml")
            data["search"] = search
            yaml.overwrite(sdir / "storage" / f"{name}.yaml", data)

    if args.force and not args.sync_search:
        assert paths is not None, "--force can only be used with path(s)"

    if args.name is None:
        args.name = []
    elif args.name.lower() == "all":
        assert lock is None, "cannot update all locations from storage location"
        args.name = yaml.read(sdir / "storage.yaml")
    else:
        assert lock is None, "cannot update all locations from storage location"
        assert args.name != "here" or paths is None, "cannot specify paths for 'here'"
        if args.base_link:
            _create_symlink_data(
                sdir, args.name, Location.from_yaml(sdir / "storage" / f"{args.name}.yaml")
            )
        assert args.name in yaml.read(sdir / "storage.yaml"), f"'{args.name}' is not a location"
        args.name = [args.name]

    if lock is not None:
        assert lock != "here"
        args.name = [lock]

    with mypathlib.cwd(sdir):
        # read existing symlinks

        if lock is None:
            symlinks = yaml.read("symlinks.yaml", [])
            symlinks = {pathlib.Path(i["path"]): pathlib.Path(i["storage"]) for i in symlinks}
            if args.clean:
                with mypathlib.cwd(base):
                    for path in list(symlinks.keys()):
                        if not path.is_symlink():
                            symlinks.pop(path)

        # update files and info

        for name in args.name:
            # "here": search for files that are not managed by shelephant
            if name == "here":
                f = "storage/here.yaml"
                Location.from_yaml(f).read().remove(list(symlinks.keys())).to_yaml(f, force=True)
                continue

            # other locations: search for files (or add files by hand)
            loc = Location.from_yaml(f"storage/{name}.yaml")
            if lock is not None:
                loc.root = pathlib.Path("..")
                loc._absroot = loc.root.absolute()
                loc.ssh = None
                loc.mount = False
            if not loc.isavailable():
                continue
            if paths is None:
                loc.read(verbose=args.verbose)
            else:
                if loc.prefix is not None:
                    p = np.array([os.path.relpath(i, loc.prefix) for i in paths])
                else:
                    p = paths
                _, i, _ = np.intersect1d(p, loc._files, return_indices=True, assume_unique=True)
                k = np.ones(p.size, dtype=bool)
                k[i] = False
                loc._append(p[k])

            if lock is not None:
                f = f"storage/{name}.yaml"
                Location.from_yaml(f).copy_files(loc).overwrite_yaml(f)
            else:
                loc.overwrite_yaml(f"storage/{name}.yaml")

            if args.shallow:
                continue

            # compute sha256/size/mtime of files that changed since the last update
            n = loc._files.size
            if args.force:
                loc.remove_info(paths=paths)
            else:
                loc.check_changes(paths=paths, verbose=args.verbose)

            if loc.has_info():
                if loc._files.size < n:
                    if lock is not None:
                        f = f"storage/{name}.yaml"
                        Location.from_yaml(f).copy_files(loc).overwrite_yaml(f)
                    else:
                        loc.overwrite_yaml(f"storage/{name}.yaml")
                continue

            off = np.sum(loc._size[loc._has_info])
            pbar = tqdm.tqdm(
                total=np.sum(loc._size) - off, disable=args.quiet, unit="B", unit_scale=True
            )
            while not loc.has_info():
                loc.getinfo(
                    paths=paths,
                    max_size=args.chunk,
                    progress=not args.quiet,
                    verbose=args.verbose,
                )
                if lock is not None:
                    f = f"storage/{name}.yaml"
                    Location.from_yaml(f).copy_files(loc).overwrite_yaml(f)
                else:
                    loc.overwrite_yaml(f"storage/{name}.yaml")
                pbar.n = np.sum(loc._size[loc._has_info]) - off
                pbar.refresh()

        # update symlinks
        if lock is not None:
            return

        storage = yaml.read(sdir / "storage.yaml")
        storage.remove("here")
        files = {}
        # - link to first available
        for name in storage[::-1]:
            loc = Location.from_yaml(pathlib.Path("storage") / f"{name}.yaml")
            prefix = loc.prefix if loc.prefix is not None else pathlib.Path(".")
            if loc.isavailable(mount=True):
                for f in loc.files(info=False):
                    if (loc._absroot / f).is_file():
                        files[prefix / pathlib.Path(f)] = pathlib.Path("data") / name
        # - unlinked files: link to first unavailable
        for name in storage[::-1]:
            loc = Location.from_yaml(pathlib.Path("storage") / f"{name}.yaml")
            prefix = loc.prefix if loc.prefix is not None else pathlib.Path(".")
            if not loc.isavailable(mount=True):
                for f in loc.files(info=False):
                    if prefix / pathlib.Path(f) not in files:
                        files[prefix / pathlib.Path(f)] = pathlib.Path("data") / name

        add_links = []
        rm_links = []

        for symlink in symlinks:
            if symlink not in files:
                rm_links.append(symlink)
            elif files[symlink] != symlinks[symlink]:
                add_links.append(symlink)
                rm_links.append(symlink)

        for symlink in files:
            if symlink not in symlinks:
                add_links.append(symlink)

        with mypathlib.cwd(sdir / ".."):
            for f in rm_links:
                if f.is_symlink():
                    f.unlink()

            unmanage = []
            for f in add_links:
                if f.is_file():
                    unmanage.append(f)
            for f in unmanage:
                add_links.remove(f)
                files.pop(f)

            if len(unmanage) > 0:
                print("Local files conflicting with dataset. No links are created for these files:")
                print("\n".join(map(str, unmanage)))

            # remove directories that are empty after removing old links
            for d in {i.parent for i in rm_links}:
                if not d.exists():
                    continue
                if not os.listdir(d):
                    d.rmdir()

            store = [{"path": str(i), "storage": str(files[i])} for i in sorted(files.keys())]
            yaml.overwrite(".shelephant/symlinks.yaml", store)

            for f in add_links:
                f.parent.mkdir(parents=True, exist_ok=True)
                s = pathlib.Path(os.path.relpath(".shelephant", f.parent)) / files[f] / f
                f.symlink_to(s)


def _cp_parser():
    """
    Return parser for :py:func:`shelephant cp`.
    """

    desc = textwrap.dedent(
        """
        Copy files between storage locations and update the database.
        After copying, the checksums on the destination are recomputed and the database updated.
        Use:

            -   ``-s``, ``--shallow`` to skip the checksum computation (store only path/size/mtime).
            -   ``-x``, ``--no-update`` to skip the database update all together.

        .. note::

            The paths that you specify are reduced to only the paths known to exist on the source.
            If you know that the paths exist, but they are not part of the database
            (or it is outdated), use ``-e``, ``--exists`` to avoid the filter.

        .. tip::

            To make a clone call ``shelephant cp source destination .`` from the dataset's root.

        .. note::

            The copied files are added to the database of the destination.
            There is no check that this fits ``dump`` and ``search`` settings.
        """
    )

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)

    parser.add_argument("--version", action="version", version=version)
    parser.add_argument("--colors", type=str, default="dark", help="color scheme [none, dark]")
    parser.add_argument("-f", "--force", action="store_true", help="overwrite without prompt")
    parser.add_argument("-q", "--quiet", action="store_true", help="do not print progress")
    parser.add_argument("-n", "--dry-run", action="store_true", help="print copy-plan and exit")
    parser.add_argument("-x", "--no-update", action="store_true", help="no database update")
    parser.add_argument("-e", "--exists", action="store_true", help="all paths exists on source")
    parser.add_argument("-s", "--shallow", action="store_true", help="do not compute checksums")
    parser.add_argument(
        "--mode",
        type=str,
        help="use 'sha256', 'rsync', and/or 'basic' to compare files",
        default="sha256,rsync" if shutil.which("rsync") is not None else "sha256,basic",
    )
    parser.add_argument("source", type=str, help="name of the source")
    parser.add_argument("destination", type=str, help="name of the destination")
    parser.add_argument("path", type=pathlib.Path, nargs="+", help="path(s) to copy")
    return parser


def cp(args: list[str]):
    """
    Command-line tool, see ``--help``.

    :param args: Command-line arguments (should be all strings).
    """
    parser = _cp_parser()
    args = parser.parse_args(args)
    sdir = _search_upwards_dir(".shelephant")
    assert sdir is not None, "Not in a shelephant dataset"
    assert not (sdir / "lock.txt").exists(), "cannot remove location from storage location"
    storage = yaml.read(sdir / "storage.yaml")
    assert args.source in storage, f"Unknown storage location {args.source}"
    assert args.destination in storage, f"Unknown storage location {args.destination}"
    base = sdir.parent
    args.path = args.path if args.path != [pathlib.Path(".")] else []
    paths = [os.path.relpath(path, base) for path in args.path]

    with mypathlib.cwd(sdir):
        opts = [f"storage/{args.source}.yaml", f"storage/{args.destination}.yaml"]
        opts += ["--colors", args.colors]
        opts += ["--mode", args.mode]
        opts += ["--force"] if args.force else []
        opts += ["--quiet"] if args.quiet else []
        opts += ["--dry-run"] if args.dry_run else []
        changed = cli.shelephant_cp(opts, paths=paths, filter_paths=not args.exists)

    if not args.dry_run and len(changed) > 0 and not args.no_update:
        if len(paths) > 0:
            _, j, _ = np.intersect1d(paths, changed, return_indices=True, assume_unique=True)
            changed = np.array(args.path)[j]
        opts = ["--quiet", "--force", args.destination]
        opts += ["--shallow"] if args.shallow else []
        update(opts + list(map(str, changed)))


def _mv_parser():
    """
    Return parser for :py:func:`shelephant mv`.
    """

    desc = textwrap.dedent(
        """
        Move files from one storage location to another.

        .. note::

            The copied files are added to the database of the destination.
            There is no check that this fits ``dump`` and ``search`` settings.
        """
    )

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)

    parser.add_argument("--version", action="version", version=version)
    parser.add_argument("--colors", type=str, default="dark", help="Color scheme [none, dark].")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite without prompt.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Do not print progress.")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Print copy-plan and exit.")
    parser.add_argument("source", type=str, help="name of the source.")
    parser.add_argument("destination", type=str, help="name of the destination.")
    parser.add_argument("path", type=pathlib.Path, nargs="+", help="path(s) to copy.")
    return parser


def mv(args: list[str]):
    """
    Command-line tool, see ``--help``.

    :param args: Command-line arguments (should be all strings).
    """

    parser = _mv_parser()
    args = parser.parse_args(args)
    sdir = _search_upwards_dir(".shelephant")
    assert sdir is not None, "Not in a shelephant dataset"
    assert not (sdir / "lock.txt").exists(), "cannot remove location from storage location"
    assert args.destination != "here", "Cannot copy to here."
    storage = yaml.read(sdir / "storage.yaml")
    assert args.source in storage, f"Unknown storage location {args.source}"
    assert args.destination in storage, f"Unknown storage location {args.destination}"
    base = sdir.parent
    paths = [os.path.relpath(path, base) for path in args.path]

    with mypathlib.cwd(sdir):
        dest = Location.from_yaml(f"storage/{args.destination}.yaml")
        assert dest.ssh is None, "Cannot move to remote location."
        opts = [f"storage/{args.source}.yaml", str(dest._absroot)]
        opts += ["--colors", args.colors]
        opts += ["--force"] if args.force else []
        opts += ["--quiet"] if args.quiet else []
        opts += ["--dry-run"] if args.dry_run else []
        cli.shelephant_mv(opts, paths)

    if not args.dry_run:
        with mypathlib.cwd(sdir):
            f = f"storage/{args.source}.yaml"
            Location.from_yaml(f).remove(paths).overwrite_yaml(f)
        update(["--quiet", "--force", args.destination] + list(map(str, args.path)))


def _rm_parser():
    """
    Return parser for :py:func:`shelephant rm`.
    """

    desc = textwrap.dedent(
        """
        Remove files from a storage location.

        .. warning::

            This removes the actual data.
            The link is also if there is no alternative source left.
        """
    )

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)

    parser.add_argument("--version", action="version", version=version)
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite without prompt.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Do not print progress.")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Print copy-plan and exit.")
    parser.add_argument("source", type=str, help="name of the source.")
    parser.add_argument("path", type=pathlib.Path, nargs="+", help="path(s) to remove.")
    return parser


def rm(args: list[str]):
    """
    Command-line tool, see ``--help``.

    :param args: Command-line arguments (should be all strings).
    """

    parser = _rm_parser()
    args = parser.parse_args(args)
    sdir = _search_upwards_dir(".shelephant")
    assert sdir is not None, "Not in a shelephant dataset"
    assert not (sdir / "lock.txt").exists(), "cannot remove location from storage location"
    storage = yaml.read(sdir / "storage.yaml")
    assert args.source in storage, f"Unknown storage location {args.source}"
    base = sdir.parent
    paths = [os.path.relpath(path, base) for path in args.path]

    with mypathlib.cwd(sdir):
        opts = [f"storage/{args.source}.yaml"]
        opts += ["--force"] if args.force else []
        opts += ["--quiet"] if args.quiet else []
        opts += ["--dry-run"] if args.dry_run else []
        cli.shelephant_rm(opts, paths)

    if not args.dry_run:
        with mypathlib.cwd(sdir):
            f = f"storage/{args.source}.yaml"
            Location.from_yaml(f).remove(paths).overwrite_yaml(f)
            update([])


def _pwd_parser():
    """
    Return parser for :py:func:`shelephant pwd`.
    """

    desc = textwrap.dedent(
        """
        Change the current working directory to a storage location.
        """
    )

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)

    parser.add_argument("--version", action="version", version=version)
    parser.add_argument("--base", action="store_true", help="Print the base directory.")
    parser.add_argument("--abspath", action="store_true", help="Print absolute path.")
    parser.add_argument("source", type=str, help="name of the source.")
    return parser


def pwd(args: list[str]):
    """
    Command-line tool, see ``--help``.

    :param args: Command-line arguments (should be all strings).
    """

    parser = _pwd_parser()
    args = parser.parse_args(args)
    sdir = _search_upwards_dir(".shelephant")
    assert sdir is not None, "Not in a shelephant dataset"
    assert not (sdir / "lock.txt").exists(), "not available from storage location"
    storage = yaml.read(sdir / "storage.yaml")
    assert args.source in storage, f"Unknown storage location {args.source}"

    cwd = pathlib.Path.cwd()
    post = os.path.relpath(cwd, sdir / "..")

    with mypathlib.cwd(sdir):
        f = f"storage/{args.source}.yaml"
        loc = Location.from_yaml(f)
        prefix = loc.prefix
        root = sdir / "storage" / loc.root

    if prefix is not None:
        common = os.path.commonprefix([prefix, post])
        post = post[len(common) :]

    if args.base:
        post = ""

    if args.abspath:
        print(os.path.normpath(root / post))
    else:
        print(os.path.relpath(root / post, os.getcwd()))


def _diff_parser():
    """
    Return parser for :py:func:`shelephant diff`.
    """

    desc = textwrap.dedent(
        """
        Show differences between two storage locations.
        """
    )

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)

    parser.add_argument("--version", action="version", version=version)
    parser.add_argument("--colors", type=str, default="dark", help="Color scheme [none, dark].")
    parser.add_argument("--pop", type=str, action="append", help='Pop direction (e.g. "==").')
    parser.add_argument("source", type=str, help="Storage location.")
    parser.add_argument("dest", type=str, help="Storage location.")
    return parser


def diff(args: list[str]):
    """
    Command-line tool, see ``--help``.

    :param args: Command-line arguments (should be all strings).
    """

    parser = _diff_parser()
    args = parser.parse_args(args)
    sdir = _search_upwards_dir(".shelephant")
    assert sdir is not None, "Not in a shelephant dataset"

    with mypathlib.cwd(sdir):
        storage = yaml.read(sdir / "storage.yaml")
        assert args.source in storage, f"Unknown storage location {args.source}"
        assert args.dest in storage, f"Unknown storage location {args.dest}"
        source = Location.from_yaml(f"storage/{args.source}.yaml")
        dest = Location.from_yaml(f"storage/{args.dest}.yaml")

    status = source.diff(dest)
    if args.pop is not None:
        for i in args.pop:
            status.pop(i)
    for key in status:
        status[key] = sorted(status[key])
    output.diff(status, colors=args.colors)


def _status_parser():
    """
    Return parser for :py:func:`shelephant status`.
    """

    desc = textwrap.dedent(
        """
        Status of the storage locations.

        .. tip::

            Use ``--list`` or ``--print0`` to get a list of files instead of a table.
            Use for example as:

            .. code-block:: bash

                shelephant cp source dest $(shelephant status --copies 1 --list)

            or to copy in batches of 100:

            .. code-block:: bash

                shelephant status --copies 1 --print0 | xargs -n 100 -0 shelephant cp source dest $@

            The latter you can also do with the ``--nout`` (``-n``) option of ``shelephant status``:

            .. code-block:: bash

                shelephant cp source dest $(shelephant status --copies 1 --list -n 100)
        """
    )

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)

    parser.add_argument("--version", action="version", version=version)
    parser.add_argument("--size", action="store_true", help="Show size of files.")
    parser.add_argument("--reverse", action="store_true", help="Reverse output order.")
    parser.add_argument("--sort", type=str, help="Sort by 'size' or 'in use'.")
    parser.add_argument("--min-copies", type=int, help="Show files with minimal number of copies.")
    parser.add_argument("--copies", type=int, help="Show files with specific number of copies.")
    parser.add_argument("--ne", action="store_true", help="Show files with unequal copies.")
    parser.add_argument("--na", action="store_true", help="Show files unavailable somewhere.")
    parser.add_argument("--unknown", action="store_true", help="Show files with unknown sha256.")
    parser.add_argument("--list", action="store_true", help="Print list of files (no table).")
    parser.add_argument("--print0", action="store_true", help="Print list of files (no table).")
    parser.add_argument("-n", "--nout", type=int, help="Maximal number of output arguments.")
    parser.add_argument("--table", type=str, default="SINGLE_BORDER", help="Select print style.")
    parser.add_argument(
        "--in-use", type=str, help="Select storage location in use (use 'none' for unavailable)."
    )
    parser.add_argument(
        "--not-on",
        type=str,
        help="List files that are not on a storage location.",
    )
    parser.add_argument(
        "--on",
        type=str,
        action="append",
        default=[],
        help="Limit to files available on storage location.",
    )
    parser.add_argument(
        "-b",
        "--relative-to-base",
        action="store_true",
        help="Show path relative to base directory of dataset.",
    )
    parser.add_argument(
        "path",
        type=str,
        nargs="*",
        help="Filter to paths (either one directory, or multiple files).",
    )
    return parser


# https://stackoverflow.com/a/1094933/2646505
def sizeof_fmt(num, suffix=""):
    for unit in ("", "K", "M", "G", "T", "P", "E", "Z"):
        if abs(num) < 1e3:
            return f"{num:.0f}{unit}{suffix}"
        num /= 1e3
    return f"{num:.1f}Y{suffix}"


def status(args: list[str]):
    """
    Command-line tool, see ``--help``.

    :param args: Command-line arguments (should be all strings).
    """

    parser = _status_parser()
    args = parser.parse_args(args)
    sdir = _search_upwards_dir(".shelephant")
    assert sdir is not None, "Not in a shelephant dataset"
    base = sdir.parent
    cwd = os.path.relpath(pathlib.Path.cwd(), base)
    paths = [os.path.relpath(path, base) for path in args.path]

    na = "----"
    if args.in_use is not None:
        if args.in_use == "none":
            args.in_use = na

    formatter = np.vectorize(sizeof_fmt)
    with mypathlib.cwd(sdir):
        symlinks = np.sort([i["path"] for i in yaml.read("symlinks.yaml", [])])
        storage = yaml.read(sdir / "storage.yaml")
        storage.remove("here")
        extra = Location.from_yaml("storage/here.yaml").files(info=False)

        sha = "x" * np.ones((len(symlinks), len(storage)), dtype=object)
        mtime = np.inf * np.ones(sha.shape, dtype=np.float64)
        inuse = na * np.ones((len(symlinks)), dtype=object)
        filesize = np.zeros((len(symlinks)), dtype=object)

        for iname, name in enumerate(storage[::-1]):
            loc = Location.from_yaml(pathlib.Path("storage") / f"{name}.yaml")
            if loc.prefix is not None:
                files = np.array([str(loc.prefix / f) for f in loc._files])
            else:
                files = loc._files
            sorter = np.argsort(files)
            idx = np.searchsorted(symlinks, files[sorter])
            s = np.array(loc._sha256)[sorter]
            m = np.array(loc._mtime)[sorter]
            f = np.array(loc._size, dtype=object)[sorter]
            h = ~loc._has_info[sorter]
            if np.any(h):
                s[h] = "?"
                m[h] = np.inf
                f[h] = "?"
            sha[idx, -1 - iname] = s
            mtime[idx, -1 - iname] = m
            filesize[idx] = f
            if loc.isavailable(mount=True):
                inuse[idx] = name

    for i in range(sha.shape[0]):
        _sha = sha[i, :]
        unique, forward, backward = np.unique(_sha, return_index=True, return_inverse=True)
        info = np.logical_and(unique != "x", unique != "?")

        if np.sum(info) == 0:
            continue

        if np.sum(info) == 1:
            _sha[_sha == unique[info][0]] = "=="
            sha[i, :] = _sha
            continue

        info = np.logical_and(_sha != "x", _sha != "?")
        _mtime = mtime[i, :]
        _mtime[~info] = np.inf
        label = np.empty(forward.size, dtype=int)
        label[np.argsort(_mtime[forward])] = np.arange(1, forward.size + 1)
        label = label[backward][info]
        _sha[info] = np.array(list(map(str, label)), dtype=object)
        sha[i, :] = _sha

    data = np.hstack((np.array([symlinks]).T, filesize.reshape(-1, 1), inuse.reshape(-1, 1), sha))

    e = "?" * np.ones((len(extra), data.shape[1]), dtype=object)
    e[:, 0] = extra
    e[:, 1] = "here"
    e[:, 2] = "?"
    data = np.vstack((data, e))

    if args.not_on is not None:
        name = args.not_on
        iname = -(len(storage) - np.argmax(np.equal(storage, name)))
        keep = np.equal(data[:, iname], "x")
        data = data[keep, :]

    if len(args.on) > 0:
        keep = np.zeros((len(data)), dtype=bool)
        for name in args.on:
            if name == "here":
                keep = np.logical_or(keep, data[:, 2] == "here")
            else:
                iname = -(len(storage) - np.argmax(np.equal(storage, name)))
                keep = np.logical_or(keep, np.not_equal(data[:, iname], "x"))
        data = data[keep, :]

    if args.min_copies is not None:
        data = data[np.sum(data[:, 3:] == "==", axis=1) >= args.min_copies]
    if args.copies is not None:
        data = data[np.sum(data[:, 3:] == "==", axis=1) == args.copies]
    if args.ne:
        data = data[np.sum(data[:, 3:] == "1", axis=1) > 0]
    if args.na:
        data = data[np.sum(data[:, 3:] == "x", axis=1) > 0]
    if args.unknown:
        data = data[np.sum(data[:, 3:] == "?=", axis=1) > 0]
    if args.in_use is not None:
        data = data[data[:, 2] == args.in_use]

    if len(paths) > 0:
        keep = np.zeros(data.shape[0], dtype=bool)
        for path in paths:
            keep = np.logical_or(
                keep, [os.path.commonpath([str(i), path]) == path for i in data[:, 0]]
            )
        data = data[keep]

    if not args.relative_to_base:
        data[:, 0] = [os.path.relpath(i, cwd) for i in data[:, 0]]

    if args.nout is not None:
        data = data[: args.nout, ...]

    if args.sort is not None:
        if args.sort.lower() == "size":
            idx = 1
        elif args.sort.lower() == "in use":
            idx = 2
        else:
            raise ValueError(f"Unknown sort option: {args.sort}")
        data = data[np.argsort(data[:, idx])]

    if args.reverse:
        data = data[::-1]

    if args.print0:
        print("\0".join(data[:, 0]))
        return

    if args.list:
        print("\n".join(data[:, 0]))
        return

    out = prettytable.PrettyTable()
    if args.table == "PLAIN_COLUMNS":
        out.set_style(prettytable.PLAIN_COLUMNS)
    elif args.table == "SINGLE_BORDER":
        out.set_style(prettytable.SINGLE_BORDER)

    if args.size:
        field_names = ["path", "size", "in use"]
        data[:, 1] = formatter(data[:, 1])
    else:
        field_names = ["path", "in use"]
        data = np.delete(data, 1, axis=1)

    out.field_names = field_names + storage

    out.align["path"] = "l"
    out.align["in use"] = "l"
    if args.size:
        out.align["size"] = "l"
    for name in storage:
        out.align[name] = "c"

    for row in data:
        out.add_row(row)

    output.autoprint(out.get_string())


def _info_parser():
    """
    Return parser for :py:func:`shelephant info`.
    """

    desc = textwrap.dedent(
        """
        Show global information about dataset.
        """
    )

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)

    parser.add_argument("--version", action="version", version=version)
    parser.add_argument("--cachedir", action="store_true", help="Print cache-dir and quit.")
    parser.add_argument(
        "--basedir", action="store_true", help="Print basedir (containing '.shelephant') and quit."
    )
    parser.add_argument("location", type=str, nargs="*", help="Name of the storage location(s).")
    return parser


def info(args: list[str]):
    """
    Command-line tool, see ``--help``.

    :param args: Command-line arguments (should be all strings).
    """

    parser = _info_parser()
    args = parser.parse_args(args)

    if args.cachedir:
        print(user_cache_dir("shelephant", os.getlogin()))
        return

    sdir = _search_upwards_dir(".shelephant")
    assert sdir is not None, "Not in a shelephant dataset"

    def _convert_path(path: pathlib.Path) -> str:
        if os.path.isabs(path):
            return path
        return os.path.relpath((sdir / "storage" / path).resolve())

    if args.basedir:
        print(sdir.parent)
        return

    locations = yaml.read(sdir / "storage.yaml")
    if len(args.location) == 0:
        args.location = locations
    else:
        assert all([i in locations for i in args.location]), "Unknown storage location(s)"

    ret = ""
    for i, location in enumerate(args.location):
        loc = Location.from_yaml(sdir / "storage" / f"{location}.yaml")
        out = prettytable.PrettyTable()
        out.set_style(prettytable.SINGLE_BORDER)
        out.field_names = ["name", location]
        out.align = "l"
        out.add_row(["root", _convert_path(loc.root)])
        if loc.prefix is not None:
            out.add_row(["prefix", loc.prefix])
        if loc.ssh is not None:
            out.add_row(["ssh", loc.ssh])
        if loc.description is not None:
            out.add_row(["description", loc.description])
        ret += out.get_string()
        if i < len(args.location) - 1:
            ret += "\n"

    output.autoprint(ret)


def _find_matching(
    text: str,
    opening: str,
    closing: str,
) -> dict:
    r"""
    Find matching 'brackets'.

    :param text: The string to consider.
    :param opening: The opening bracket (e.g. "(", "[", "{").
    :param closing: The closing bracket (e.g. ")", "]", "}").
    :return: Dictionary with ``{index_opening: index_closing}``
    """

    opening = [i.span()[0] for i in re.finditer(opening, text)]
    closing = [i.span()[1] for i in re.finditer(closing, text)]

    if len(opening) == 0:
        return {}

    if len(opening) > len(closing):
        raise IndexError("Unmatching opening...closing found")

    opening = np.array(opening, dtype=int)
    closing = np.array(closing, dtype=int) * -1
    brackets = np.concatenate([opening, closing])
    brackets = brackets[np.argsort(np.abs(brackets))]

    ret = {}
    stack = []

    for i in brackets:
        if i >= 0:
            stack.append(i)
        else:
            if len(stack) == 0:
                raise IndexError(f"No closing {closing} at: {i:d}")
            j = stack.pop()
            ret[j] = -1 * i

    if len(stack) > 0:
        i = stack.pop()
        raise IndexError(f"No opening {opening} at: {i:d}")

    return ret


def _linenumbers(text: str) -> list[int]:
    """
    Return the line-number of each character in ``text``.

    :param text: The text to consider.
    :return: List of line-numbers.
    """
    lineno = np.empty(len(text), dtype=int)
    i = 0
    line = 0
    for line, match in enumerate(re.finditer(r"\n", text)):
        lineno[i : match.span()[0]] = line
        i = match.span()[0]
        lineno[i] = line
        i += 1
    lineno[i:] = line + 1
    lineno = np.append(lineno, line + 2)
    return lineno


def _gitignore_parser():
    """
    Return parser for :py:func:`shelephant gitignore`.
    """

    desc = textwrap.dedent(
        """
        Add all symbolic links managed to the dataset's root ``.gitignore``.

        .. note::

            This is the ``/path/to/dataset/.gitignore`` file, not
            ``/path/to/dataset/.shelephant/.gitignore``.
        """
    )

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=desc)

    parser.add_argument("--version", action="version", version=version)
    return parser


def gitignore(args: list[str]):
    """
    Command-line tool, see ``--help``.

    :param args: Command-line arguments (should be all strings).
    """

    parser = _gitignore_parser()
    args = parser.parse_args(args)
    sdir = _search_upwards_dir(".shelephant")
    assert sdir is not None, "Not in a shelephant dataset"
    path = sdir / ".." / ".gitignore"

    if path.exists():
        ignore = path.read_text()
        brackets = _find_matching(ignore, re.escape("# <shelephant>"), re.escape("# </shelephant>"))
        if len(brackets) > 0:
            assert len(brackets) == 1, "Multiple shelephant sections found"
            brackets = [(key, value) for key, value in brackets.items()][0]
            lineno = _linenumbers(ignore)
            ignore = ignore.splitlines()
            ignore = (
                "\n".join(ignore[: lineno[brackets[0]]]).rstrip()
                + "\n"
                + "\n".join(ignore[lineno[brackets[1]] + 1 :]).lstrip()
            )
    else:
        ignore = ""

    with mypathlib.cwd(sdir):
        symlinks = [i["path"] for i in yaml.read("symlinks.yaml", [])]

    ignore += "\n# <shelephant>\n" + "\n".join(symlinks) + "\n# </shelephant>\n"
    path.write_text(ignore.lstrip())


def git(args: list[str]):
    """
    Run git from ``.shelephant`` directory.

    :param args: Command-line arguments (should be all strings).
    """
    with mypathlib.cwd(_search_upwards_dir(".shelephant")):
        print(exec_cmd(f"git {' '.join(args)}"))
