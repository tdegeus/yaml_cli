import contextlib
import io
import os
import pathlib
import re
import shutil
import unittest

import shelephant
from shelephant._tests import create_dummy_files
from shelephant.cli import f_dump
from shelephant.cli import f_hostinfo
from shelephant.cli import shelephant_cp
from shelephant.cli import shelephant_diff
from shelephant.cli import shelephant_dump
from shelephant.cli import shelephant_hostinfo
from shelephant.cli import shelephant_mv
from shelephant.cli import shelephant_parse
from shelephant.cli import shelephant_rm
from shelephant.path import cwd
from shelephant.path import tempdir

has_ssh = shelephant.ssh.has_keys_set("localhost")
has_rsync = shutil.which("rsync") is not None


def _plain(text):
    return list(filter(None, [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]))


class Test_shelephant_parse(unittest.TestCase):
    def test_basic(self):
        """
        shelephant_parse <file.yaml>
        """
        with tempdir(), contextlib.redirect_stdout(io.StringIO()) as sio:
            files = ["foo.txt", "bar.txt"]
            create_dummy_files(files)
            shelephant_dump(["-i"] + files)
            check = pathlib.Path(f_dump).read_text()
            shelephant_parse([f_dump])

        self.assertEqual(sio.getvalue().strip(), check.strip())


class Test_shelephant_dump(unittest.TestCase):
    def test_checksum(self):
        with tempdir():
            files = ["foo.txt", "bar.txt"]
            check = create_dummy_files(files)
            shelephant_dump(["-i"] + files)
            data = shelephant.dataset.Location.from_yaml(f_dump)
            self.assertTrue(check == data)

    def test_search(self):
        with tempdir():
            files = ["foo.txt", "bar.txt", "a.txt", "b.txt", "c.txt", "d.txt"]
            check = create_dummy_files(files)
            shelephant.yaml.dump("search.yaml", {"root": ".", "search": [{"rglob": "*.txt"}]})
            shelephant_dump(["-i", "--search", "search.yaml"])
            data = shelephant.dataset.Location.from_yaml(f_dump)
            self.assertTrue(check == data)

    def test_find(self):
        with tempdir():
            files = ["foo.txt", "bar.txt", "a.txt", "b.txt", "c.txt", "d.txt"]
            check = create_dummy_files(files)
            shelephant_dump(["-i", "-c", "find . -name '*.txt'"])
            data = shelephant.dataset.Location.from_yaml(f_dump)
            self.assertTrue(check == data)

    def test_append(self):
        with tempdir():
            files = ["foo.pdf", "bar.pdf", "a.txt", "b.txt", "c.txt", "d.txt"]
            check = create_dummy_files(files)
            pdf = [i for i in files if i.endswith(".pdf")]

            # plain

            shelephant_dump(pdf)
            data = shelephant.dataset.Location.from_yaml(f_dump)
            self.assertEqual(data.files(info=False), pdf)

            shelephant_dump(["-a", "-c", "find . -name '*.txt'"])
            data = shelephant.dataset.Location.from_yaml(f_dump).sort()
            self.assertEqual(data.files(info=False), sorted(files))

            # with details

            shelephant_dump(["-f", "-i"] + pdf)
            data = shelephant.dataset.Location.from_yaml(f_dump)
            self.assertEqual(data.files(info=False), pdf)

            shelephant_dump(["-a", "-i", "-c", "find . -name '*.txt'"])
            data = shelephant.dataset.Location.from_yaml(f_dump)
            self.assertTrue(check == data)

    def test_all(self):
        with tempdir():
            pathlib.Path("src").mkdir()
            nested = [os.path.join("src", i) for i in ["f.txt", "g.txt"]]
            files = ["foo.txt", "bar.txt", "a.txt", "b.txt", "c.txt", "d.txt"] + nested
            check = create_dummy_files(files)
            shelephant_dump(["-i", "--all"])
            data = shelephant.dataset.Location.from_yaml(f_dump)
            check.remove(nested)
            self.assertTrue(check == data)

        with tempdir():
            pathlib.Path("src").mkdir()
            nested = [os.path.join("src", i) for i in ["f.txt", "g.txt"]]
            files = ["foo.txt", "bar.txt", "a.txt", "b.txt", "c.txt", "d.txt"] + nested
            check = create_dummy_files(files)
            shelephant_dump(["-i", "--all", "--recursive"])
            data = shelephant.dataset.Location.from_yaml(f_dump)
            self.assertTrue(check == data)

    def test_raw(self):
        with tempdir():
            names = ["./a.txt", "../foo/../b.txt"]
            shelephant_dump(["--raw"] + names)
            data = shelephant.yaml.read(f_dump)
            self.assertEqual(data, names)

            shelephant_dump(["-f"] + names)
            data = shelephant.yaml.read(f_dump)
            self.assertEqual(data, ["a.txt", os.path.join("..", "b.txt")])

    def test_abspath(self):
        with tempdir():
            root = pathlib.Path(".").absolute()
            files = ["foo.txt", "bar.txt", "a.txt", "b.txt", "c.txt", "d.txt"]
            shelephant_dump(["--abspath"] + files)
            data = shelephant.yaml.read(f_dump)
            check = [str(root / i) for i in files]
            self.assertEqual(data, check)


class Test_shelephant_hostinfo(unittest.TestCase):
    def test_search(self):
        with tempdir():
            files = ["foo.txt", "bar.txt"]
            check = create_dummy_files(files)
            shelephant.yaml.dump("info.yaml", {"root": ".", "search": [{"rglob": "*.txt"}]})
            shelephant_hostinfo(["-iu", "info.yaml"])
            loc = shelephant.dataset.Location.from_yaml("info.yaml")
            self.assertTrue(check == loc)


class Test_shelephant_cp(unittest.TestCase):
    def test_basic(self):
        """
        shelephant_cp <sourceinfo.yaml> <dest_dirname>
        """
        with tempdir(), contextlib.redirect_stdout(io.StringIO()) as sio:
            pathlib.Path("src").mkdir()
            pathlib.Path("dest").mkdir()

            with cwd("dest"):
                create_dummy_files(["foo.txt"])
                create_dummy_files(["bar.txt"], keep=slice(2, None, None))

            with cwd("src"):
                files = ["foo.txt", "bar.txt", "more.txt", "even_more.txt"]
                check = create_dummy_files(files)
                shelephant_dump(files)
                args = [f_dump, "../dest"]
                shelephant_cp(["-n", "--colors", "none"] + args)
                shelephant_cp(["-f", "--quiet"] + args)

            data = shelephant.dataset.Location(root="dest", files=files).getinfo()
            self.assertTrue(check == data)

        if has_rsync:
            expect = [
                "bar.txt => bar.txt",
                "more.txt -> more.txt",
                "even_more.txt -> even_more.txt",
                "foo.txt == foo.txt",
            ]
        else:
            expect = [
                "foo.txt => foo.txt",
                "bar.txt => bar.txt",
                "more.txt -> more.txt",
                "even_more.txt -> even_more.txt",
            ]
        ret = _plain(sio.getvalue())
        self.assertEqual(ret, expect)

    def test_basic_basic(self):
        """
        shelephant_cp <sourceinfo.yaml> <dest_dirname> --mode basic
        """
        with tempdir(), contextlib.redirect_stdout(io.StringIO()) as sio:
            pathlib.Path("src").mkdir()
            pathlib.Path("dest").mkdir()

            with cwd("dest"):
                create_dummy_files(["foo.txt"])
                create_dummy_files(["bar.txt"], keep=slice(2, None, None))

            with cwd("src"):
                files = ["foo.txt", "bar.txt", "more.txt", "even_more.txt"]
                check = create_dummy_files(files)
                shelephant_dump(files)
                args = [f_dump, "../dest", "--mode", "basic"]
                shelephant_cp(["-n", "--colors", "none"] + args)
                shelephant_cp(["-f", "--quiet"] + args)

            data = shelephant.dataset.Location(root="dest", files=files).getinfo()
            self.assertTrue(check == data)

        expect = [
            "foo.txt => foo.txt",
            "bar.txt => bar.txt",
            "more.txt -> more.txt",
            "even_more.txt -> even_more.txt",
        ]
        ret = _plain(sio.getvalue())
        self.assertEqual(ret, expect)

    def test_destinfo(self):
        """
        shelephant_cp <sourceinfo.yaml> <destinfo.yaml>
        """
        with tempdir(), contextlib.redirect_stdout(io.StringIO()) as sio:
            pathlib.Path("src").mkdir()
            pathlib.Path("dest").mkdir()

            with cwd("dest"):
                create_dummy_files(["foo.txt"])
                create_dummy_files(["bar.txt"], keep=slice(2, None, None))

            with cwd("src"):
                files = ["foo.txt", "bar.txt", "more.txt", "even_more.txt"]
                check = create_dummy_files(files)
                shelephant_dump(files)
                shelephant_hostinfo(["../dest"])
                args = [f_dump, f_hostinfo]
                shelephant_cp(["-n", "--colors", "none"] + args)
                shelephant_cp(["-f", "--quiet"] + args)

            data = shelephant.dataset.Location(root="dest", files=files).getinfo()
            self.assertTrue(check == data)

        if has_rsync:
            expect = [
                "bar.txt => bar.txt",
                "more.txt -> more.txt",
                "even_more.txt -> even_more.txt",
                "foo.txt == foo.txt",
            ]
        else:
            expect = [
                "foo.txt => foo.txt",
                "bar.txt => bar.txt",
                "more.txt -> more.txt",
                "even_more.txt -> even_more.txt",
            ]
        ret = _plain(sio.getvalue())
        self.assertEqual(ret, expect)

    def test_destinfo_sha256(self):
        """
        shelephant_cp <sourceinfo.yaml> <destinfo.yaml>
        """
        with tempdir(), contextlib.redirect_stdout(io.StringIO()) as sio:
            pathlib.Path("src").mkdir()
            pathlib.Path("dest").mkdir()

            with cwd("dest"):
                create_dummy_files(["foo.txt"])
                create_dummy_files(["bar.txt"], keep=slice(2, None, None))
                shelephant_dump(["-i", "foo.txt", "bar.txt"])

            with cwd("src"):
                files = ["foo.txt", "bar.txt", "more.txt", "even_more.txt"]
                check = create_dummy_files(files)
                shelephant_dump(["-i"] + files)
                shelephant_hostinfo(["../dest", "-d"])
                args = [f_dump, f_hostinfo]
                shelephant_cp(["-n", "--colors", "none"] + args)
                shelephant_cp(["-f", "--quiet"] + args)

            data = shelephant.dataset.Location(root="dest", files=files).getinfo()
            self.assertTrue(check == data)

        expect = [
            "bar.txt => bar.txt",
            "more.txt -> more.txt",
            "even_more.txt -> even_more.txt",
            "foo.txt == foo.txt",
        ]
        ret = _plain(sio.getvalue())
        self.assertEqual(ret, expect)

    def test_destinfo_sha256_2(self):
        """
        shelephant_cp <sourceinfo.yaml> <destinfo.yaml>
        """
        with tempdir(), contextlib.redirect_stdout(io.StringIO()) as sio:
            pathlib.Path("src").mkdir()
            pathlib.Path("dest").mkdir()

            with cwd("dest"):
                create_dummy_files(["foo.txt"])
                create_dummy_files(["bar.txt"], keep=slice(2, None, None))
                shelephant_dump(["foo.txt", "bar.txt"])

            with cwd("src"):
                files = ["foo.txt", "bar.txt", "more.txt", "even_more.txt"]
                check = create_dummy_files(files)
                shelephant_dump(["-i"] + files)
                shelephant_hostinfo(["../dest", "-d", "--info"])
                args = [f_dump, f_hostinfo]
                shelephant_cp(["-n", "--colors", "none"] + args)
                shelephant_cp(["-f", "--quiet"] + args)

            data = shelephant.dataset.Location(root="dest", files=files).getinfo()
            self.assertTrue(check == data)

        expect = [
            "bar.txt => bar.txt",
            "more.txt -> more.txt",
            "even_more.txt -> even_more.txt",
            "foo.txt == foo.txt",
        ]
        ret = _plain(sio.getvalue())
        self.assertEqual(ret, expect)

    def test_sourceinfo(self):
        """
        shelephant_cp <sourceinfo.yaml> <dest_dirname>
        """
        with tempdir(), contextlib.redirect_stdout(io.StringIO()) as sio:
            pathlib.Path("src").mkdir()
            pathlib.Path("dest").mkdir()

            with cwd("dest"):
                create_dummy_files(["foo.txt"])
                create_dummy_files(["bar.txt"], keep=slice(2, None, None))

            with cwd("src"):
                files = ["foo.txt", "bar.txt", "more.txt", "even_more.txt"]
                check = create_dummy_files(files)
                shelephant_dump(files)

            with cwd("dest"):
                shelephant_hostinfo(["../src", "-d"])
                args = [f_hostinfo, "."]
                shelephant_cp(["-n", "--colors", "none"] + args)
                shelephant_cp(["-f", "--quiet"] + args)
                shelephant_dump(["-i", "-c", "find . -name '*.txt'"])
                data = shelephant.dataset.Location.from_yaml(f_dump)

            self.assertTrue(check == data)

        if has_rsync:
            expect = [
                "bar.txt => bar.txt",
                "more.txt -> more.txt",
                "even_more.txt -> even_more.txt",
                "foo.txt == foo.txt",
            ]
        else:
            expect = [
                "foo.txt => foo.txt",
                "bar.txt => bar.txt",
                "more.txt -> more.txt",
                "even_more.txt -> even_more.txt",
            ]
        ret = _plain(sio.getvalue())
        self.assertEqual(ret, expect)

    def test_ssh_send(self):
        """
        shelephant_cp <sourceinfo.yaml> <dest_dirname_on_host> --ssh <user@host>
        """
        if not has_ssh:
            raise unittest.SkipTest("'ssh localhost' does not work")

        with (
            tempdir(),
            shelephant.ssh.tempdir("localhost") as remote,
            contextlib.redirect_stdout(io.StringIO()) as sio,
        ):
            files = ["foo.txt", "bar.txt", "more.txt", "even_more.txt"]
            check = create_dummy_files(files)
            shelephant_dump(files)

            args = [f_dump, str(remote), "--ssh", "localhost"]
            shelephant_cp(["-n", "--colors", "none"] + args)
            shelephant_cp(["-f", "--quiet"] + args)
            data = shelephant.dataset.Location(root=remote, files=files).getinfo()

        self.assertTrue(check == data)

        expect = [
            "foo.txt -> foo.txt",
            "bar.txt -> bar.txt",
            "more.txt -> more.txt",
            "even_more.txt -> even_more.txt",
        ]
        ret = _plain(sio.getvalue())
        self.assertEqual(ret, expect)

    def test_ssh_send2(self):
        """
        shelephant_cp <sourceinfo.yaml> <remote_destinfo.yaml>
        """
        if not has_ssh:
            raise unittest.SkipTest("'ssh localhost' does not work")

        with (
            tempdir(),
            shelephant.ssh.tempdir("localhost") as remote,
            contextlib.redirect_stdout(io.StringIO()) as sio,
        ):
            files = ["foo.txt", "bar.txt", "more.txt", "even_more.txt"]
            check = create_dummy_files(files)
            shelephant_dump(files)
            shelephant_hostinfo([str(remote), "--ssh", "localhost"])

            args = [f_dump, f_hostinfo]
            shelephant_cp(["-n", "--colors", "none"] + args)
            shelephant_cp(["-f", "--quiet"] + args)
            data = shelephant.dataset.Location(root=remote, files=files).getinfo()

        self.assertTrue(check == data)

        expect = [
            "foo.txt -> foo.txt",
            "bar.txt -> bar.txt",
            "more.txt -> more.txt",
            "even_more.txt -> even_more.txt",
        ]
        ret = _plain(sio.getvalue())
        self.assertEqual(ret, expect)

    def test_ssh_get(self):
        """
        shelephant_cp <remote_sourceinfo.yaml> <dest_dirname>
        """
        if not has_ssh:
            raise unittest.SkipTest("'ssh localhost' does not work")

        with (
            tempdir(),
            shelephant.ssh.tempdir("localhost") as remote,
            contextlib.redirect_stdout(io.StringIO()) as sio,
        ):
            with cwd(remote):
                files = ["foo.txt", "bar.txt", "more.txt", "even_more.txt"]
                check = create_dummy_files(files)
                shelephant_dump(files)

            shelephant_hostinfo([str(remote), "--ssh", "localhost", "-d"])
            args = [f_hostinfo, "."]
            shelephant_cp(["-n", "--colors", "none"] + args)
            shelephant_cp(["-f", "--quiet"] + args)
            data = shelephant.dataset.Location(root=".", files=files).getinfo()

        self.assertTrue(check == data)

        expect = [
            "foo.txt -> foo.txt",
            "bar.txt -> bar.txt",
            "more.txt -> more.txt",
            "even_more.txt -> even_more.txt",
        ]
        ret = _plain(sio.getvalue())
        self.assertEqual(ret, expect)


class Test_shelephant_mv(unittest.TestCase):
    def test_basic(self):
        """
        shelephant_mv <sourceinfo.yaml> <dest_dirname>
        """
        with tempdir(), contextlib.redirect_stdout(io.StringIO()) as sio:
            pathlib.Path("src").mkdir()
            pathlib.Path("dest").mkdir()

            with cwd("dest"):
                create_dummy_files(["foo.txt"])
                create_dummy_files(["bar.txt"], keep=slice(2, None, None))

            with cwd("src"):
                files = ["foo.txt", "bar.txt", "more.txt", "even_more.txt"]
                check = create_dummy_files(files)
                shelephant_dump(files)
                args = [f_dump, "../dest"]
                shelephant_mv(["-n", "--colors", "none"] + args)
                shelephant_mv(["-f", "--quiet"] + args)
                self.assertFalse(any([os.path.exists(f) for f in files]))

            data = shelephant.dataset.Location(root="dest", files=files).getinfo()
            self.assertTrue(check == data)

        expect = [
            "foo.txt => foo.txt",
            "bar.txt => bar.txt",
            "more.txt -> more.txt",
            "even_more.txt -> even_more.txt",
        ]
        ret = _plain(sio.getvalue())
        self.assertEqual(ret, expect)


class Test_shelephant_rm(unittest.TestCase):
    def test_basic(self):
        """
        shelephant_rm <sourceinfo.yaml>
        """
        with tempdir(), contextlib.redirect_stdout(io.StringIO()) as sio:
            pathlib.Path("src").mkdir()

            files = ["foo.txt", "bar.txt", "more.txt", "even_more.txt"]
            create_dummy_files(files)
            shelephant_dump(files)
            args = [f_dump]
            shelephant_rm(["-n"] + args)
            shelephant_rm(["-f", "--quiet"] + args)
            self.assertFalse(any([os.path.exists(f) for f in files]))

        expect = [
            "rm foo.txt",
            "rm bar.txt",
            "rm more.txt",
            "rm even_more.txt",
        ]
        ret = _plain(sio.getvalue())
        self.assertEqual(ret, expect)

    def test_ssh(self):
        """
        shelephant_rm <sourceinfo.yaml>
        """
        if not has_ssh:
            raise unittest.SkipTest("'ssh localhost' does not work")

        with (
            tempdir(),
            shelephant.ssh.tempdir("localhost") as remote,
            contextlib.redirect_stdout(io.StringIO()) as sio,
        ):
            with cwd(remote):
                files = ["foo.txt", "bar.txt", "more.txt", "even_more.txt"]
                create_dummy_files(files)
                shelephant_dump(files)

            shelephant_hostinfo([str(remote), "--ssh", "localhost", "-d"])
            args = [f_hostinfo]
            shelephant_rm(["-n"] + args)
            shelephant_rm(["-f", "--quiet"] + args)
            self.assertFalse(any([os.path.exists(remote / f) for f in files]))

        expect = [
            "rm foo.txt",
            "rm bar.txt",
            "rm more.txt",
            "rm even_more.txt",
        ]
        ret = _plain(sio.getvalue())
        self.assertEqual(ret, expect)


class Test_shelephant_diff(unittest.TestCase):
    def test_output_sha256(self):
        """
        shelephant_diff <sourceinfo.yaml> <destinfo.yaml>
        """
        with tempdir():
            pathlib.Path("src").mkdir()
            pathlib.Path("dest").mkdir()

            with cwd("dest"):
                create_dummy_files(["foo.txt"])
                create_dummy_files(["bar.txt"], keep=slice(2, None, None))
                create_dummy_files(["receive.txt"], keep=slice(6, None, None))
                shelephant_dump(["foo.txt", "bar.txt", "receive.txt"])

            with cwd("src"):
                files = ["foo.txt", "bar.txt", "more.txt", "even_more.txt"]
                create_dummy_files(files)
                shelephant_dump(["-i"] + files)
                shelephant_hostinfo(["../dest", "-d", "--info"])
                shelephant_diff([f_dump, f_hostinfo, "-o", "foo.yaml"])
                data = shelephant.yaml.read("foo.yaml")

        expect = {
            "!=": ["bar.txt"],
            "->": ["even_more.txt", "more.txt"],
            "<-": ["receive.txt"],
            "==": ["foo.txt"],
        }
        self.assertEqual(data, expect)

    def test_output(self):
        """
        shelephant_diff <sourceinfo.yaml> <destinfo.yaml>
        """
        with tempdir():
            pathlib.Path("src").mkdir()
            pathlib.Path("dest").mkdir()

            with cwd("dest"):
                create_dummy_files(["foo.txt"])
                create_dummy_files(["bar.txt"], keep=slice(2, None, None))
                create_dummy_files(["receive.txt"], keep=slice(6, None, None))

            with cwd("src"):
                files = ["foo.txt", "bar.txt", "more.txt", "even_more.txt"]
                create_dummy_files(files)
                shelephant_dump(files)
                args = [f_dump, "../dest"]
                if has_rsync:
                    shelephant_diff(args + ["-o", "rsync.yaml", "--mode", "rsync"])
                    data_rsync = shelephant.yaml.read("rsync.yaml")
                shelephant_diff(args + ["-o", "basic.yaml", "--mode", "basic"])
                data_basic = shelephant.yaml.read("basic.yaml")

        if has_rsync:
            expect = {
                "!=": ["bar.txt"],
                "->": ["more.txt", "even_more.txt"],
                "==": ["foo.txt"],
            }
            self.assertEqual(data_rsync, expect)

        expect = {
            "?=": ["foo.txt", "bar.txt"],
            "->": ["more.txt", "even_more.txt"],
        }
        self.assertEqual(data_basic, expect)

    def test_output_filter(self):
        """
        shelephant_diff <sourceinfo.yaml> <destinfo.yaml>
        """
        with tempdir():
            pathlib.Path("src").mkdir()
            pathlib.Path("dest").mkdir()

            with cwd("dest"):
                create_dummy_files(["foo.txt"])
                create_dummy_files(["bar.txt"], keep=slice(2, None, None))
                create_dummy_files(["receive.txt"], keep=slice(6, None, None))
                shelephant_dump(["foo.txt", "bar.txt", "receive.txt"])

            with cwd("src"):
                files = ["foo.txt", "bar.txt", "more.txt", "even_more.txt"]
                create_dummy_files(files)
                shelephant_dump(["-i"] + files)
                shelephant_hostinfo(["../dest", "-d", "--info"])
                args = [f_dump, f_hostinfo]
                shelephant_diff(args + ["-o", "foo.yaml", "--filter", "<-, ->"])
                data = shelephant.yaml.read("foo.yaml")

        expect = {
            "->": ["even_more.txt", "more.txt"],
            "<-": ["receive.txt"],
        }
        self.assertEqual(data, expect)

    def test_output_filter_list(self):
        """
        shelephant_diff <sourceinfo.yaml> <destinfo.yaml>
        """
        with tempdir():
            pathlib.Path("src").mkdir()
            pathlib.Path("dest").mkdir()

            with cwd("dest"):
                create_dummy_files(["foo.txt"])
                create_dummy_files(["bar.txt"], keep=slice(2, None, None))
                create_dummy_files(["receive.txt"], keep=slice(6, None, None))
                shelephant_dump(["foo.txt", "bar.txt", "receive.txt"])

            with cwd("src"):
                files = ["foo.txt", "bar.txt", "more.txt", "even_more.txt"]
                create_dummy_files(files)
                shelephant_dump(["-i"] + files)
                shelephant_hostinfo(["../dest", "-d", "--info"])
                args = [f_dump, f_hostinfo]
                shelephant_diff(args + ["-o", "foo.yaml", "--filter", "<-", "--list"])
                data = shelephant.yaml.read("foo.yaml")

        self.assertEqual(data, ["receive.txt"])

    def test_table(self):
        """
        shelephant_diff <sourceinfo.yaml> <destinfo.yaml>
        """
        with tempdir(), contextlib.redirect_stdout(io.StringIO()) as sio:
            pathlib.Path("src").mkdir()
            pathlib.Path("dest").mkdir()

            with cwd("dest"):
                files = ["foo.txt", "bar.txt", "receive.txt"]
                create_dummy_files([files[0]])
                create_dummy_files([files[1]], keep=slice(2, None, None))
                create_dummy_files([files[2]], keep=slice(6, None, None))
                shelephant_dump(["-i"] + files)

            with cwd("src"):
                files = ["foo.txt", "bar.txt", "more.txt", "even_more.txt"]
                create_dummy_files(files)
                shelephant_dump(["-i"] + files)
                shelephant_hostinfo(["../dest", "-d"])
                shelephant_diff([f_dump, f_hostinfo, "--colors", "none"])

        expect = [
            "bar.txt != bar.txt",
            "receive.txt <- receive.txt",
            "even_more.txt -> even_more.txt",
            "more.txt -> more.txt",
            "foo.txt == foo.txt",
        ]

        ret = _plain(sio.getvalue())
        self.assertEqual(ret, expect)


if __name__ == "__main__":
    unittest.main()
