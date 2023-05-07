import os
import pathlib
import shutil
import subprocess

import shelephant


def run(cmd):
    return subprocess.check_output(cmd, shell=True).decode("utf-8")


for dirname in ["myssh_send", "myssh_get"]:
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)

    os.mkdir(dirname)

pathlib.Path("myssh_send/bar.txt").write_text("bar")
pathlib.Path("myssh_send/foo.txt").write_text("foo")
pathlib.Path("myssh_get/foo.txt").write_text("foo")

shelephant.shelephant_dump(
    ["-o", "myssh_send/shelephant_dump.yaml", "myssh_send/bar.txt", "myssh_send/foo.txt"]
)
shelephant.shelephant_dump(["-o", "myssh_get/shelephant_dump.yaml", "myssh_get/foo.txt"])

run("shelephant_checksum -o myssh_send/shelephant_checksum.yaml myssh_send/shelephant_dump.yaml")
run("shelephant_checksum -o myssh_get/shelephant_checksum.yaml myssh_get/shelephant_dump.yaml")
