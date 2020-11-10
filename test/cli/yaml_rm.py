import subprocess
import os

def run(cmd):
    return subprocess.check_output(cmd, shell=True).decode('utf-8')

open('foo.txt', 'w').write('foo')
open('bar.txt', 'w').write('bar')

output = run("yaml_dump -f foo.txt bar.txt")
output = run("yaml_rm -f dump.yaml")

assert not os.path.isfile('foo.txt')
assert not os.path.isfile('bar.txt')
