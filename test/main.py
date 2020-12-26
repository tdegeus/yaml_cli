import unittest
import subprocess
import os
import shutil
from shelephant.cli import YamlGetItem
from shelephant.cli import YamlRead
from shelephant.cli import YamlDump


def run(cmd):
    return subprocess.check_output(cmd, shell=True).decode('utf-8')


class Test_tools(unittest.TestCase):

    def test_FlattenList(self):

        from shelephant.cli import FlattenList

        arg = [1, [2, 2, 2], 4]
        ret = [1, 2, 2, 2, 4]

        self.assertEqual(ret, FlattenList(arg))

    def test_Squash(self):

        from shelephant.cli import Squash

        arg = {'foo': [1, 2], 'bar': {'foo': [3, 4], 'bar': 5}}
        ret = [1, 2, 3, 4, 5]

        self.assertEqual(ret, Squash(arg))


class Test_checksum(unittest.TestCase):

    def test_basic(self):

        with open('foo.txt', 'w') as file:
            file.write('foo')

        with open('bar.txt', 'w') as file:
            file.write('bar')

        output = run('shelephant_dump foo.txt bar.txt')
        output = run('shelephant_checksum shelephant_dump.yaml')
        data = YamlGetItem('shelephant_checksum.yaml')

        keys = [
            '2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae',
            'fcde2b2edba56bf408601fb721fe9b5c338d10ee429ea04fae5511b68fbf8fb9',
        ]

        self.assertEqual(data, keys)

        os.remove('foo.txt')
        os.remove('bar.txt')
        os.remove('shelephant_dump.yaml')
        os.remove('shelephant_checksum.yaml')


class Test_dump(unittest.TestCase):

    def test_basic(self):

        with open('foo.txt', 'w') as file:
            file.write('foo')

        with open('bar.txt', 'w') as file:
            file.write('bar')

        for dirname in ['mydir']:
            if os.path.isdir(dirname):
                shutil.rmtree(dirname)

        os.mkdir('mydir')

        with open('mydir/foo.txt', 'w') as file:
            file.write('foo')

        with open('mydir/bar.txt', 'w') as file:
            file.write('bar')

        output = run('shelephant_dump -s -o dump_1.yaml foo.txt bar.txt')
        output = run('shelephant_dump -s -o dump_2.yaml *.txt')
        output = run('shelephant_dump -s -o mydir/dump_3.yaml mydir/*.txt')

        with open('dump_1.yaml', 'r') as file:
            dump_1 = file.read()

        with open('dump_2.yaml', 'r') as file:
            dump_2 = file.read()

        with open('mydir/dump_3.yaml', 'r') as file:
            dump_3 = file.read()

        self.assertEqual(dump_1, dump_2)
        self.assertEqual(dump_1, dump_3)

        os.remove('foo.txt')
        os.remove('bar.txt')
        os.remove('dump_1.yaml')
        os.remove('dump_2.yaml')
        shutil.rmtree('mydir')

    def test_append(self):

        with open('foo.txt', 'w') as file:
            file.write('foo')

        with open('bar.txt', 'w') as file:
            file.write('bar')

        with open('foo.pdf', 'w') as file:
            file.write('foo')

        with open('bar.pdf', 'w') as file:
            file.write('bar')

        output = run('shelephant_dump foo.txt bar.txt')
        output = run('shelephant_dump -a foo.pdf bar.pdf')

        self.assertEqual(YamlRead('shelephant_dump.yaml'), ['foo.txt', 'bar.txt', 'foo.pdf', 'bar.pdf'])

        os.remove('foo.txt')
        os.remove('bar.txt')
        os.remove('foo.pdf')
        os.remove('bar.pdf')
        os.remove('shelephant_dump.yaml')


class Test_extract(unittest.TestCase):

    def test_single_path(self):

        data = {
            'foo' : ['foo.txt', 'bar.txt'],
            'bar' : ['foo.pdf', 'bar.pdf'],
            'key' : ['foo.key', 'bar.key'],
        }

        YamlDump('dump.yaml', data, force=True)

        output = run('shelephant_extract --force dump.yaml "foo"')

        self.assertEqual(YamlRead('dump.yaml'), ['foo.txt', 'bar.txt'])

        os.remove('dump.yaml')

    def test_multiple_paths(self):

        data = {
            'foo' : ['foo.txt', 'bar.txt'],
            'bar' : ['foo.pdf', 'bar.pdf'],
            'key' : ['foo.key', 'bar.key'],
            'sub' : {
                'foo' : ['foo.txt', 'bar.txt'],
                'bar' : ['foo.pdf', 'bar.pdf'],
                'key' : ['foo.key', 'bar.key'],
            },
        }

        YamlDump('dump.yaml', data, force=True)

        output = run('shelephant_extract --force dump.yaml "/sub/foo" "foo"')

        self.assertEqual(YamlRead('dump.yaml'), {'foo': ['foo.txt', 'bar.txt'], 'sub': {'foo': ['foo.txt', 'bar.txt']}})

        os.remove('dump.yaml')

    def test_multiple_paths_squash(self):

        data = {
            'foo' : ['foo.txt', 'bar.txt'],
            'bar' : ['foo.pdf', 'bar.pdf'],
            'key' : ['foo.key', 'bar.key'],
            'sub' : {
                'foo' : ['foo2.txt', 'bar2.txt'],
                'bar' : ['foo2.pdf', 'bar2.pdf'],
                'key' : ['foo2.key', 'bar2.key'],
            },
        }

        YamlDump('dump.yaml', data, force=True)

        output = run('shelephant_extract --force dump.yaml --squash "/sub/foo" "foo"')

        self.assertEqual(YamlRead('dump.yaml'), ['foo2.txt', 'bar2.txt', 'foo.txt', 'bar.txt'])

        os.remove('dump.yaml')


class Test_merge(unittest.TestCase):

    def test_basic(self):

        with open('foo.txt', 'w') as file:
            file.write('foo')

        with open('bar.txt', 'w') as file:
            file.write('bar')

        output = run('shelephant_dump -o main.yaml foo.txt')
        output = run('shelephant_dump -o branch.yaml bar.txt')
        output = run('shelephant_merge --force branch.yaml main.yaml')

        self.assertEqual(YamlRead('main.yaml'), ['foo.txt', 'bar.txt'])

        os.remove('foo.txt')
        os.remove('bar.txt')
        os.remove('main.yaml')
        os.remove('branch.yaml')

        os.mkdir('dira')
        os.mkdir('dirb')

        with open('dira/foo.txt', 'w') as file:
            file.write('foo')

        with open('dira/bar.txt', 'w') as file:
            file.write('bar')

        with open('dirb/foo.txt', 'w') as file:
            file.write('foo')

        with open('dirb/bar.txt', 'w') as file:
            file.write('bar')

        output = run('shelephant_dump -o dira/dump.yaml dira/foo.txt dira/bar.txt')
        output = run('shelephant_dump -o dirb/dump.yaml dirb/foo.txt dirb/bar.txt')
        output = run('shelephant_merge --force dira/dump.yaml dirb/dump.yaml')

        self.assertEqual(YamlRead('dirb/dump.yaml'), ['foo.txt', 'bar.txt', '../dira/foo.txt', '../dira/bar.txt'])

        shutil.rmtree('dira')
        shutil.rmtree('dirb')


class Test_remote(unittest.TestCase):

    def test_basic(self):

        for dirname in ['mysrc', 'mydest']:
            if os.path.isdir(dirname):
                shutil.rmtree(dirname)

        os.mkdir('mysrc')
        os.mkdir('mydest')

        with open('mysrc/foo.txt', 'w') as file:
            file.write('foo')

        with open('mysrc/bar.txt', 'w') as file:
            file.write('bar')

        output = run('shelephant_dump --sort -o mysrc/files.yaml mysrc/*.txt')
        output = run('shelephant_checksum -o mysrc/checksum.yaml mysrc/files.yaml')
        output = run('shelephant_remote -o mydest/remote.yaml --files mysrc/files.yaml --checksum mysrc/checksum.yaml')
        output = run('shelephant_get --force mydest/remote.yaml')
        output = run('shelephant_dump --sort -o mydest/files.yaml mydest/*.txt')
        output = run('shelephant_checksum -o mydest/checksum.yaml mydest/files.yaml')

        self.assertEqual(YamlRead('mysrc/files.yaml'), YamlRead('mydest/files.yaml'))
        self.assertEqual(YamlRead('mysrc/checksum.yaml'), YamlRead('mydest/checksum.yaml'))

        shutil.rmtree('mysrc')
        shutil.rmtree('mydest')


class Test_get(unittest.TestCase):

    def test_basic(self):

        for dirname in ['mysrc', 'mydest']:
            if os.path.isdir(dirname):
                shutil.rmtree(dirname)

        os.mkdir('mysrc')
        os.mkdir('mydest')

        with open('mysrc/foo.txt', 'w') as file:
            file.write('foo')

        with open('mysrc/bar.txt', 'w') as file:
            file.write('bar')

        operations = [
            'bar.txt -> bar.txt',
            'foo.txt -> foo.txt',
            'mydest/bar.txt',
            'mydest/foo.txt',
        ]

        output = run('shelephant_dump --sort -o mysrc/files.yaml mysrc/*.txt')
        output = run('shelephant_checksum -o mysrc/checksum.yaml mysrc/files.yaml')
        output = run('shelephant_remote -o mydest/remote.yaml --files mysrc/files.yaml --checksum mysrc/checksum.yaml')
        output = run('shelephant_get --colors none --force mydest/remote.yaml')

        output = output.split('\n')
        output = output[4:-1]
        output[-2] = output[-2].split(') ')[1]
        output[-1] = output[-1].split(') ')[1]
        self.assertEqual(output, operations)

        output = run('shelephant_dump --sort -o mydest/files.yaml mydest/*.txt')
        output = run('shelephant_checksum -o mydest/checksum.yaml mydest/files.yaml')

        self.assertEqual(YamlRead('mysrc/files.yaml'), YamlRead('mydest/files.yaml'))
        self.assertEqual(YamlRead('mysrc/checksum.yaml'), YamlRead('mydest/checksum.yaml'))

        shutil.rmtree('mysrc')
        shutil.rmtree('mydest')

    def test_partial(self):

        for dirname in ['mysrc', 'mydest']:
            if os.path.isdir(dirname):
                shutil.rmtree(dirname)

        os.mkdir('mysrc')
        os.mkdir('mydest')

        with open('mysrc/foo.txt', 'w') as file:
            file.write('foo')

        with open('mysrc/bar.txt', 'w') as file:
            file.write('bar')

        shutil.copy('mysrc/foo.txt', 'mydest/foo.txt')

        operations = [
            'bar.txt -> bar.txt',
            'foo.txt == foo.txt',
            'mydest/bar.txt',
        ]

        output = run('shelephant_dump --sort -o mysrc/files.yaml mysrc/*.txt')
        output = run('shelephant_checksum -o mysrc/checksum.yaml mysrc/files.yaml')
        output = run('shelephant_remote -o mydest/remote.yaml --files mysrc/files.yaml --checksum mysrc/checksum.yaml')
        output = run('shelephant_get --colors none --force mydest/remote.yaml')

        output = output.split('\n')
        output = output[4:-1]
        output[-1] = output[-1].split(') ')[1]
        self.assertEqual(output, operations)

        output = run('shelephant_dump --sort -o mydest/files.yaml mydest/*.txt')
        output = run('shelephant_checksum -o mydest/checksum.yaml mydest/files.yaml')

        self.assertEqual(YamlRead('mysrc/files.yaml'), YamlRead('mydest/files.yaml'))
        self.assertEqual(YamlRead('mysrc/checksum.yaml'), YamlRead('mydest/checksum.yaml'))

        shutil.rmtree('mysrc')
        shutil.rmtree('mydest')

    def test_partial_localchecksum(self):

        for dirname in ['mysrc', 'mydest']:
            if os.path.isdir(dirname):
                shutil.rmtree(dirname)

        os.mkdir('mysrc')
        os.mkdir('mydest')

        with open('mysrc/foo.txt', 'w') as file:
            file.write('foo')

        with open('mysrc/bar.txt', 'w') as file:
            file.write('bar')

        shutil.copy('mysrc/foo.txt', 'mydest/foo.txt')

        operations = [
            'bar.txt -> bar.txt',
            'foo.txt == foo.txt',
            'mydest/bar.txt',
        ]

        output = run('shelephant_dump --sort -o mysrc/files.yaml mysrc/*.txt')
        output = run('shelephant_checksum -o mysrc/checksum.yaml mysrc/files.yaml')
        output = run('shelephant_dump --sort -o mydest/files.yaml mydest/*.txt')
        output = run('shelephant_checksum -o mydest/checksum.yaml mydest/files.yaml')
        output = run('shelephant_remote -o mydest/remote.yaml --files mysrc/files.yaml --checksum mysrc/checksum.yaml')
        output = run('shelephant_remote -o mydest/local.yaml --files mydest/files.yaml --checksum mydest/checksum.yaml')
        output = run('shelephant_get --colors none --force --local mydest/local.yaml mydest/remote.yaml')

        output = output.split('\n')
        output = output[4:-1]
        output[-1] = output[-1].split(') ')[1]
        self.assertEqual(output, operations)

        output = run('shelephant_dump --sort --force -o mydest/files.yaml mydest/*.txt')
        output = run('shelephant_checksum --force -o mydest/checksum.yaml mydest/files.yaml')

        self.assertEqual(YamlRead('mysrc/files.yaml'), YamlRead('mydest/files.yaml'))
        self.assertEqual(YamlRead('mysrc/checksum.yaml'), YamlRead('mydest/checksum.yaml'))

        shutil.rmtree('mysrc')
        shutil.rmtree('mydest')


class Test_mv(unittest.TestCase):

    def test_basic(self):

        for dirname in ['mysrc', 'mydest']:
            if os.path.isdir(dirname):
                shutil.rmtree(dirname)

        os.mkdir('mysrc')
        os.mkdir('mydest')

        with open('mysrc/foo.txt', 'w') as file:
            file.write('foo')

        with open('mysrc/bar.txt', 'w') as file:
            file.write('bar')

        output = run('shelephant_dump -o mysrc/files.yaml mysrc/*.txt')
        output = run('shelephant_checksum -o mysrc/checksum.yaml mysrc/files.yaml')
        output = run('shelephant_mv --force mysrc/files.yaml mydest')
        output = run('shelephant_dump --sort -o mydest/files.yaml mydest/*.txt')
        output = run('shelephant_checksum -o mydest/checksum.yaml mydest/files.yaml')

        self.assertEqual(YamlRead('mysrc/files.yaml'), YamlRead('mydest/files.yaml'))
        self.assertEqual(YamlRead('mysrc/checksum.yaml'), YamlRead('mydest/checksum.yaml'))

        shutil.rmtree('mysrc')
        shutil.rmtree('mydest')


class Test_cp(unittest.TestCase):

    def test_basic(self):

        for dirname in ['mysrc', 'mydest']:
            if os.path.isdir(dirname):
                shutil.rmtree(dirname)

        os.mkdir('mysrc')
        os.mkdir('mydest')

        with open('mysrc/foo.txt', 'w') as file:
            file.write('foo')

        with open('mysrc/bar.txt', 'w') as file:
            file.write('bar')

        output = run('shelephant_dump -o mysrc/files.yaml mysrc/*.txt')
        output = run('shelephant_checksum -o mysrc/checksum.yaml mysrc/files.yaml')
        output = run('shelephant_cp --force mysrc/files.yaml mydest')
        output = run('shelephant_dump --sort -o mydest/files.yaml mydest/*.txt')
        output = run('shelephant_checksum -o mydest/checksum.yaml mydest/files.yaml')

        self.assertEqual(YamlRead('mysrc/files.yaml'), YamlRead('mydest/files.yaml'))
        self.assertEqual(YamlRead('mysrc/checksum.yaml'), YamlRead('mydest/checksum.yaml'))
        self.assertTrue(os.path.isfile('mysrc/foo.txt'))
        self.assertTrue(os.path.isfile('mysrc/bar.txt'))

        shutil.rmtree('mysrc')
        shutil.rmtree('mydest')


class Test_rm(unittest.TestCase):

    def test_basic(self):

        with open('foo.txt', 'w') as file:
            file.write('foo')

        with open('bar.txt', 'w') as file:
            file.write('bar')

        output = run('shelephant_dump foo.txt bar.txt')
        output = run('shelephant_rm --force shelephant_dump.yaml')

        self.assertFalse(os.path.isfile('foo.txt'))
        self.assertFalse(os.path.isfile('bar.txt'))

        os.remove('shelephant_dump.yaml')


class Test_parse(unittest.TestCase):

    def test_basic(self):

        with open('foo.txt', 'w') as file:
            file.write('foo')

        with open('bar.txt', 'w') as file:
            file.write('bar')

        output = run('shelephant_dump foo.txt bar.txt')
        output = run('shelephant_parse shelephant_dump.yaml')

        self.assertEqual(list(filter(None, output.split('\n'))), ['- foo.txt', '- bar.txt'])

        os.remove('shelephant_dump.yaml')
        os.remove('foo.txt')
        os.remove('bar.txt')


if __name__ == '__main__':

    unittest.main()
