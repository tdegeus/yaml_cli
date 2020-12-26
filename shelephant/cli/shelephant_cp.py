'''shelephant_cp
    Copy files listed in a (field of a) YAML-file.
    The filenames are assumed either absolute, or relative to the input YAML-file.

Usage:
    shelephant_cp [options] <destination>
    shelephant_cp [options] <input.yaml> <destination>

Argument:
    <input.yaml>    YAML-file with filenames. Default: shelephant_dump.yaml
    <destination>   Prefix of the destination.

Options:
    -c, --checksum  Use checksum to skip files that are the same.
    -k, --key=N     Path in the YAML-file, separated by "/". [default: /]
        --colors=M  Select color scheme from: none, dark. [default: dark]
    -q, --quiet     Do not print progress.
    -f, --force     Move without prompt.
    -h, --help      Show help.
        --version   Show version.

(c - MIT) T.W.J. de Geus | tom@geus.me | www.geus.me | github.com/tdegeus/shelephant
'''

import docopt
import shutil

from .. import __version__
from . import ShelephantCopy


def main():

    args = docopt.docopt(__doc__, version=__version__)

    return ShelephantCopy(
        copy_function = shutil.copy,
        yaml_src = args['<input.yaml>'] if args['<input.yaml>'] else 'shelephant_dump.yaml',
        yaml_key = list(filter(None, args['--key'].split('/'))),
        dest_dir = args['<destination>'],
        checksum = args['--checksum'],
        quiet = args['--quiet'],
        force = args['--force'],
        theme_name = args['--colors'].lower())


if __name__ == '__main__':

    main()