*************
Copying files
*************

List of files
=============

Locally
-------

1.  List the files that you would like to have copied.
    For example:

    .. code-block:: bash

        cd /path/to/your/files
        shelephant_dump *.h5

    This creates a file ``shelephant_dump.yaml`` with a list of files:

    .. code-block:: yaml

        - file1.h5
        - file2.h5
        - file3.h5

    .. note::

        The filenames are relative to ``shelephant_dump.yaml``.

2.  Copy all files to some destination:

    .. code-block:: bash

        shelephant_cp shelephant_dump.yaml /path/to/destination

    This copies:

    .. code-block:: bash

        file1.h5 -> /path/to/destination/file1.h5
        file2.h5 -> /path/to/destination/file2.h5
        file3.h5 -> /path/to/destination/file3.h5

    .. note::

        A copy plan is proposed before copying files.
        Copying only proceeds if the plan is accepted.
        This plan is based on precomputed *sha256* checksums, *rsync* criteria, and/or simple file existence.
        Based on the available information and backend a default combination is used, but this can be customised.
        **Important:** files that are listed as equal are not touched in any way.

To SSH host
-----------

1.  List the files that you would like to have copied.
    For example:

    .. code-block:: bash

        cd /path/to/your/files
        shelephant_dump *.h5

    This creates a file ``shelephant_dump.yaml`` with a list of files (see above).

2.  Copy to your remote host:

    .. code-block:: bash

        shelephant_cp shelephant_dump.yaml /path/on/host --ssh user@host

    .. note::

        *rsync* is used to propose a copy plan and to copy files if that plan is accepted.
        In this case *rsync* is a mandatory dependency.

    .. tip::

        You can store the host information by:

        .. code-block:: bash

            shelephant_hostinfo /path/on/host --ssh user@host

        and then copy files by:

        .. code-block:: bash

            shelephant_cp shelephant_dump.yaml shelephant_hostinfo.yaml

From SSH host
-------------

1.  List the files that you would like to have copied *on the host*.
    For example:

    .. code-block:: bash

        ssh user@host
        cd /path/on/host
        shelephant_dump *.h5

    This creates a file ``shelephant_dump.yaml`` with a list of files (see above).

2.  Copy from the remote host *on your local machine*:

    .. code-block:: bash

        cd /path/to/copied/files
        shelephant_hostinfo /path/on/destination --ssh user@host -d
        shelephant_cp shelephant_hostinfo.yaml .

Watch directory
===============

Suppose that you have a data container in

.. code-block:: bash

    /path/to/your/files

and you would like to keep a backup of certain files (e.g. ``*.h5``) in

.. code-block:: bash

    /path/to/backup

1.  Create a file ``containerinfo.yaml`` in ``/path/to/backup`` with the following content:

    .. code-block:: yaml

        root: /path/to/your/files  # may be relative
        search:
            - rglob: '*.h5'

2.  Get updates from that source:

    .. code-block:: bash

        shelephant_hostinfo -iu containerinfo.yaml

    .. tip::

        ``-i`` computes the *sha256* checksums, which may not be needed depending on you use.

    .. note::

        Run this command (and the command below) from ``/path/to/backup``.

3.  Update the 'backup':

    .. code-block:: bash

        shelephant_cp containerinfo.yaml .

    .. note::

        This will show a copy plan and ask for confirmation.

    .. tip::

        To compare files based on their *sha256* checksums, for example create in ``/path/to/backup`` a file ``localinfo.yaml`` with the following content:

        .. code-block:: yaml

            root: "."
            search:
                - rglob: '*.h5'

        and then run:

        .. code-block:: bash

            shelephant_hostinfo -iu localinfo.yaml

        To copy now use:

        .. code-block:: bash

            shelephant_diff containerinfo.yaml localinfo.yaml

        (You can also use ``shelephant_cp`` to copy. In that case the copy-plan can be based purely on *sha256* if you set ``--mode=sha256``.)
