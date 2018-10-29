import os
from collections import OrderedDict
from pwd import getpwuid
import datetime


def get_file_info(fpath, raw=False):
    """
    Get OS-level info about a file path.

    Args:
        fpath (PathLike): a path to a file.
        raw (bool): if True returns raw counter timestamps, otherwise
            returns datetime wrapper objects.

    Returns:
        Dict: containing
            created: Time of creation (varies by OS, see os.stat docs)
            filesize: Size in bytes of the file
            last_accessed: Time of last access.
            last_modified: Time of last modification.

    Example:
        >>> import ubelt as ub
        >>> fpath = ub.__file__
        >>> info = get_file_info(fpath)
        >>> print(ub.repr2(info, nl=1))  # xdoc: +IGNORE_WANT
        {
            'created': datetime.datetime(2018, 10, 25, 12, 30, 36, 806656),
            'filesize': 5868,
            'last_accessed': datetime.datetime(2018, 10, 25, 12, 30, 38, 886705),
            'last_modified': datetime.datetime(2018, 10, 25, 12, 30, 36, 806656),
            'owner': 'joncrall',
        }
    """
    statbuf = os.stat(fpath)

    owner = getpwuid(statbuf.st_uid).pw_name

    info = OrderedDict([
        ('created', statbuf.st_ctime),
        ('filesize', statbuf.st_size),
        ('last_accessed', statbuf.st_atime),
        ('last_modified', statbuf.st_mtime),
        ('owner', owner)
    ])
    # permission = [os.access(fpath, os.W_OK), os.access(fpath, os.X_OK)]

    if not raw:
        time_keys = [
            'last_accessed',
            'last_modified',
            'created',
        ]
        for key in time_keys:
            info[key] = datetime.datetime.fromtimestamp(info[key])
    return info
