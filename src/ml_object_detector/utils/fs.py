from pathlib import Path
import os
import logging

log = logging.getLogger(__name__)


def ensure_file_exists(
    path: Path,
    *,  # everything after must be passed by name
    mode: int = 0o644, # POSIX permission bits for new files (default 0644, i.e. rw-r--r--)
    create_parents: bool = True,
) -> None:
    """
    Guarantee that *path* exists and is a **regular file**.

    Parameters
    ----------
    path : pathlib.Path
        Desired file path.
    mode : int, optional
        POSIX permissions for a newly created file (default 0644).
    create_parents : bool, optional
        Whether to create parent directories if they do not exist.
    """

    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fd = os.open(
            path,
            os.O_CREAT, # create the file if it's not there
            os.O_EXCL, # if it is there, fail with FileExistsError
            os.O_WRONLY, # open with write-only
            mode)
        os.close(fd)
        log.debug("Created file: %s", path)

    except FileExistsError:
        # Make sure the it is a file
        if not path.is_file():
            raise IsADirectoryError(f"{path} exists but is not a regular file.")
        # The file is present
        log.debug("File already exists: %s", path)

    except PermissionError as e:
        log.error("No permission to create %s: %s:", path, e)
        raise

    except OSError as e:
        # Cover I/O errors, disk full ...
        log.error("Failed to ensure file %s: %s", path, e)
        raise


def ensure_directory_exists(path: Path, mode: int = 0o755) -> None:
    """
    Guarantee that *path* exists and is a directory.

    Parameters
    ----------
    path : pathlib.Path
        Directory path to create if it doesn't exist.
    mode : int, optional
        POSIX permissions to use when creating a new directory.
    """
    try:
        # One atomic call; safe if it already exists
        path.mkdir(parents=True, exist_ok=True, mode=mode)

        # If something *else* (file, symlink) sits there, raise
        if not path.is_dir():
            raise NotADirectoryError(f"{path} exists but is not a directory")

        log.debug("Directory ready: %s", path)

    except PermissionError as e:  # common in Docker/K8s, read-only volumes, etc.
        log.error("No permission to create %s: %s", path, e)
        raise
    except OSError as e:  # catch-all for other filesystem errors
        log.error("Failed to ensure directory %s: %s", path, e)
        raise
