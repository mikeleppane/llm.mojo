"""Reading a raw text corpus from disk.

The dataset pipeline starts from a plain-text file (tiny Shakespeare here). When
the file is absent, the error names the download script — the actionable fix —
rather than just reporting a missing path.
"""


def load_text(path: String) raises -> String:
    """Read the whole file at `path` and return its contents as a String.

    Args:
        path: File to read.

    Returns:
        The file contents. Allocates the full file in memory (the corpus is
        ~1 MB, so this is fine).

    Raises:
        Error: If the file cannot be read; the message names the download
            script and preserves the underlying error.
    """
    try:
        return open(path, "r").read()
    except e:
        raise Error(
            "load_text: could not read '"
            + path
            + "'. If this is the corpus, fetch it with `pixi run python"
            + " scripts/download_tinyshakespeare.py`. Underlying error: "
            + String(e)
        )
