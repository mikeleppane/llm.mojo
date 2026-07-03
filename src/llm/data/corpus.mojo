# Reading a raw text corpus from disk.
#
# The dataset pipeline starts from a plain-text file (tiny Shakespeare here). The
# only subtlety worth a module is the failure message: when the file is absent
# the error must tell the reader how to produce it, not just that a path was
# missing — a stack trace that ends in "No such file" wastes the reader's time
# when the fix ("run the download script") is known in advance.


def load_text(path: String) raises -> String:
    # Read the whole file at `path` and return its contents as a String.
    # Allocates the full file in memory (the corpus is ~1 MB, so this is fine).
    # Raises if the file cannot be read, with a message that names the download
    # script — the actionable fix — and preserves the underlying error.
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
