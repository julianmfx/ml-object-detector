from pathlib import Path
import re
import unicodedata

regex_expression = re.compile(r"[^a-z0-9]+")

def slugify(filename: str, max_len: int = 30) -> str:
    """
    Turn arbitrary text into a filesystem-safe slug.

    - lower-cases the text
    - strips accents (café → cafe)
    - drop the extension
    - replaces any non-alphanumeric run with a single hyphen
    - collapses consecutive hyphens
    - trims to *max_len* characters
    """
    # Work on the stem only
    stem = Path(filename).stem

    # Strip accents -----------------------
    stem = (
        unicodedata.normalize("NFKD", stem)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )

    # Replace unwanted chars with "-", collapse duplicates
    slug = regex_expression.sub("-", stem)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")

    return slug[:max_len] or "query"
