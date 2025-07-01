
import re
import unicodedata

def slugify(text: str, max_len: int = 30) -> str:
    """
    Turn arbitrary text into a filesystem-safe slug.

    • lower-cases the text
    • strips accents (café → cafe)
    • replaces any non-alphanumeric run with a single hyphen
    • collapses consecutive hyphens
    • trims to *max_len* characters
    """

    # Strip accents -----------------------
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Lower clase and replace wanted chars with -
    text = re.sub(r"[^a-z0-0]+", "-", text.lower())

    # Collapse runs of - and trim
    text = re.sub(r"-{2,}", "-", text).strip("-")

    return text[:max_len] or "query"
