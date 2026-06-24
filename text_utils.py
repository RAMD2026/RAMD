# text_utils.py
import re

def normalize_name(s: str) -> str:
    """Lower, split underscores/camelCase, strip punctuation, collapse spaces."""
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    s = s.strip().replace("_", " ").replace("-", " ")
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)  # camelCase → camel Case
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
