"""Small LaTeX clean-up helpers shared by the writing and refinement agents."""

from __future__ import annotations

import re

_CITE = re.compile(r"[ \t]*\\(cite[a-zA-Z]*)\s*(\[[^\]]*\])?\s*\{([^}]*)\}")


def strip_fences(text: str) -> str:
    """Remove markdown code-fence lines an LLM may wrap around LaTeX."""
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("```")).strip() + "\n"


def restrict_citations(latex: str, valid_keys: set[str]) -> tuple[str, list[str]]:
    """Drop every cited key that is not in valid_keys.

    A \\cite left with no valid key is removed.  Returns the new source and the
    keys that were removed, so callers can log them.
    """
    removed: list[str] = []

    def fix(m: re.Match) -> str:
        keys = [k.strip() for k in m.group(3).split(",") if k.strip()]
        kept = [k for k in keys if k in valid_keys]
        removed.extend(k for k in keys if k not in valid_keys)
        if not kept:
            return ""
        return f"{m.group(0)[: m.start(1) - m.start(0)]}{m.group(1)}{m.group(2) or ''}{{{','.join(kept)}}}"

    return _CITE.sub(fix, latex), removed


def disable_unused_bibliography(latex: str) -> str:
    """Comment out \\bibliography when nothing is cited; an empty list breaks the build."""
    if re.search(r"\\cite[a-zA-Z]*\s*(\[[^\]]*\])?\s*\{", latex):
        return latex
    return re.sub(r"^(\\bibliography(?:style)?\{)", r"% \1", latex, flags=re.M)


def escape_bibtex(value: str) -> str:
    """Escape the characters that break a BibTeX text field (& % #), keeping existing escapes."""
    return re.sub(r"(?<!\\)([&%#])", r"\\\1", value)


_FLOAT = re.compile(r"\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}", re.S)


def dedupe_floats(latex: str) -> tuple[str, int]:
    """Drop figure/table environments whose \\label already appeared in an earlier one.

    Returns the new source and how many floats were removed.
    """
    seen: set[str] = set()
    removed = 0

    def check(m: re.Match) -> str:
        nonlocal removed
        labels = re.findall(r"\\label\{([^}]*)\}", m.group(0))
        if labels and all(l in seen for l in labels):
            removed += 1
            return ""
        seen.update(labels)
        return m.group(0)

    return _FLOAT.sub(check, latex), removed
