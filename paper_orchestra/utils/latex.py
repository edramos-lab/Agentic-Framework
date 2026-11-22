"""Small LaTeX clean-up helpers shared by the writing and refinement agents."""

from __future__ import annotations

import re
from pathlib import Path

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


_PUNCTUATION = {"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "--", "\u2014": "---", "\u2212": "-",
                "\u2018": "`", "\u2019": "'", "\u201c": "``", "\u201d": "''", "\u00a0": " ", "\u2009": " ", "\u200b": ""}


def latex_safe(text: str) -> str:
    """Make text typesettable by pdflatex: typographic punctuation to its LaTeX form, letters outside Latin-1 (which
    need another font encoding) to the plain letter, anything else (e.g. Thai) dropped."""
    import unicodedata

    out = []
    for ch in text:
        ch = _PUNCTUATION.get(ch, ch)
        if all(ord(c) < 256 for c in ch):
            out.append(ch)
            continue
        out.append("".join(c for c in unicodedata.normalize("NFKD", ch) if ord(c) < 128))
    return "".join(out)


def escape_bibtex(value: str) -> str:
    """Escape the characters that break a BibTeX text field (& % #), keeping existing escapes, and make it pdflatex-safe."""
    return re.sub(r"(?<!\\)([&%#])", r"\\\1", latex_safe(value))


def drop_unreadable_entries(bib: str) -> str:
    """Remove BibTeX entries whose title has no readable text (a title in a script pdflatex cannot print)."""
    def keep(m: re.Match) -> str:
        title = re.search(r"title\s*=\s*\{(.*?)\},?\s*\n", m.group(0), re.S)
        letters = len(re.findall(r"[A-Za-z]", title.group(1))) if title else 0
        return m.group(0) if letters >= 8 else ""

    return re.sub(r"@\w+\{[^,\s]+,.*?\n\}\n?", keep, bib, flags=re.S)


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


_GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}")


def drop_missing_figures(latex: str, available_stems: set[str]) -> tuple[str, list[str]]:
    """Remove figure environments whose image is not among available_stems (file names without extension).

    Returns the new source and the image names that were missing.
    """
    missing: list[str] = []

    def check(m: re.Match) -> str:
        for ref in _GRAPHIC.findall(m.group(0)):
            if Path(ref).stem not in available_stems:
                missing.append(ref)
                return ""
        return m.group(0)

    fig = re.compile(r"\\begin\{(figure\*?)\}.*?\\end\{\1\}", re.S)
    return fig.sub(check, latex), missing


_ENV = re.compile(r"\\(begin|end)\{([A-Za-z*]+)\}")
_DANGLING = re.compile(r"(?:~?\\(?:ref|eqref|cite[a-zA-Z]*|label|begin|end|textbf|textit|emph|caption|includegraphics)\s*(?:\[[^\]]*)?)\s*$")


def _uncommented(latex: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in latex.splitlines())


def unclosed_environments(latex: str) -> list[str]:
    """Names of \\begin{...} environments that are never closed, outermost first (a cut-off section leaves some)."""
    stack: list[str] = []
    for kind, name in _ENV.findall(_uncommented(latex)):
        if kind == "begin":
            stack.append(name)
        elif name in stack:
            while stack and stack.pop() != name:
                pass
    return stack


def looks_cut_off(latex: str) -> list[str]:
    """Why a piece of LaTeX looks cut off (empty list when it does not): unclosed environments, a dangling command."""
    problems = [f"unclosed environment: {n}" for n in unclosed_environments(latex)]
    body = _uncommented(latex).rstrip()
    if _DANGLING.search(body):
        problems.append("ends in the middle of a command")
    return problems


def close_cut_off(latex: str) -> str:
    """Last resort for a section that is still cut off: drop a dangling command and close the open environments."""
    body = latex.rstrip()
    body = _DANGLING.sub("", body).rstrip()
    closers = "".join(f"\\end{{{n}}}\n" for n in reversed(unclosed_environments(body)))
    return body + "\n" + closers


def split_author_list(name: str) -> list[str]:
    """One BibTeX author per item.  A record sometimes holds a whole list as one name ("A B, C D, E F"); BibTeX reads
    that as a single name with too many commas and stops.  One comma ("Last, First") is a normal name and is kept."""
    name = name.strip()
    if name.count(",") >= 2 and " and " not in name:
        return [part.strip() for part in name.split(",") if part.strip()]
    return [name] if name else []


def fix_bibtex_authors(bib: str) -> str:
    """Rewrite author fields whose names are separated by commas instead of "and"."""
    def fix(m: re.Match) -> str:
        value = m.group(2)
        if value.count(",") >= 2 and " and " not in value:
            value = " and ".join(split_author_list(value))
        return m.group(1) + value + m.group(3)

    return re.sub(r"^(\s*author\s*=\s*\{)(.*)(\},?)$", fix, bib, flags=re.M)


_REF = re.compile(r"\\(ref|eqref|autoref|cref)\{([^}]+)\}")
_LABEL = re.compile(r"\\label\{([^}]+)\}")


def _by_caption(latex: str, label: str, kind: str, defined: set[str]) -> str | None:
    """The label of the float whose caption shares the most words with the sentence that cites `label` (None if unclear)."""
    float_re = re.compile(r"\\begin\{(figure\*?|table\*?)\}(.*?)\\end\{\1\}", re.S)
    captions = {}
    for m in float_re.finditer(latex):
        cap, lab = re.search(r"\\caption\{(.*)\}", m.group(2)), _LABEL.search(m.group(2))
        if cap and lab and lab.group(1) in defined and lab.group(1).startswith(kind):
            captions[lab.group(1)] = cap.group(1)
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", latex) if "{" + label + "}" in x]
    if not captions or not sentences:
        return None
    significant = lambda text: {w for w in re.findall(r"[a-z]{5,}", text.lower())} - {"table", "figure", "reports", "shows", "using"}
    context = significant(sentences[0])
    scored = sorted(((len(context & significant(c)), lab) for lab, c in captions.items()), reverse=True)
    return scored[0][1] if scored[0][0] > 0 and (len(scored) == 1 or scored[1][0] < scored[0][0]) else None


def _label_heading(latex: str, label: str) -> str | None:
    """Add \\label{label} after the one \\section/\\subsection whose title shares word stems with it (sec:methods -> Methodology)."""
    stems = lambda text: {w[:4] for w in re.findall(r"[a-z]{4,}", text.lower())}
    want = stems(label.split(":", 1)[-1].replace("_", " "))
    heads = [(len(want & stems(m.group(2))), m) for m in re.finditer(r"\\(section|subsection)\*?\{([^}]*)\}", latex)]
    heads = sorted(((n, m) for n, m in heads if n > 0), key=lambda x: -x[0])
    if not heads or (len(heads) > 1 and heads[1][0] == heads[0][0]):
        return None
    m = heads[0][1]
    return latex[:m.end()] + f"\\label{{{label}}}" + latex[m.end():]


def fix_undefined_refs(latex: str) -> tuple[str, dict[str, str], list[str]]:
    """Point a \\ref to a label that does not exist at the defined label of the same kind (tab:, fig:, sec:) that shares
    the most words with it (tab:probe_results -> tab:probe), when exactly one does.  Returns the new source, the labels it
    remapped, and the ones it could not resolve."""
    defined = set(_LABEL.findall(latex))
    words = lambda label: set(re.split(r"[_:\-]+", label.lower())) - {"tab", "fig", "sec", "eq", "results", "result"}
    remap: dict[str, str] = {}
    unresolved: list[str] = []
    for label in dict.fromkeys(m.group(2) for m in _REF.finditer(latex)):
        if label in defined:
            continue
        kind = label.split(":", 1)[0] + ":" if ":" in label else ""
        scored = sorted(((len(words(label) & words(d)), d) for d in defined if d.startswith(kind)), reverse=True)
        if scored and scored[0][0] > 0 and (len(scored) == 1 or scored[1][0] < scored[0][0]):
            remap[label] = scored[0][1]
            continue
        guess = _by_caption(latex, label, kind, defined)
        if guess:
            remap[label] = guess
        else:
            labelled = _label_heading(latex, label) if kind == "sec:" else None
            if labelled is None:
                unresolved.append(label)
            else:
                latex = labelled  # the heading now carries the label
                defined.add(label)
    if remap:
        latex = _REF.sub(lambda m: f"\\{m.group(1)}{{{remap.get(m.group(2), m.group(2))}}}", latex)
    return latex, remap, unresolved
