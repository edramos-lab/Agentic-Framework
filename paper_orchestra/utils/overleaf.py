"""Package a finished run as a zip that Overleaf can compile."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

from .latex import (
    dedupe_floats,
    drop_missing_figures,
    disable_unused_bibliography,
    escape_bibtex,
    restrict_citations,
    strip_fences,
)

_BIB_KEY = re.compile(r"^@\w+\{([^,\s]+),", re.M)
_BIB_TEXT = re.compile(r"^(\s*(?:title|author|journal)\s*=\s*\{)(.*)(\},?)$", re.M)
_INCLUDE = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]*)(\})")
_FIGURE_EXT = (".pdf", ".png", ".jpg", ".jpeg")


class NoManuscript(Exception):
    """The run has no manuscript to package yet."""


def _first(out_dir: Path, *names: str) -> Path | None:
    return next((p for n in names if (p := out_dir / n).is_file()), None)


def build_overleaf_zip(out_dir: Path) -> bytes:
    """Return the bytes of a zip holding main.tex, references.bib and figures/."""
    tex_path = _first(out_dir, "refinementagent/manuscript_final.tex", "writingagent/manuscript.tex")
    if tex_path is None:
        raise NoManuscript("this run has no manuscript yet")
    tex = strip_fences(tex_path.read_text(encoding="utf-8"))

    bib_path = _first(out_dir, "literatureagent/references.bib", "writingagent/references.bib")
    bib = bib_path.read_text(encoding="utf-8") if bib_path else ""
    bib = _BIB_TEXT.sub(lambda m: m.group(1) + escape_bibtex(m.group(2)) + m.group(3), bib)

    # Figures the plotting agent produced, by file stem
    figures = {p.stem: p for p in sorted((out_dir / "plottingagent").glob("*"))
               if p.suffix.lower() in _FIGURE_EXT} if (out_dir / "plottingagent").is_dir() else {}
    used: dict[str, Path] = {}

    def point_at_figure(m: re.Match) -> str:
        ref = Path(m.group(2))
        src = figures.get(ref.stem)
        if src is None:
            return m.group(0)
        used[src.name] = src
        return f"{m.group(1)}figures/{src.name}{m.group(3)}"

    tex = _INCLUDE.sub(point_at_figure, tex)
    tex, _ = restrict_citations(tex, set(_BIB_KEY.findall(bib)))
    tex, _ = drop_missing_figures(tex, {Path(n).stem for n in used})
    tex, _ = dedupe_floats(tex)
    if "\\end{document}" not in tex:
        tex += "\n\\bibliographystyle{IEEEtran}\n\\bibliography{references}\n\n\\end{document}\n"
    tex = disable_unused_bibliography(tex)
    tex = re.sub(r"\n{3,}", "\n\n", tex)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("main.tex", tex)
        z.writestr("references.bib", bib)
        for name, path in used.items():
            z.write(path, f"figures/{name}")
    return buf.getvalue()
