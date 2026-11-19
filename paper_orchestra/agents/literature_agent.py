"""Step 3 — Literature Review Agent.

Searches Semantic Scholar (via the public API) and optionally the web,
then uses the LLM to synthesize introduction and related-work paragraphs
along with a .bib file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from ..base_agent import BaseAgent
from ..models import BibEntry, LiteratureReviewOutput, PipelineState, ResearchNode

logger = logging.getLogger(__name__)

S2_API = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,authors,year,venue,abstract,externalIds,citationCount"
OPENALEX_API = "https://api.openalex.org/works"
OPENALEX_FIELDS = "id,doi,title,publication_year,authorships,primary_location,cited_by_count,abstract_inverted_index,is_retracted"
SCITE_API = "https://api.scite.ai"


class LiteratureAgent(BaseAgent):
    name = "LiteratureAgent"

    def execute(self, state: PipelineState) -> PipelineState:
        if not state.outline:
            self.logger.warning("No outline — skipping literature review.")
            return state

        # 1. Collect papers: OpenAlex first (no key, generous limits), Semantic Scholar as backup
        sources = self.config.get("sources", ["openalex", "semantic_scholar"])
        all_papers: list[dict] = []
        for node in state.outline.research_graph:
            query = node.query or node.label
            found: list[dict] = []
            for src in sources:
                found += self._search_openalex(query, 5) if src == "openalex" else self._search_s2(query, 5)
            all_papers.extend(found)
            self.logger.info("  Node '%s': found %d papers", node.label, len(found))

        unique = self._dedupe(all_papers)
        self.logger.info("Total unique papers: %d", len(unique))

        # 1b. Check every DOI with scite: drop retracted papers, keep support/contradiction counts
        report: dict = {"dropped": [], "unverified": [], "checked": 0}
        if self.config.get("verify_with_scite", True):
            unique = self._verify_with_scite(unique, report)
        self._write("literature_verification.json", json.dumps(report, indent=2, ensure_ascii=False))

        # 2. Build bib entries with unique, LaTeX-safe keys
        bib_entries = self._assign_keys([self._paper_to_bib(p) for p in unique])

        # 3. Write .bib file
        bib_text = "\n\n".join(e.bibtex for e in bib_entries if e.bibtex)
        bib_path = self._write("references.bib", bib_text)

        # 4. Use LLM to synthesize intro & related work
        intro, related = self._synthesize(state, bib_entries)

        state.literature = LiteratureReviewOutput(
            intro_paragraphs=intro,
            related_work_paragraphs=related,
            bib_entries=bib_entries,
            bib_file_path=bib_path,
        )

        # Persist the generated sections
        self._write("intro_draft.md", "\n\n".join(intro))
        self._write("related_work_draft.md", "\n\n".join(related))

        return state

    # ----- Semantic Scholar --------------------------------------------------

    def _search_s2(self, query: str, limit: int = 5) -> list[dict]:
        """Search Semantic Scholar for papers matching *query*."""
        headers = {"x-api-key": os.environ["S2_API_KEY"]} if os.environ.get("S2_API_KEY") else {}
        for attempt in range(4):
            try:
                resp = httpx.get(
                    f"{S2_API}/paper/search",
                    params={"query": query, "limit": limit, "fields": S2_FIELDS},
                    headers=headers,
                    timeout=30,
                )
                if resp.status_code == 429 and attempt < 3:
                    time.sleep(2 ** attempt * 2)  # the public API rate-limits anonymous callers
                    continue
                resp.raise_for_status()
                return resp.json().get("data", [])
            except Exception as exc:
                self.logger.warning("S2 search failed for '%s': %s", query, exc)
                return []
        return []

    # ----- OpenAlex ----------------------------------------------------------

    def _search_openalex(self, query: str, limit: int = 5) -> list[dict]:
        """Search OpenAlex and return papers in the same shape as Semantic Scholar's."""
        params = {"search": query, "per-page": limit, "select": OPENALEX_FIELDS}
        mailto = os.environ.get("OPENALEX_MAILTO") or self.config.get("openalex_mailto")
        if mailto:
            params["mailto"] = mailto  # puts the request in OpenAlex's faster "polite pool"
        try:
            resp = httpx.get(OPENALEX_API, params=params, timeout=30)
            resp.raise_for_status()
            works = resp.json().get("results", [])
        except Exception as exc:
            self.logger.warning("OpenAlex search failed for '%s': %s", query, exc)
            return []

        papers = []
        for w in works:
            doi = (w.get("doi") or "").removeprefix("https://doi.org/")
            source = ((w.get("primary_location") or {}).get("source") or {}).get("display_name", "")
            papers.append({
                "paperId": w.get("id", ""),
                "title": w.get("title") or "Untitled",
                "authors": [{"name": a["author"]["display_name"]} for a in w.get("authorships", [])
                            if a.get("author", {}).get("display_name")],
                "year": w.get("publication_year"),
                "venue": source or "",
                "abstract": self._abstract_from_index(w.get("abstract_inverted_index")),
                "externalIds": {"DOI": doi} if doi else {},
                "citationCount": w.get("cited_by_count", 0),
                "retracted": bool(w.get("is_retracted")),
                "_source": "openalex",
            })
        return papers

    @staticmethod
    def _abstract_from_index(index: dict | None) -> str:
        """OpenAlex ships abstracts as {word: [positions]}; put them back in order."""
        if not index:
            return ""
        words: dict[int, str] = {}
        for word, positions in index.items():
            for pos in positions:
                words[pos] = word
        return " ".join(words[i] for i in sorted(words))

    # ----- de-duplication, scite check, keys ---------------------------------

    @staticmethod
    def _dedupe(papers: list[dict]) -> list[dict]:
        """Merge duplicates found by several sources or queries (same DOI, else same title)."""
        merged: dict[str, dict] = {}
        for p in papers:
            doi = ((p.get("externalIds") or {}).get("DOI") or "").lower()
            key = doi or re.sub(r"[^a-z0-9]", "", (p.get("title") or "").lower())
            if not key:
                continue
            if key not in merged:
                merged[key] = p
            else:  # keep the first record but fill gaps from the duplicate
                kept = merged[key]
                for field in ("abstract", "venue", "year"):
                    if not kept.get(field) and p.get(field):
                        kept[field] = p[field]
                kept["retracted"] = kept.get("retracted", False) or p.get("retracted", False)
        return list(merged.values())

    def _verify_with_scite(self, papers: list[dict], report: dict) -> list[dict]:
        """Drop retracted papers and attach scite's support/contradiction counts."""

        def check(paper: dict) -> dict:
            doi = (paper.get("externalIds") or {}).get("DOI", "")
            if not doi:
                return {"doi": "", "status": "unverified", "reason": "no DOI"}
            try:
                info = httpx.get(f"{SCITE_API}/papers/{doi}", timeout=20)
                tally = httpx.get(f"{SCITE_API}/tallies/{doi}", timeout=20)
                if info.status_code != 200:
                    return {"doi": doi, "status": "unverified", "reason": f"scite HTTP {info.status_code}"}
                body = info.json()
                counts = tally.json() if tally.status_code == 200 else {}
                notices = [n.get("status") for n in body.get("editorialNotices", [])]
                return {"doi": doi, "status": "ok", "retracted": bool(body.get("retracted")),
                        "notices": notices,
                        "scite": {k: counts.get(k, 0) for k in ("supporting", "contradicting", "mentioning")}}
            except Exception as exc:
                return {"doi": doi, "status": "unverified", "reason": str(exc)[:120]}

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(check, papers))

        kept = []
        for paper, res in zip(papers, results):
            report["checked"] += 1
            if res["status"] != "ok":
                report["unverified"].append({"title": paper.get("title"), **res})
            # Either source can flag a retraction, and a flag from OpenAlex counts even if scite had no record
            if paper.get("retracted") or res.get("retracted"):
                report["dropped"].append({"title": paper.get("title"), "doi": res.get("doi", ""),
                                          "notices": res.get("notices", ["openalex"])})
                self.logger.warning("  Dropped retracted paper: %s", paper.get("title"))
                continue
            if res["status"] == "ok":
                paper["scite"] = res["scite"]
            kept.append(paper)
        self.logger.info("  scite: %d checked, %d dropped, %d unverified",
                         report["checked"], len(report["dropped"]), len(report["unverified"]))
        return kept

    @staticmethod
    def _assign_keys(entries: list[BibEntry]) -> list[BibEntry]:
        """Give every entry a unique, LaTeX-safe key (smith2023, smith2023a, ...)."""
        used: dict[str, int] = {}
        for e in entries:
            base = unicodedata.normalize("NFKD", e.cite_key).encode("ascii", "ignore").decode()
            base = re.sub(r"[^a-z0-9]", "", base.lower()) or "paper"
            n = used.get(base, 0)
            used[base] = n + 1
            key = base if n == 0 else f"{base}{chr(ord('a') + n - 1)}"
            e.bibtex = e.bibtex.replace("{" + e.cite_key + ",", "{" + key + ",", 1)
            e.cite_key = key
        return entries

    @staticmethod
    def _paper_to_bib(paper: dict) -> BibEntry:
        """Convert an S2 paper dict to our BibEntry model + BibTeX string."""
        authors = [a.get("name", "") for a in paper.get("authors", [])]
        year = paper.get("year")
        title = paper.get("title", "Untitled")
        venue = paper.get("venue", "")
        doi = (paper.get("externalIds") or {}).get("DOI", "")
        abstract = paper.get("abstract", "") or ""

        # Generate a cite key: first-author-last-name + year
        first_last = authors[0].split()[-1] if authors else "unknown"
        cite_key = f"{first_last.lower()}{year or ''}"

        bibtex = textwrap.dedent(f"""\
            @article{{{cite_key},
              title  = {{{title}}},
              author = {{{' and '.join(authors)}}},
              year   = {{{year or ''}}},
              journal = {{{venue}}},
              doi    = {{{doi}}},
            }}""")

        return BibEntry(
            cite_key=cite_key,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            abstract=abstract[:500],
            bibtex=bibtex,
            source=paper.get("_source", "semantic_scholar"),
            scite=paper.get("scite", {}),
        )

    # ----- LLM synthesis -----------------------------------------------------

    def _synthesize(
        self, state: PipelineState, bib_entries: list[BibEntry]
    ) -> tuple[list[str], list[str]]:
        """Use the LLM to draft introduction and related-work sections."""
        idea = state.inputs.idea
        papers_summary = "\n".join(
            f"- [{e.cite_key}] {e.title} ({e.year}): {e.abstract[:200]}"
            + (f" [scite: {e.scite.get('supporting', 0)} supporting, "
               f"{e.scite.get('contradicting', 0)} contradicting citations]" if e.scite else "")
            for e in bib_entries[:30]
        )

        system = (
            "You are an academic writing assistant.  Given a research idea and "
            "a list of related papers, write:\n"
            "1. An Introduction section (3-4 paragraphs) that motivates the work "
            "and positions it in the literature.  Cite papers using \\cite{key}.\n"
            "2. A Related Work section (3-5 paragraphs) that surveys the most "
            "relevant prior work grouped thematically.  Cite appropriately.\n\n"
            "Return JSON with keys 'intro' (list of paragraph strings) and "
            "'related_work' (list of paragraph strings).  Return ONLY JSON.\n"
            "Cite ONLY keys from the Available Papers list; never invent a citation.  "
            "Where a paper has contradicting citations, say the finding is disputed "
            "instead of presenting it as settled."
        )
        user = (
            f"# Research Idea\nTitle: {idea.title}\n"
            f"Abstract: {idea.abstract_draft}\n"
            f"Research questions: {', '.join(idea.research_questions)}\n\n"
            f"# Available Papers\n{papers_summary}"
        )

        try:
            parsed = self._prompt_json(system, user, max_tokens=6000)
            intro = parsed.get("intro", [])
            related = parsed.get("related_work", [])
        except Exception as exc:
            self.logger.warning("LLM synthesis failed: %s — using empty draft", exc)
            intro, related = [], []

        return intro, related
