"""Step 3 — Literature Review Agent.

Searches Semantic Scholar (via the public API) and optionally the web,
then uses the LLM to synthesize introduction and related-work paragraphs
along with a .bib file.
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path

import httpx

from ..base_agent import BaseAgent
from ..models import BibEntry, LiteratureReviewOutput, PipelineState, ResearchNode

logger = logging.getLogger(__name__)

S2_API = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,authors,year,venue,abstract,externalIds,citationCount"


class LiteratureAgent(BaseAgent):
    name = "LiteratureAgent"

    def execute(self, state: PipelineState) -> PipelineState:
        if not state.outline:
            self.logger.warning("No outline — skipping literature review.")
            return state

        # 1. Collect papers from Semantic Scholar for each research node
        all_papers: list[dict] = []
        for node in state.outline.research_graph:
            papers = self._search_s2(node.query or node.label, limit=5)
            all_papers.extend(papers)
            self.logger.info("  Node '%s': found %d papers", node.label, len(papers))

        # Deduplicate by paperId
        seen = set()
        unique: list[dict] = []
        for p in all_papers:
            pid = p.get("paperId", "")
            if pid and pid not in seen:
                seen.add(pid)
                unique.append(p)

        self.logger.info("Total unique papers: %d", len(unique))

        # 2. Build bib entries
        bib_entries = [self._paper_to_bib(p) for p in unique]

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
        try:
            resp = httpx.get(
                f"{S2_API}/paper/search",
                params={"query": query, "limit": limit, "fields": S2_FIELDS},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as exc:
            self.logger.warning("S2 search failed for '%s': %s", query, exc)
            return []

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
        )

    # ----- LLM synthesis -----------------------------------------------------

    def _synthesize(
        self, state: PipelineState, bib_entries: list[BibEntry]
    ) -> tuple[list[str], list[str]]:
        """Use the LLM to draft introduction and related-work sections."""
        idea = state.inputs.idea
        papers_summary = "\n".join(
            f"- [{e.cite_key}] {e.title} ({e.year}): {e.abstract[:200]}"
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
            "'related_work' (list of paragraph strings).  Return ONLY JSON."
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
