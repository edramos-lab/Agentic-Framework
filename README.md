# PaperOrchestra 🎼

**A multi-agent framework for end-to-end research paper generation**

PaperOrchestra coordinates five specialized AI agents through a pipeline that transforms a research idea into a submission-ready PDF — including running experiments, reviewing literature, generating plots, and iteratively refining the manuscript.

## Architecture

```
INPUTS                  STEP 1          STEPS 2+3+NEW (parallel)      STEP 4           STEP 5           OUTPUT
┌──────────────┐    ┌───────────┐    ┌─────────────────────────┐  ┌────────────┐  ┌──────────────┐  ┌──────────┐
│ Idea (I)     │    │  Outline  │    │ Plotting Agent          │  │  Section   │  │  AgentReview │  │Submission│
│ Template (T) │───▶│  Agent    │───▶│ Experiment Runs Agent   │─▶│  Writing   │─▶│  + Content   │─▶│  -ready  │
│ Guideline(G) │    │           │    │ Literature Review Agent  │  │  Agent     │  │  Refinement  │  │   PDF    │
│ Exp. Reqs    │    └───────────┘    └─────────────────────────┘  └────────────┘  └──────────────┘  └──────────┘
└──────────────┘     JSON outline     Plots, .bib, exp logs        Full LaTeX     Scored revisions
```

## Quick Start

```bash
# Install
pip install -e .

# Configure (copy and edit the example)
cp examples/pipeline_config.yaml my_config.yaml
# Edit my_config.yaml — set your LLM API key, idea, etc.

# Run
paper-orchestra run --config my_config.yaml --output ./my_paper
```

## Pipeline Steps

| Step | Agent | Input | Output |
|------|-------|-------|--------|
| 1 | **OutlineAgent** | Idea + Template + Guideline + Exp. Reqs | JSON outline (viz plan, research graph, writing plan, experiment plan) |
| 2 | **PlottingAgent** | Visualization plan | PDF/PNG figures |
| NEW | **ExperimentAgent** | Experiment plan + scripts | Run logs, metrics, documentation |
| 3 | **LiteratureAgent** | Research graph | Intro + Related Work drafts, .bib |
| 4 | **WritingAgent** | Outline + plots + literature + experiments | Full LaTeX manuscript |
| 5 | **ReviewAgent** + **RefinementAgent** | Manuscript | Iteratively refined PDF |

Steps 2, NEW, and 3 run **in parallel** for efficiency.

## Configuration

See `examples/pipeline_config.yaml` for a fully commented example. Key sections:

- `llm` — Provider, model, API key, temperature
- `idea` — Your research idea (title, abstract, questions, contributions)
- `guideline` — Conference/journal formatting rules
- `experiment_requirements` — Datasets, models, metrics, baselines, scripts
- `agents` — Per-agent overrides

## Supported LLM Backends

- **OpenAI** — GPT-4o, GPT-4o-mini, o1, etc.
- **Anthropic** — Claude Sonnet, Opus, Haiku
- **Local** — Any OpenAI-compatible endpoint (vLLM, Ollama, etc.)

## Project Structure

```
paper_orchestra/
├── __init__.py
├── cli.py                 # CLI entry point
├── orchestrator.py        # Main pipeline coordinator
├── models.py              # Pydantic data models
├── llm.py                 # Unified LLM client
├── base_agent.py          # Abstract base agent class
├── agents/
│   ├── outline_agent.py   # Step 1: JSON outline
│   ├── plotting_agent.py  # Step 2: Figures & diagrams
│   ├── experiment_agent.py# NEW: Run experiments
│   ├── literature_agent.py# Step 3: Lit review + .bib
│   ├── writing_agent.py   # Step 4: Section writing
│   ├── review_agent.py    # Step 5a: Manuscript scoring
│   └── refinement_agent.py# Step 5b: Iterative refinement
└── utils/
    └── logging_setup.py
```

## License

MIT
