"""CLI entry point for PaperOrchestra.

Usage:
    paper-orchestra run --config pipeline.yaml
    paper-orchestra run --idea idea.yaml --template template.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .llm import LLMClient
from .models import (
    ConferenceGuideline,
    ExperimentRequirements,
    IdeaSummary,
    PipelineInputs,
)
from .orchestrator import PaperOrchestra
from .utils.logging_setup import setup_logging


def load_config(path: Path) -> dict:
    """Load a YAML or JSON configuration file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text) or {}
    return json.loads(text)


def build_inputs_from_config(cfg: dict) -> PipelineInputs:
    """Construct PipelineInputs from a flat config dict."""
    idea = IdeaSummary(**cfg.get("idea", {}))
    guideline = ConferenceGuideline(**cfg.get("guideline", {}))
    exp_cfg = cfg.get("experiment_requirements")
    exp = ExperimentRequirements(**exp_cfg) if exp_cfg else None

    return PipelineInputs(
        idea=idea,
        latex_template_path=Path(cfg.get("latex_template_path", "template.tex")),
        guideline=guideline,
        experiment_requirements=exp,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="paper-orchestra",
        description="Multi-agent research paper writing pipeline",
    )
    sub = parser.add_subparsers(dest="command")

    # --- run ---
    run_p = sub.add_parser("run", help="Run the full pipeline")
    run_p.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to pipeline YAML/JSON config",
    )
    run_p.add_argument("--output", "-o", type=Path, default=Path("./output"))
    run_p.add_argument("--log-level", default="INFO")

    args = parser.parse_args(argv)

    if args.command != "run":
        parser.print_help()
        sys.exit(1)

    setup_logging(args.log_level)

    cfg = load_config(args.config)
    inputs = build_inputs_from_config(cfg)

    # Build LLM client
    llm_cfg = cfg.get("llm", {})
    llm = LLMClient(
        provider=llm_cfg.get("provider", "openai"),
        model=llm_cfg.get("model", "gpt-4o"),
        api_key=llm_cfg.get("api_key", ""),
        base_url=llm_cfg.get("base_url"),
        temperature=llm_cfg.get("temperature", 0.3),
        max_tokens=llm_cfg.get("max_tokens", 4096),
    )

    orchestra = PaperOrchestra(
        llm=llm,
        work_dir=args.output,
        config=cfg.get("agents", {}),
        max_parallel=cfg.get("max_parallel", 3),
    )

    state = orchestra.run(inputs)

    # Print summary
    print("\n═══ Pipeline Complete ═══")
    if state.draft:
        print(f"  Draft: {state.draft.output_path} ({state.draft.word_count} words)")
    if state.refinement:
        print(f"  Refinement cycles: {state.refinement.total_cycles}")
        print(f"  Stopped reason: {state.refinement.stopped_reason}")
        if state.refinement.final_pdf_path:
            print(f"  Final PDF: {state.refinement.final_pdf_path}")
    if state.experiment_output:
        ok = sum(1 for r in state.experiment_output.results if r.status == "success")
        print(f"  Experiments: {ok}/{len(state.experiment_output.results)} succeeded")


if __name__ == "__main__":
    main()
