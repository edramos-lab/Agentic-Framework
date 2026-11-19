"""NEW — Experimental Runs Agent (Google Colab CLI backend).

Runs in parallel with Steps 2 and 3.  Executes the experiments defined
in the outline's experiment_plan on **Google Colab** via the official
`google-colab-cli` package, captures logs and metrics, and produces
structured documentation.

Workflow per experiment:
  1. Provision a Colab session (`colab new --gpu <type>`)
  2. Install required pip packages (`colab install`)
  3. Upload data/config files (`colab upload`)
  4. Execute the script (`colab exec -f <script>`)
  5. Download artifacts and metrics (`colab download`)
  6. Export session log (`colab log`)
  7. Release the VM (`colab stop`) — unless keep_session is set

Dependencies between experiments are resolved with topological sort
(Kahn's algorithm) so that prerequisite runs finish first.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from ..base_agent import BaseAgent
from ..models import (
    ColabConfig,
    ExperimentPlan,
    ExperimentResult,
    ExperimentRunsOutput,
    JetsonConfig,
    PipelineState,
)


class ExperimentAgent(BaseAgent):
    name = "ExperimentAgent"

    # ----- main entry --------------------------------------------------------

    def execute(self, state: PipelineState) -> PipelineState:
        if not state.outline or not state.outline.experiment_plan:
            self.logger.info("No experiment plan — skipping.")
            return state

        exp_reqs = state.inputs.experiment_requirements
        timeout = exp_reqs.timeout_minutes * 60 if exp_reqs else 7200
        colab_cfg = exp_reqs.colab if exp_reqs else ColabConfig()
        jetson_cfg = exp_reqs.jetson if exp_reqs else JetsonConfig()
        scripts_dir = exp_reqs.scripts_dir if exp_reqs else ""
        default_target = exp_reqs.backend if exp_reqs else "colab"

        ordered = self._topological_sort(state.outline.experiment_plan)
        targets = {p.experiment_id: (p.target or default_target) for p in ordered}
        host = os.environ.get("JETSON_HOST") or jetson_cfg.host

        # Colab is for training, the Jetson for edge deployment.  Each is only
        # needed when some plan targets it.
        use_colab = "colab" in targets.values()
        if use_colab and not self._colab_cli_available():
            self.logger.error("google-colab-cli not found. Install with: pip install google-colab-cli")
            self._skip_target(ordered, targets, "colab", "google-colab-cli not installed")
            use_colab = False
        if "jetson" in targets.values() and not host:
            self.logger.error("A plan targets the Jetson but no host is set (jetson.host or JETSON_HOST).")
            self._skip_target(ordered, targets, "jetson", "no Jetson host configured")

        # Shared Colab session, provisioned lazily on the first Colab plan
        shared_session: str | None = None
        if use_colab and colab_cfg.keep_session and "colab" in targets.values():
            shared_session = self._provision_session(colab_cfg)
            if shared_session:
                self._install_packages(shared_session, colab_cfg.pip_packages)
                self._upload_files(shared_session, colab_cfg.upload_files)

        results: list[ExperimentResult] = []
        done: dict[str, ExperimentResult] = {}
        for plan in ordered:
            target = targets.get(plan.experiment_id)
            if target is None:
                continue  # its target is unavailable; already logged and reported
            self.step(plan.experiment_id, plan.name, "running", target)
            if target == "jetson":
                # Carry artifacts (trained model files) from the experiments it depends on
                carried = [a for dep in plan.depends_on if dep in done for a in done[dep].artifacts]
                result = self._run_experiment_jetson(plan, jetson_cfg, host, timeout, scripts_dir, carried)
            else:
                result = self._run_experiment(plan, timeout, scripts_dir, colab_cfg, shared_session)
                result.target = "colab"
                result.device = f"Google Colab {colab_cfg.gpu_type or colab_cfg.tpu_type}"
            results.append(result)
            done[plan.experiment_id] = result
            self.step(plan.experiment_id, plan.name, result.status, target,
                      result.error_message[:200] if result.status == "failed" else "")

        if shared_session and not colab_cfg.keep_session:
            self._stop_session(shared_session)

        state.experiment_output = ExperimentRunsOutput(
            results=results,
            summary_table_path=self._write_summary(results),
            documentation_path=self._write_documentation(results, state.outline.experiment_plan),
            latex_tables=self._write_latex_tables(results),
        )
        self.logger.info(
            "Experiments: %d succeeded, %d failed",
            sum(1 for r in results if r.status == "success"),
            sum(1 for r in results if r.status == "failed"),
        )
        return state

    def _skip_target(self, plans, targets: dict[str, str], target: str, why: str) -> None:
        """Drop every plan aimed at an unavailable target and tell the UI why."""
        for plan in plans:
            if targets.get(plan.experiment_id) == target:
                self.step(plan.experiment_id, plan.name, "skipped", target, why)
                del targets[plan.experiment_id]

    # ----- script handling ---------------------------------------------------

    @staticmethod
    def _split_script(spec: str) -> tuple[str, list[str]]:
        """'train.py --config base' -> ('train.py', ['--config', 'base'])."""
        parts = shlex.split(spec) if spec else []
        return (parts[0], parts[1:]) if parts else ("", [])

    @staticmethod
    def _resolve_script(scripts_dir: str, name: str) -> tuple[Path | None, list[str]]:
        """Find a script in scripts_dir, then the bundled example folders."""
        searched = [d for d in (scripts_dir, "examples/experiments", "experiments", ".") if d]
        for d in searched:
            candidate = Path(d) / name
            if candidate.is_file():
                return candidate, searched
        return None, searched

    # ----- Jetson (SSH) backend ----------------------------------------------

    @staticmethod
    def _ssh_base(cfg: JetsonConfig, host: str, tool: str) -> list[str]:
        """Common ssh/scp options; `tool` is 'ssh' or 'scp' (they differ on the port flag)."""
        cmd = [
            tool,
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"ConnectTimeout={cfg.connect_timeout}",
            "-P" if tool == "scp" else "-p", str(cfg.port),
        ]
        if cfg.ssh_key:
            cmd += ["-i", os.path.expanduser(cfg.ssh_key)]
        return cmd

    def _run_experiment_jetson(
        self,
        plan: ExperimentPlan,
        cfg: JetsonConfig,
        host: str,
        timeout: int,
        scripts_dir: str,
        carried: list[Path] | None = None,
    ) -> ExperimentResult:
        self.logger.info("  Running experiment on Jetson %s: %s", host, plan.name)
        result = ExperimentResult(
            experiment_id=plan.experiment_id,
            status="running",
            started_at=datetime.now(timezone.utc),
            target="jetson",
            device="NVIDIA Jetson Orin Nano",
            colab_session=f"jetson:{host}",
        )
        script_name, script_args = self._split_script(plan.script)
        script_path, searched = self._resolve_script(scripts_dir, script_name)
        if script_path is None:
            result.status = "failed"
            result.error_message = f"Script '{script_name}' not found in: {', '.join(searched)}"
            result.finished_at = datetime.now(timezone.utc)
            return result

        target = f"{cfg.user}@{host}"
        remote_dir = f"{cfg.remote_dir}/{plan.experiment_id}"
        artifact_dir = self.agent_dir / plan.experiment_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()

        try:
            self._run_cmd(self._ssh_base(cfg, host, "ssh") + [target, f"mkdir -p {shlex.quote(remote_dir)}"], timeout=30, check=True)
            self._run_cmd(
                self._ssh_base(cfg, host, "scp") + [str(script_path), f"{target}:{remote_dir}/"],
                timeout=120, check=True,
            )
            for extra in [*cfg.upload_files, *map(str, carried or [])]:
                local = Path(extra).expanduser()
                if not local.exists():
                    raise RuntimeError(f"upload file not found: {local}")
                self._run_cmd(
                    self._ssh_base(cfg, host, "scp") + [str(local), f"{target}:{remote_dir}/"],
                    timeout=600, check=True,
                )
            env = " ".join(f"{k}={shlex.quote(v)}" for k, v in cfg.env_vars.items())
            args = " ".join(shlex.quote(a) for a in script_args)
            remote_cmd = (f"cd {shlex.quote(remote_dir)} && {env} "
                          f"{shlex.quote(cfg.python)} {shlex.quote(script_path.name)} {args}")
            proc = self._run_cmd(
                self._ssh_base(cfg, host, "ssh") + [target, remote_cmd], timeout=timeout
            )
            result.duration_seconds = time.perf_counter() - t0

            log_path = self.agent_dir / f"{plan.experiment_id}.log"
            log_path.write_text(
                f"=== JETSON EXEC: {plan.script} on {target} ===\n\n"
                f"=== STDOUT ===\n{proc.stdout}\n\n=== STDERR ===\n{proc.stderr}",
                encoding="utf-8",
            )
            result.log_path = log_path

            if proc.returncode != 0:
                result.status = "failed"
                result.error_message = proc.stderr[:2000]
            else:
                result.status = "success"
                for out in plan.expected_outputs:
                    local = artifact_dir / Path(out).name
                    dl = self._run_cmd(
                        self._ssh_base(cfg, host, "scp") + [f"{target}:{remote_dir}/{out}", str(local)],
                        timeout=120,
                    )
                    if dl.returncode == 0 and local.exists():
                        result.artifacts.append(local)
                        if local.suffix == ".json":
                            try:
                                result.metrics.update(json.loads(local.read_text()))
                            except (json.JSONDecodeError, OSError):
                                pass
                result.device = result.metrics.get("device", result.device)
        except subprocess.TimeoutExpired:
            result.status = "failed"
            result.error_message = f"Timed out after {timeout}s"
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            result.status = "failed"
            result.error_message = (getattr(exc, "stderr", None) or str(exc))[:2000]

        result.finished_at = datetime.now(timezone.utc)
        return result

    # ----- Colab CLI helpers -------------------------------------------------

    @staticmethod
    def _colab_cli_available() -> bool:
        """Check whether the `colab` CLI is on PATH."""
        return shutil.which("colab") is not None

    def _run_cmd(
        self,
        cmd: list[str],
        *,
        timeout: int = 300,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Run a CLI command and return the CompletedProcess."""
        self.logger.debug("  $ %s", " ".join(cmd))
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )

    def _provision_session(self, cfg: ColabConfig) -> str | None:
        """Create a Colab session and return its name, or None on failure."""
        cmd = ["colab", "new", "-s", cfg.session_name]
        if cfg.tpu_type:
            cmd += ["--tpu", cfg.tpu_type]
        elif cfg.gpu_type:
            cmd += ["--gpu", cfg.gpu_type]
        if cfg.high_mem:
            cmd.append("--high-mem")

        self.logger.info("  Provisioning Colab session: %s (GPU=%s)", cfg.session_name, cfg.gpu_type)
        try:
            proc = self._run_cmd(cmd, timeout=120)
            if proc.returncode != 0:
                self.logger.error("  colab new failed: %s", proc.stderr.strip())
                return None
            self.logger.info("  Session '%s' ready.", cfg.session_name)
            return cfg.session_name
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            self.logger.error("  Failed to provision session: %s", exc)
            return None

    def _install_packages(self, session: str, packages: list[str]) -> None:
        """Install pip packages on the Colab session."""
        if not packages:
            return
        cmd = ["colab", "install", "-s", session] + packages
        self.logger.info("  Installing packages: %s", ", ".join(packages))
        try:
            proc = self._run_cmd(cmd, timeout=300)
            if proc.returncode != 0:
                self.logger.warning("  Package install warning: %s", proc.stderr.strip()[:500])
        except subprocess.TimeoutExpired:
            self.logger.warning("  Package install timed out.")

    def _upload_files(self, session: str, local_paths: list[str]) -> None:
        """Upload local files to the Colab session at /content/."""
        for path in local_paths:
            local = Path(path)
            if not local.exists():
                self.logger.warning("  Upload skipped — file not found: %s", path)
                continue
            remote = f"/content/{local.name}"
            cmd = ["colab", "upload", "-s", session, str(local), remote]
            self.logger.debug("  Uploading %s → %s", local, remote)
            try:
                self._run_cmd(cmd, timeout=120)
            except subprocess.TimeoutExpired:
                self.logger.warning("  Upload timed out for %s", path)

    def _download_artifacts(
        self,
        session: str,
        remote_patterns: list[str],
        dest_dir: Path,
    ) -> list[Path]:
        """Download result files from the Colab session."""
        downloaded: list[Path] = []
        for remote_path in remote_patterns:
            local = dest_dir / Path(remote_path).name
            cmd = ["colab", "download", "-s", session, remote_path, str(local)]
            self.logger.debug("  Downloading %s → %s", remote_path, local)
            try:
                proc = self._run_cmd(cmd, timeout=120)
                if proc.returncode == 0 and local.exists():
                    downloaded.append(local)
                else:
                    self.logger.warning("  Download failed for %s: %s", remote_path, proc.stderr.strip()[:200])
            except subprocess.TimeoutExpired:
                self.logger.warning("  Download timed out for %s", remote_path)
        return downloaded

    def _export_log(self, session: str, dest: Path) -> Path | None:
        """Export the session execution log as a markdown file."""
        cmd = ["colab", "log", "-s", session, "-o", str(dest)]
        try:
            proc = self._run_cmd(cmd, timeout=60)
            if proc.returncode == 0 and dest.exists():
                return dest
        except subprocess.TimeoutExpired:
            pass
        return None

    def _stop_session(self, session: str) -> None:
        """Release the Colab VM."""
        self.logger.info("  Stopping Colab session: %s", session)
        try:
            self._run_cmd(["colab", "stop", "-s", session], timeout=60)
        except subprocess.TimeoutExpired:
            self.logger.warning("  Timed out stopping session %s", session)

    # ----- experiment execution ----------------------------------------------

    def _run_experiment(
        self,
        plan: ExperimentPlan,
        timeout: int,
        scripts_dir: str,
        colab_cfg: ColabConfig,
        shared_session: str | None = None,
    ) -> ExperimentResult:
        self.logger.info("  Running experiment: %s", plan.name)

        result = ExperimentResult(
            experiment_id=plan.experiment_id,
            status="running",
            started_at=datetime.now(timezone.utc),
        )

        if not plan.script:
            result.status = "failed"
            result.error_message = "No script specified"
            result.finished_at = datetime.now(timezone.utc)
            return result

        script_name, script_args = self._split_script(plan.script)
        script_path, searched = self._resolve_script(scripts_dir, script_name)
        if script_path is None:
            result.status = "failed"
            result.error_message = f"Script '{script_name}' not found in: {', '.join(searched)}"
            result.finished_at = datetime.now(timezone.utc)
            return result

        # Artifact download directory
        artifact_dir = self.agent_dir / plan.experiment_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        session_name = shared_session
        owns_session = False  # True if we created the session for this experiment

        try:
            # --- 1. Provision (or reuse shared session) ---
            if not session_name:
                per_exp_name = f"{colab_cfg.session_name}-{plan.experiment_id}"
                session_name = self._provision_session(
                    colab_cfg.model_copy(update={"session_name": per_exp_name})
                )
                if not session_name:
                    raise RuntimeError("Failed to provision Colab session")
                owns_session = True

                # Install packages & upload on new per-experiment session
                self._install_packages(session_name, colab_cfg.pip_packages)
                self._upload_files(session_name, colab_cfg.upload_files)

            result.colab_session = session_name

            # --- 2. Upload the experiment script ---
            remote_script = f"/content/{script_path.name}"
            upload_cmd = ["colab", "upload", "-s", session_name, str(script_path), remote_script]
            self._run_cmd(upload_cmd, timeout=60)

            # --- 3. Execute via `colab exec` ---
            exec_file = script_path
            if script_args:
                # `colab exec` takes no script arguments, so run the script through a
                # small wrapper that sets sys.argv first.
                exec_file = self.agent_dir / f"{plan.experiment_id}_wrapper.py"
                exec_file.write_text(
                    "import runpy, sys\n"
                    f"sys.argv = {[script_path.name, *script_args]!r}\n"
                    f"runpy.run_path({remote_script!r}, run_name='__main__')\n",
                    encoding="utf-8",
                )
            exec_cmd = ["colab", "exec", "-s", session_name, "-f", str(exec_file)]

            # Pass env vars
            for key, val in colab_cfg.env_vars.items():
                exec_cmd += ["--env", f"{key}={val}"]

            # Pass config overrides as env vars too
            for key, val in (
                state_overrides := {
                    **colab_cfg.env_vars,
                }
            ).items():
                pass  # already added above

            self.logger.info("  Executing on Colab: %s", plan.script)
            proc = self._run_cmd(exec_cmd, timeout=timeout)
            duration = time.perf_counter() - t0

            # Save stdout/stderr log locally
            log_path = self.agent_dir / f"{plan.experiment_id}.log"
            log_content = (
                f"=== COLAB EXEC: {plan.script} ===\n"
                f"=== Session: {session_name} | GPU: {colab_cfg.gpu_type} ===\n\n"
                f"=== STDOUT ===\n{proc.stdout}\n\n"
                f"=== STDERR ===\n{proc.stderr}"
            )
            log_path.write_text(log_content, encoding="utf-8")
            result.log_path = log_path
            result.duration_seconds = duration

            if proc.returncode == 0:
                result.status = "success"

                # --- 4. Download expected outputs ---
                remote_outputs = [
                    f"/content/{out}" for out in plan.expected_outputs
                ]
                # Also add any configured download patterns
                remote_outputs += colab_cfg.download_patterns

                downloaded = self._download_artifacts(
                    session_name, remote_outputs, artifact_dir
                )
                result.artifacts = downloaded

                # Try to parse metrics from any downloaded JSON
                for art in downloaded:
                    if art.suffix == ".json":
                        try:
                            result.metrics.update(json.loads(art.read_text()))
                        except (json.JSONDecodeError, OSError):
                            pass
            else:
                result.status = "failed"
                result.error_message = proc.stderr[:2000]

            # --- 5. Export Colab session log (notebook) ---
            nb_path = self.agent_dir / f"{plan.experiment_id}_colab.md"
            exported = self._export_log(session_name, nb_path)
            if exported:
                result.colab_notebook_path = exported

        except subprocess.TimeoutExpired:
            result.status = "failed"
            result.error_message = f"Timed out after {timeout}s"
            result.duration_seconds = timeout
        except Exception as exc:
            result.status = "failed"
            result.error_message = str(exc)
        finally:
            # Release per-experiment sessions
            if owns_session and session_name:
                self._stop_session(session_name)

        result.finished_at = datetime.now(timezone.utc)
        if result.duration_seconds == 0.0:
            result.duration_seconds = time.perf_counter() - t0
        return result

    # ----- documentation -----------------------------------------------------

    def _write_summary(self, results: list[ExperimentResult]) -> Path:
        """Write a Markdown summary table of all experiment results."""
        lines = [
            "| Experiment | Status | Target | Device | Duration (s) | Key Metrics |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for r in results:
            metrics_str = ", ".join(f"{k}={v}" for k, v in list(r.metrics.items())[:5])
            lines.append(
                f"| {r.experiment_id} | {r.status} | {r.target} "
                f"| {r.device or '—'} | {r.duration_seconds:.1f} | {metrics_str} |"
            )
        return self._write("experiment_summary.md", "\n".join(lines))

    def _write_documentation(
        self, results: list[ExperimentResult], plans: list[ExperimentPlan]
    ) -> Path:
        """Write detailed run documentation."""
        plan_map = {p.experiment_id: p for p in plans}
        sections: list[str] = ["# Experiment Run Documentation (Google Colab)\n"]

        for r in results:
            plan = plan_map.get(r.experiment_id)
            sections.append(f"## {r.experiment_id}: {plan.name if plan else 'Unknown'}")
            sections.append(f"- **Status**: {r.status}")
            sections.append(f"- **Colab Session**: `{r.colab_session}`")
            sections.append(f"- **Duration**: {r.duration_seconds:.1f}s")
            if plan:
                sections.append(f"- **Script**: `{plan.script}`")
                sections.append(f"- **Description**: {plan.description}")
            if r.colab_notebook_path:
                sections.append(f"- **Session Log**: `{r.colab_notebook_path}`")
            if r.metrics:
                sections.append("- **Metrics**:")
                for k, v in r.metrics.items():
                    sections.append(f"  - {k}: {v}")
            if r.artifacts:
                sections.append("- **Artifacts**:")
                for a in r.artifacts:
                    sections.append(f"  - `{a}`")
            if r.error_message:
                sections.append(f"- **Error**: {r.error_message[:500]}")
            sections.append("")

        return self._write("experiment_documentation.md", "\n".join(sections))

    def _write_latex_tables(self, results: list[ExperimentResult]) -> dict[str, str]:
        """Build a LaTeX table from every edge benchmark result.

        Numbers come straight from the benchmark JSON so the writing agent can
        paste the table verbatim instead of retyping them.
        """
        labels = {
            "pt": "PyTorch (TorchScript)", "onnx": "ONNX Runtime FP32",
            "trt_fp32": "TensorRT FP32", "trt_fp16": "TensorRT FP16", "trt_int8": "TensorRT INT8",
        }
        tables: dict[str, str] = {}
        for r in results:
            variants = r.metrics.get("variants") if r.status == "success" else None
            if not variants:
                continue
            rows = []
            for key, v in variants.items():
                name = labels.get(key, key)
                if v.get("status") != "success":
                    rows.append(f"{name} & \\multicolumn{{5}}{{c}}{{failed}} \\\\")
                    continue
                rows.append(
                    f"{name} & {v['client_mean_ms']:.2f} $\\pm$ {v['client_std_ms']:.2f} & "
                    f"{v['client_p95_ms']:.2f} & {v['fps']:.1f} & "
                    f"{v['server_compute_mean_ms']:.2f} & {v['size_mb']:.1f} \\\\"
                )
            body = "\n".join(rows)
            tex = (
                "\\begin{tabular}{lccccc}\n\\hline\n"
                "Variant & Latency (ms) & p95 (ms) & FPS & Server compute (ms) & Size (MB) \\\\\n\\hline\n"
                f"{body}\n\\hline\n\\end{{tabular}}\n"
            )
            self._write(f"{r.experiment_id}_table.tex", tex)
            tables[r.experiment_id] = tex
        return tables

    # ----- dependency resolution ---------------------------------------------

    @staticmethod
    def _topological_sort(plans: list[ExperimentPlan]) -> list[ExperimentPlan]:
        """Sort experiments respecting depends_on edges (Kahn's algorithm)."""
        by_id = {p.experiment_id: p for p in plans}
        in_degree = {p.experiment_id: 0 for p in plans}
        adj: dict[str, list[str]] = {p.experiment_id: [] for p in plans}

        for p in plans:
            for dep in p.depends_on:
                if dep in adj:
                    adj[dep].append(p.experiment_id)
                    in_degree[p.experiment_id] += 1

        queue = [eid for eid, deg in in_degree.items() if deg == 0]
        ordered: list[ExperimentPlan] = []

        while queue:
            eid = queue.pop(0)
            ordered.append(by_id[eid])
            for child in adj[eid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        # If some experiments remain (cycle), append them anyway
        visited = {p.experiment_id for p in ordered}
        for p in plans:
            if p.experiment_id not in visited:
                ordered.append(p)

        return ordered
