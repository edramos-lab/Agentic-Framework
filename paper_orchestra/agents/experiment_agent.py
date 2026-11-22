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
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from ..base_agent import BaseAgent
from ..domains import domain_of
from ..utils.metrics import compact_metrics
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
        self._dataset = exp_reqs.dataset if exp_reqs else None
        self._domain = domain_of(exp_reqs)
        timeout = exp_reqs.timeout_minutes * 60 if exp_reqs else 7200
        colab_cfg = exp_reqs.colab if exp_reqs else ColabConfig()
        jetson_cfg = exp_reqs.jetson if exp_reqs else JetsonConfig()
        scripts_dir = exp_reqs.scripts_dir if exp_reqs else ""
        default_target = exp_reqs.backend if exp_reqs else "colab"

        ordered = self._topological_sort(state.outline.experiment_plan)
        reused = self._reuse_results(ordered, Path(exp_reqs.reuse_run_dir), colab_cfg, default_target, set(exp_reqs.reuse_rerun)) \
            if exp_reqs and exp_reqs.reuse_run_dir else {}
        targets = {p.experiment_id: (p.target or default_target) for p in ordered if p.experiment_id not in reused}
        host = os.environ.get("JETSON_HOST") or jetson_cfg.host

        # Colab is for training, the Jetson for edge deployment.  Each is only
        # needed when some plan targets it.
        use_colab = "colab" in targets.values()
        if use_colab and not self._colab_cli_available():
            self.logger.error("google-colab-cli not found. Install with: pip install google-colab-cli")
            self._skip_target(ordered, targets, "colab", "google-colab-cli not installed")
            use_colab = False
        elif use_colab and not self._colab_signed_in():
            msg = "the colab CLI is not signed in (run `colab status` once and paste the login code)"
            self.logger.error("Colab: %s", msg)
            self._skip_target(ordered, targets, "colab", msg)
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
            if plan.experiment_id in reused:
                result = reused[plan.experiment_id]
                results.append(result)
                done[plan.experiment_id] = result
                self.step(plan.experiment_id, plan.name, "success", result.target, result.colab_session.replace("reused:", "reused from run "))
                continue
            target = targets.get(plan.experiment_id)
            if target is None:
                continue  # its target is unavailable; already logged and reported
            unmet = [d for d in plan.depends_on if d not in done or done[d].status != "success"]
            if unmet:  # running on the output of a step that did not produce it would measure something else
                msg = f"skipped: it needs {', '.join(unmet)}, which did not succeed"
                self.logger.error("  %s %s", plan.experiment_id, msg)
                now = datetime.now(timezone.utc)
                skipped = ExperimentResult(experiment_id=plan.experiment_id, status="failed", target=target,
                                           error_message=msg, started_at=now, finished_at=now)
                results.append(skipped)
                done[plan.experiment_id] = skipped
                self.step(plan.experiment_id, plan.name, "skipped", target, msg)
                continue
            self.step(plan.experiment_id, plan.name, "running", target)
            if target == "jetson":
                # Carry artifacts (trained model files) from the experiments it depends on
                carried = [a for dep in plan.depends_on if dep in done for a in done[dep].artifacts]
                result = self._run_experiment_jetson(plan, jetson_cfg, host, timeout, scripts_dir, carried)
            else:
                carried = [a for dep in plan.depends_on if dep in done for a in done[dep].artifacts]
                result = self._run_experiment(plan, timeout, scripts_dir, colab_cfg, shared_session, carried)
                result.target = "colab"
                result.device = f"Google Colab {colab_cfg.gpu_type or colab_cfg.tpu_type}"
            results.append(result)
            done[plan.experiment_id] = result
            self.step(plan.experiment_id, plan.name, result.status, target,
                      result.error_message[:200] if result.status == "failed" else "")

        if shared_session and not colab_cfg.keep_session:
            self._stop_session(shared_session)

        self._cross_check(results)
        history = self._supplied_history(scripts_dir)
        if history:
            results.append(history)

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

    def _reuse_results(self, plans: list[ExperimentPlan], prev: Path, colab_cfg: ColabConfig,
                       default_target: str, rerun: set[str] = frozenset()) -> dict[str, ExperimentResult]:
        """Results of experiments an earlier run already finished: same script, every expected output present.

        The earlier run's outputs are copied into this run, so the figures, tables and the paper draw on them exactly
        as on fresh ones; the result says which run it came from.
        """
        try:
            previous = json.loads((prev / "outlineagent" / "outline.json").read_text())["experiment_plan"]
        except (OSError, KeyError, json.JSONDecodeError):
            self.logger.warning("  Nothing to reuse: no outline in %s", prev)
            return {}
        by_script = {p["script"]: p["experiment_id"] for p in previous if p.get("script")}
        out: dict[str, ExperimentResult] = {}
        for plan in plans:
            if plan.script in rerun:
                continue  # the caller asked for this one to run again
            folder = prev / "experimentagent" / by_script.get(plan.script, "-")
            wanted = plan.expected_outputs
            if not wanted or not all((folder / name).is_file() for name in wanted):
                continue  # incomplete there: run it again
            dest = self.agent_dir / plan.experiment_id
            dest.mkdir(parents=True, exist_ok=True)
            artifacts = []
            for name in wanted:
                shutil.copy(folder / name, dest / name)
                artifacts.append(dest / name)
            metrics: dict = {}
            for art in artifacts:
                if art.suffix == ".json":
                    try:
                        metrics.update(json.loads(art.read_text()))
                    except (json.JSONDecodeError, OSError):
                        pass
            target = plan.target or default_target
            now = datetime.now(timezone.utc)
            device = (f"Google Colab {colab_cfg.gpu_type or colab_cfg.tpu_type}" if target == "colab"
                      else metrics.get("device") or "NVIDIA Jetson Orin Nano")
            out[plan.experiment_id] = ExperimentResult(
                experiment_id=plan.experiment_id, status="success", target=target, device=device,
                colab_session=f"reused:{prev.name}", artifacts=artifacts, metrics=metrics, started_at=now, finished_at=now)
            self.logger.info("  %s (%s): reused from run %s", plan.experiment_id, plan.script, prev.name)
        return out

    def _dataset_env(self) -> dict[str, str]:
        """The form's dataset section as env vars for the Colab scripts (nothing for the leukemia recipe, whose scripts read none)."""
        d, dom = getattr(self, "_dataset", None), getattr(self, "_domain", None)
        if d is None or dom is None or dom.name == "microscopy":
            return {}
        env = {"DATASET_SOURCE": d.source, "DATASET_ID": d.id, "SPLIT": d.split, "PREPROCESS": dom.preprocess, **dom.env}
        if d.group_regex:
            env["GROUP_REGEX"] = d.group_regex
        return env

    @staticmethod
    def _cross_check(results: list[ExperimentResult]) -> None:
        """Say whether the Jetson's predictions equal those of the trained model on the same test images."""
        trained = next((r for r in results if r.status == "success" and r.metrics.get("predictions") and r.metrics.get("backbone")), None)
        bench = next((r for r in results if r.status == "success" and r.metrics.get("variants")), None)
        if not trained or not bench:
            return
        ref, out = trained.metrics["predictions"], {}
        for key, v in bench.metrics["variants"].items():
            pred = (v.get("accuracy") or {}).get("predictions")
            if pred is not None and len(pred) == len(ref):
                out[key] = {"equal": int(sum(a == b for a, b in zip(pred, ref))), "n": len(ref)}
        if out:
            bench.metrics["agreement_with_colab_test_predictions"] = out

    def _skip_target(self, plans, targets: dict[str, str], target: str, why: str) -> None:
        """Drop every plan aimed at an unavailable target and tell the UI why."""
        for plan in plans:
            if targets.get(plan.experiment_id) == target:
                self.step(plan.experiment_id, plan.name, "skipped", target, why)
                del targets[plan.experiment_id]

    def _supplied_history(self, scripts_dir: str) -> ExperimentResult | None:
        """Per-epoch training/validation curves the author recorded (e.g. in Colab) as training_history.json.

        The pipeline cannot train, so curves exist only when the author supplies them; they are labelled as such.
        """
        path, _ = self._resolve_script(scripts_dir, "training_history.json")
        if path is None:
            return None
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning("training_history.json unreadable: %s", exc)
            return None
        self.logger.info("Using the author's training_history.json (%s)", path)
        return ExperimentResult(
            experiment_id="training_history", status="success", target="colab",
            device="Google Colab (curves supplied by the author)", metrics={"history": history},
        )

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

    @staticmethod
    def _sibling_imports(script: Path) -> list[Path]:
        """Modules next to `script` that it imports, directly or through another of them, so they travel with it."""
        found: list[Path] = []
        todo = [script]
        while todo:
            current = todo.pop()
            names = re.findall(r"^\s*(?:from|import)\s+(\w+)", current.read_text(encoding="utf-8", errors="ignore"), re.M)
            for n in dict.fromkeys(names):
                f = current.with_name(f"{n}.py")
                if f.is_file() and f != script and f not in found:
                    found.append(f)
                    todo.append(f)
        return found

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
        # one folder per run: a leftover model or cache from an earlier run must not be picked up by this one
        remote_dir = f"{cfg.remote_dir}/{self.agent_dir.parent.name}/{plan.experiment_id}"
        artifact_dir = self.agent_dir / plan.experiment_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()

        try:
            for attempt in range(1, 4):  # the Jetson sits behind a VPN that now and then drops for a moment
                try:
                    self._run_cmd(self._ssh_base(cfg, host, "ssh") + [target, f"mkdir -p {shlex.quote(remote_dir)}"],
                                  timeout=45, check=True)
                    break
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    if attempt == 3:
                        raise
                    self.logger.warning("  Jetson not reachable (attempt %d/3): %s", attempt, str(exc)[:120])
                    time.sleep(30)
            self._run_cmd(
                self._ssh_base(cfg, host, "scp") + [str(script_path), f"{target}:{remote_dir}/"],
                timeout=120, check=True,
            )
            for helper in self._sibling_imports(script_path):
                self._run_cmd(self._ssh_base(cfg, host, "scp") + [str(helper), f"{target}:{remote_dir}/"], timeout=120, check=True)
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
                if not result.metrics:
                    # The planned output file may not match what the script writes; the
                    # benchmark also prints its results as JSON, so read them from stdout.
                    parsed = self._json_from_stdout(proc.stdout)
                    if parsed:
                        saved = artifact_dir / "stdout_metrics.json"
                        saved.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
                        result.artifacts.append(saved)
                        result.metrics.update(parsed)
                result.device = result.metrics.get("device", result.device)
        except subprocess.TimeoutExpired:
            result.status = "failed"
            result.error_message = f"Timed out after {timeout}s"
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            result.status = "failed"
            result.error_message = (getattr(exc, "stderr", None) or str(exc))[:2000]

        result.finished_at = datetime.now(timezone.utc)
        return result

    @staticmethod
    def _json_from_stdout(text: str) -> dict:
        """Return the JSON object a script printed to stdout, or {} when there is none."""
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    # ----- Colab CLI helpers -------------------------------------------------

    @staticmethod
    def _colab_cli_available() -> bool:
        """Check whether the `colab` CLI is on PATH."""
        return shutil.which("colab") is not None

    def _colab_signed_in(self) -> bool:
        """True when the colab CLI holds valid credentials (signed out, `colab status` asks for a login code)."""
        try:
            proc = self._run_cmd(["colab", "status"], timeout=60)
        except subprocess.TimeoutExpired:
            return False
        return "authorization code" not in (proc.stdout + proc.stderr)

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
            stdin=subprocess.DEVNULL,  # a CLI that asks for a login code must fail, not wait forever
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
        for attempt in range(1, 4):  # a busy GPU type can take minutes to hand out
            try:
                proc = self._run_cmd(cmd, timeout=300)
                if proc.returncode == 0:
                    self.logger.info("  Session '%s' ready.", cfg.session_name)
                    return cfg.session_name
                self.logger.error("  colab new failed (attempt %d/3): %s", attempt, (proc.stderr or proc.stdout).strip()[-300:])
            except subprocess.TimeoutExpired:
                self.logger.error("  colab new did not answer within 300 s (attempt %d/3)", attempt)
            except FileNotFoundError as exc:
                self.logger.error("  Failed to provision session: %s", exc)
                return None
            self._stop_session(cfg.session_name)  # never leave a half-created, possibly billed, session behind
            if attempt < 3:
                time.sleep(30)
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

    # ----- running a long script on Colab -----------------------------------

    def _upload_secret_file(self, session: str, secret: str, remote: str) -> None:
        """Send a secret to the VM as a file (uploads are logged by name only; executed code is logged in full)."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / Path(remote).name
            f.write_text(secret)
            f.chmod(0o600)
            r = self._run_cmd(["colab", "upload", "-s", session, str(f), remote], timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"could not send {Path(remote).name} to Colab: " + (r.stderr or r.stdout).strip()[-200:])

    def _upload_kaggle_token(self, session: str) -> None:
        """Send the Kaggle token to the VM as a file: file uploads are logged by name only, executed code is logged in full."""
        import json as _json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            token = Path(tmp) / "kaggle.json"
            token.write_text(_json.dumps({"username": os.environ["KAGGLE_USERNAME"], "key": os.environ["KAGGLE_KEY"]}))
            token.chmod(0o600)
            r = self._run_cmd(["colab", "upload", "-s", session, str(token), "/content/kaggle.json"], timeout=120)
        if r.returncode != 0:
            raise RuntimeError("could not send the Kaggle token to Colab: " + (r.stderr or r.stdout).strip()[-200:])

    COLAB_UPLOAD_CHUNK = 32 * 1024 * 1024  # Colab's contents API answers HTTP 500 to a single upload of ~80 MB or more

    def _colab_upload(self, session: str, local: Path, remote: str, eid: str) -> None:
        """Upload a file to the VM; one that is too big for a single request goes up in parts and is rejoined there."""
        def put(src: Path, dest: str) -> None:
            r = self._run_cmd(["colab", "upload", "-s", session, str(src), dest], timeout=1800)
            if r.returncode != 0:
                raise RuntimeError(f"upload of {src.name} to Colab failed: {(r.stderr or r.stdout).strip()[-300:]}")

        size = local.stat().st_size
        if size <= self.COLAB_UPLOAD_CHUNK:
            put(local, remote)
            return
        parts = []
        parts_dir = self.agent_dir / f"{eid}_parts"
        parts_dir.mkdir(exist_ok=True)
        try:
            with local.open("rb") as f:
                while chunk := f.read(self.COLAB_UPLOAD_CHUNK):
                    part = parts_dir / f"{local.name}.part{len(parts):03d}"
                    part.write_bytes(chunk)
                    put(part, f"{remote}.part{len(parts):03d}")
                    part.unlink()
                    parts.append(part)
        finally:
            shutil.rmtree(parts_dir, ignore_errors=True)
        join = (
            "import glob, os\n"
            f"parts = sorted(glob.glob({remote!r} + '.part*'))\n"
            f"with open({remote!r}, 'wb') as out:\n"
            "    for p in parts:\n"
            "        with open(p, 'rb') as f:\n"
            "            while True:\n"
            "                b = f.read(1 << 24)\n"
            "                if not b: break\n"
            "                out.write(b)\n"
            "        os.remove(p)\n"
            f"print('JOINED', os.path.getsize({remote!r}), len(parts))\n"
        )
        r = self._colab_exec_code(session, f"{eid}_join_{local.stem}", join)
        if f"JOINED {size} {len(parts)}" not in r.stdout:
            raise RuntimeError(f"rejoining {local.name} on Colab failed: {(r.stderr or r.stdout).strip()[-300:]}")
        self.logger.info("  Uploaded %s to Colab in %d parts (%.0f MB)", local.name, len(parts), size / 1e6)

    def _colab_exec_code(self, session: str, name: str, code: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
        """One short `colab exec` of a snippet; a lost connection costs only this call."""
        snippet = self.agent_dir / f"{name}.py"
        snippet.write_text(code, encoding="utf-8")
        return self._run_cmd(["colab", "exec", "-s", session, "--timeout", "0", "-f", str(snippet)], timeout=timeout)

    def _run_colab_background(self, session: str, plan: ExperimentPlan, remote_script: str, args: list[str],
                              env: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
        """Start the script as a detached process on the VM, then poll it with short `colab exec` calls.

        One long `colab exec` is a single websocket that has to survive the whole training run; it drops after a
        minute or two.  The detached process does not care: it keeps running and the poll just asks how it is doing.
        """
        eid = plan.experiment_id
        out, done = f"/content/{eid}.out", f"/content/{eid}.exit"
        launch = (
            "import os, subprocess, sys\n"
            f"for f in ({out!r}, {done!r}):\n"
            "    if os.path.exists(f): os.remove(f)\n"
            f"cmd = '{{}} {{}} {{}} > {{}} 2>&1; echo $? > {{}}'.format(sys.executable, {remote_script!r}, "
            f"' '.join(__import__('shlex').quote(a) for a in {args!r}), {out!r}, {done!r})\n"
            f"subprocess.Popen(['bash', '-c', cmd], cwd='/content', env={{**os.environ, **{env!r}}}, "
            "start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "print('LAUNCHED')\n"
        )
        for attempt in range(1, 4):
            r = self._colab_exec_code(session, f"{eid}_launch", launch)
            if r.returncode == 0 and "LAUNCHED" in r.stdout:
                break
            if attempt == 3:
                raise RuntimeError("could not start the script on Colab: " + (r.stderr or r.stdout)[-400:])
            time.sleep(10)

        poll = (
            "import os\n"
            f"o, d = {out!r}, {done!r}\n"
            "text = open(o, errors='replace').read() if os.path.exists(o) else ''\n"
            "code = open(d).read().strip() if os.path.exists(d) else ''\n"
            "print('STATE:' + ('done:' + code if code else 'running'))\n"
            "print('SIZE:%d' % len(text))\n"
            "print(text[-15000:] if code else text[-300:])\n"
        )
        deadline, lost, tail, state = time.monotonic() + timeout, 0, "", ""
        while time.monotonic() < deadline:
            time.sleep(45)
            try:
                r = self._colab_exec_code(session, f"{eid}_poll", poll)
            except subprocess.TimeoutExpired:
                r = None
            if r is None or r.returncode != 0 or "STATE:" not in r.stdout:
                lost += 1
                self.logger.warning("  Colab poll failed (%d/8): %s", lost, ((r.stderr or r.stdout)[-160:] if r else "timeout"))
                if lost >= 8:
                    raise RuntimeError("lost contact with the Colab session")
                continue
            lost = 0
            state = re.search(r"STATE:(\S+)", r.stdout).group(1)
            tail = r.stdout.split("SIZE:", 1)[-1].split("\n", 1)[-1]
            last = [ln for ln in tail.strip().splitlines() if ln.strip()]
            if last:
                self.step(eid, plan.name, "running", "colab", re.sub(r"\s+", " ", last[-1])[:160])
            if state.startswith("done"):
                break
        else:
            raise subprocess.TimeoutExpired(["colab", "exec", remote_script], timeout)
        code = int(state.split(":")[1] or 1)
        return subprocess.CompletedProcess(["colab", "exec", remote_script], code, tail, tail if code else "")

    def _run_experiment(
        self,
        plan: ExperimentPlan,
        timeout: int,
        scripts_dir: str,
        colab_cfg: ColabConfig,
        shared_session: str | None = None,
        carried: list[Path] | None = None,
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

            # Files produced by the experiments this one depends on (e.g. the packed training images)
            sources = "\n".join(f.read_text(encoding="utf-8", errors="ignore")
                               for f in [script_path, *self._sibling_imports(script_path)])
            source = getattr(self._dataset, "source", "kaggle")  # a token goes only to a session whose data come from that service
            if "kaggle_data" in sources and source == "kaggle" and os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
                self._upload_kaggle_token(session_name)
            if "zenodo_data" in sources and source == "zenodo" and os.environ.get("ZENODO_TOKEN"):
                self._upload_secret_file(session_name, os.environ["ZENODO_TOKEN"], "/content/zenodo_token")
            for extra in carried or []:
                # a 185 MB test-image zip or a model the script never reads is not worth a minute of upload
                if Path(extra).is_file() and Path(extra).name in sources:
                    self._colab_upload(session_name, Path(extra), f"/content/{Path(extra).name}", plan.experiment_id)

            # --- 2. Upload the experiment script ---
            remote_script = f"/content/{script_path.name}"
            upload_cmd = ["colab", "upload", "-s", session_name, str(script_path), remote_script]
            self._run_cmd(upload_cmd, timeout=60)
            for helper in self._sibling_imports(script_path):  # e.g. the probe imports triton_benchmark
                self._run_cmd(["colab", "upload", "-s", session_name, str(helper), f"/content/{helper.name}"], timeout=60)

            # --- 3. Run the script in the background on the VM and poll it ---
            self.logger.info("  Executing on Colab: %s", plan.script)
            # no Colab-secret lookup that times out; scripts that can run on the Jetson or on Colab are told which they are on
            env = {**colab_cfg.env_vars, "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "PROBE_BACKEND": "torch", **self._dataset_env()}
            proc = self._run_colab_background(session_name, plan, remote_script, script_args, env, timeout)
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

                # `colab exec` exits 0 even when the script raised, so a crash shows only as missing outputs
                got = {a.name for a in downloaded}
                missing = [o for o in plan.expected_outputs if Path(o).name not in got]
                err = re.sub(r"\x1b?\[[0-9;]*m", "", proc.stderr)
                if plan.expected_outputs and (not downloaded or (missing and "Traceback (most recent call last)" in err)):
                    result.status = "failed"
                    result.error_message = (f"no {', '.join(missing)} produced; " + err.strip()[-700:])[:2000]

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
            metrics_str = ", ".join(f"{k}={v}" for k, v in list(compact_metrics(r.metrics).items())[:5] if not isinstance(v, (dict, list)))
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
                for k, v in compact_metrics(r.metrics).items():
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
            ordered = sorted(variants.items(), key=lambda kv: list(labels).index(kv[0]) if kv[0] in labels else len(labels))
            for key, v in ordered:
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
            acc_tex = self._accuracy_table(variants, labels)
            if acc_tex:
                self._write(f"{r.experiment_id}_accuracy_table.tex", acc_tex)
                tables[f"{r.experiment_id} accuracy"] = acc_tex
        for r in results:
            probe_tex = self._probe_table(r.metrics) if r.status == "success" else ""
            if probe_tex:
                self._write(f"{r.experiment_id}_probe_table.tex", probe_tex)
                name = "shortcut probe" if getattr(self, "_domain", None) is not None and self._domain.name == "photos" else "morphology probe"
                tables[f"{r.experiment_id} {name}"] = probe_tex
            for key, tex in (self._stats_tables(r.metrics) if r.status == "success" else {}).items():
                self._write(f"{r.experiment_id}_{key.replace(' ', '_')}_table.tex", tex)
                tables[f"{r.experiment_id} {key}"] = tex
            more = {**self._cv_tables(r.metrics), **self._audit_tables(r.metrics), **self._pose_tables(r.metrics), **self._benchmark_tables(r.metrics), **self._masked_tables(r.metrics)} if r.status == "success" else {}
            for key, tex in more.items():
                self._write(f"{r.experiment_id}_{key.replace(' ', '_')}_table.tex", tex)
                tables[f"{r.experiment_id} {key}"] = tex
        return tables

    @staticmethod
    def _accuracy_table(variants: dict, labels: dict[str, str]) -> str:
        """Accuracy, macro-F1 and AUC of each deployed variant on the held-out test images."""
        rows = []
        for key in labels:
            acc = (variants.get(key) or {}).get("accuracy")
            if not acc:
                continue
            note = "" if acc.get("valid", True) else "$^{*}$"
            rows.append(f"{labels[key]}{note} & {acc['accuracy']:.4f} & {acc['macro_f1']:.4f} & {acc['macro_auc']:.4f} \\\\")
        if not rows:
            return ""
        tex = ("\\begin{tabular}{lccc}\n\\hline\nVariant & Accuracy & Macro-F1 & Macro-AUC \\\\\n\\hline\n"
               + "\n".join(rows) + "\n\\hline\n")
        if any(not (variants.get(k) or {}).get("accuracy", {}).get("valid", True) for k in labels):
            tex += ("\\multicolumn{4}{l}{\\footnotesize $^{*}$ INT8 engine built without a calibration cache; "
                    "accuracy is not representative.} \\\\\n")
        return tex + "\\end{tabular}\n"

    @staticmethod
    def _cv_tables(metrics: dict) -> dict[str, str]:
        """Backbones compared by cross-validation (with the reference rows), and the single scoring of the final model."""
        if "backbones" not in metrics or "references" not in metrics:
            return {}
        rows, ci = [], lambda d: f"{d['mean']:.3f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]" if "ci95" in d else f"{d['mean']:.3f}"
        for name, d in {**metrics["backbones"], **metrics["references"]}.items():
            mark = "$^{\\dagger}$" if name == metrics.get("selected") else ""
            rows.append(f"{name.replace('_', ' ')}{mark} & {ci(d['balanced_accuracy'])} & {d['macro_f1']['mean']:.3f} & {ci(d['accuracy'])} & "
                        f"{d['macro_auc']['mean']:.3f} & {d['ece']['mean']:.3f} \\\\")
        t1 = ("\\begin{tabular}{lccccc}\n\\hline\nModel & Balanced accuracy [95\\% CI] & Macro-F1 & Accuracy [95\\% CI] & Macro-AUC & ECE \\\\\n\\hline\n"
              + "\n".join(rows) + "\n\\hline\n\\multicolumn{6}{l}{\\footnotesize $^{\\dagger}$ selected by balanced accuracy; "
              "out-of-fold, averaged over repeats.} \\\\\n\\end{tabular}\n")
        out = {"cross-validation": t1}
        if metrics.get("predictions") and metrics.get("true"):
            classes = [c.replace("_", " ") for c in metrics["classes"]]
            rec = " & ".join("n/a" if r is None else f"{r:.2f}" for r in metrics.get("per_class_recall", []))
            lo, hi = metrics.get("accuracy_ci95", [float("nan")] * 2)
            out["test split"] = ("\\begin{tabular}{l" + "c" * (len(classes) + 2) + "}\n\\hline\nImages & Accuracy [95\\% CI] & " + " & ".join(f"Recall {c}" for c in classes)
                                 + f" \\\\\n\\hline\n{metrics['n_test']} & {metrics['accuracy']:.3f} [{lo:.2f}, {hi:.2f}] & {rec} \\\\\n\\hline\n\\end{{tabular}}\n")
        return out

    @staticmethod
    def _audit_tables(metrics: dict) -> dict[str, str]:
        """The photograph audit: per-class statistics of a few features, and what classifiers learn from each feature group."""
        if "classifiers" not in metrics or "per_class" not in metrics or "features" not in metrics:
            return {}
        classes = metrics["classes"]
        feats = [("bg_L", "Background $L^*$"), ("bg_a", "Background $a^*$"), ("bg_b", "Background $b^*$"), ("person_fraction", "Person area share"),
                 ("person_box_aspect", "Person box height/width"), ("log_file_bytes", "Log file size"), ("capture_day", "Capture day")]
        rows = []
        for key, label in feats:
            if key not in metrics["features"]:
                continue
            f = metrics["features"][key]
            vals = " & ".join(f"{metrics['per_class'][c][key]['mean']:.2f} $\\pm$ {metrics['per_class'][c][key]['std']:.2f}" for c in classes)
            p = f["p"]
            rows.append(f"{label} & {vals} & {'$<$0.001' if p < 0.001 else format(p, '.3f')} & {f['eta_squared']:.2f} \\\\")
        names = [c.replace("_", " ") for c in classes]
        t1 = (f"\\begin{{tabular}}{{l{'c' * len(classes)}cc}}\n\\hline\nFeature & {' & '.join(names)} & Kruskal $p$ & $\\eta^2$ \\\\\n\\hline\n"
              + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")
        crow = []
        for key, d in metrics["classifiers"].items():
            p = d["permutation_p"]
            crow.append(f"{key.replace('_', ' ')} & {d['cv_accuracy']:.3f} & {d['majority_baseline']:.3f} & "
                        f"{'$<$' + format(1 / (d['permutations'] + 1), '.3f') if p <= 1 / (d['permutations'] + 1) + 1e-9 else format(p, '.3f')} \\\\")
        t2 = ("\\begin{tabular}{lccc}\n\\hline\nFeatures given to the classifier & CV accuracy & Majority baseline & Permutation $p$ \\\\\n\\hline\n"
              + "\n".join(crow) + "\n\\hline\n\\end{tabular}\n")
        return {"dataset audit": t1, "audit classifiers": t2}

    @staticmethod
    def _benchmark_tables(metrics: dict) -> dict[str, str]:
        """Pose-detector benchmark: speed and availability, agreement between detectors, geometry classifier per detector."""
        dets = metrics.get("detectors")
        if not dets or "agreement" not in metrics:
            return {}
        classes = metrics["classes"]
        label = {k: v["label"] for k, v in dets.items()}
        pct = lambda v: f"{100 * v:.0f}\\%"
        rows = []
        for k, d in dets.items():
            dev = "CPU" if d["device"].startswith("cpu") else "GPU"
            ov, lat = d["availability_overall"], d["latency_ms"]
            p = d["availability_test"]["p"]
            rows.append(f"{d['label']} & {d['person_finder']} & {dev} & {lat['median']:.0f} / {lat['p90']:.0f} & {pct(ov['both_hips'])} & {pct(ov['both_knees'])} & {pct(ov['both_ankles'])} & "
                        f"{d['knees_and_ankles_found']} & {'$<$0.001' if p < 0.001 else format(p, '.3f')} \\\\")
        t1 = ("\\begin{tabular}{llcccccc c}\n\\hline\nDetector & Person finder & Run on & ms median / p90 & Hips & Knees & Ankles & Knees+ankles (images) & Class $p$ \\\\\n\\hline\n"
              + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n").replace("{llcccccc c}", "{llccccccc}")
        arows = []
        for pair, a in metrics["agreement"].items():
            x, z = pair.split("|")
            f = lambda v: "n/a" if v is None else (f"{v:.1f}" if isinstance(v, float) and abs(v) > 1.5 else f"{v:.2f}")
            arows.append(f"{label[x]} vs {label[z]} & {f(a['hip_px'])} & {f(a['knee_px'])} & {f(a['ankle_px'])} & {f(a['spearman_knee_over_ankle_gap'])} & {f(a['spearman_tibia_inward_tilt'])} & {a['images_both_complete']} \\\\")
        t2 = ("\\begin{tabular}{lcccccc}\n\\hline\nPair & Hip (px) & Knee (px) & Ankle (px) & $\\rho$ knee/ankle gap & $\\rho$ tibial tilt & Images \\\\\n\\hline\n"
              + "\n".join(arows) + "\n\\hline\n\\end{tabular}\n")
        grows = []
        for k, d in dets.items():
            g = d.get("geometry_classifier")
            if not g:
                grows.append(f"{d['label']} & n/a & n/a & n/a & n/a \\\\")
                continue
            pp = g["permutation_p"]
            per = "/".join(str(g["n_per_class"].get(c, "")) for c in classes)
            grows.append(f"{d['label']} & {g['n']} ({per}) & {g['cv_accuracy']:.3f} & {g['majority_baseline']:.3f} & "
                         f"{'$<$' + format(1 / (g['permutations'] + 1), '.3f') if pp <= 1 / (g['permutations'] + 1) + 1e-9 else format(pp, '.3f')} \\\\")
        t3 = ("\\begin{tabular}{lcccc}\n\\hline\nDetector & Images (per class) & CV accuracy & Majority baseline & Permutation $p$ \\\\\n\\hline\n"
              + "\n".join(grows) + "\n\\hline\n\\end{tabular}\n")
        return {"keypoint detector speed and availability": t1, "keypoint detector agreement": t2, "keypoint detector geometry classifiers": t3}

    @staticmethod
    def _masked_tables(metrics: dict) -> dict[str, str]:
        """Segmentation networks trained and tested on photographs with the person masked out, against the shuffled-label reference."""
        if "mask_grow_px" not in metrics or "models" not in metrics:
            return {}
        names = {"original": "Original", "person_masked": "Person masked (silhouette stays)", "scene_only": "Person inpainted (room only)", "silhouette_only": "Silhouette only"}
        rows = []
        for m, forms in metrics["models"].items():
            for f, v in forms.items():
                rows.append(f"{m.replace('_', ' ')} & {names.get(f, f)} & {v['accuracy']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] & {v['balanced_accuracy']:.3f} & "
                            f"{v['shuffled_label_accuracy']:.3f} & {v['verdict']} \\\\")
        t = (f"\\begin{{tabular}}{{llcccl}}\n\\hline\nModel & Input & Accuracy [95\\% CI] & Balanced acc. & Shuffled labels & Verdict (majority {metrics['majority_baseline']:.3f}) \\\\\n\\hline\n"
             + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")
        return {"segmentation models on masked photographs": t}

    @staticmethod
    def _pose_tables(metrics: dict) -> dict[str, str]:
        """Keypoint detection rates per class, the leg geometry per class, and the classifiers given detectability or geometry."""
        if "availability" not in metrics or "classifiers" not in metrics or "detector" not in metrics:
            return {}
        classes, av = metrics["classes"], metrics["availability"]
        names = [c.replace("_", " ") for c in classes]
        label = {"person_detected": "Person found", "both_knees": "Both knees found", "both_ankles": "Both ankles found", "both_hips": "Both hips found"}
        rows = [f"{label[k]} & " + " & ".join(f"{100 * av[c][k]:.0f}\\%" for c in classes) + " \\\\" for k in label]
        t1 = (f"\\begin{{tabular}}{{l{'c' * len(classes)}}}\n\\hline\nKeypoints & {' & '.join(names)} \\\\\n\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")
        feats = [("knee_over_ankle_gap", "Knee gap / ankle gap"), ("tibia_inward_tilt", "Tibia inward tilt ($^\\circ$)"), ("tilt_asymmetry", "Tilt asymmetry ($^\\circ$)"),
                 ("knee_angle_mean", "Hip-knee-ankle angle ($^\\circ$)")]
        grow = []
        for key, lab in feats:
            if key not in metrics.get("features", {}):
                continue
            f, cells = metrics["features"][key], []
            for c in classes:
                d = metrics["per_class"][c][key]
                cells.append("n/a" if d["mean"] is None else f"{d['mean']:.2f} $\\pm$ {d['std']:.2f} ({d['n']})")
            grow.append(f"{lab} & " + " & ".join(cells) + f" & {'$<$0.001' if f['p'] < 0.001 else format(f['p'], '.3f')} & {f['eta_squared']:.2f} \\\\")
        t2 = (f"\\begin{{tabular}}{{l{'c' * len(classes)}cc}}\n\\hline\nFeature (images) & {' & '.join(names)} & Kruskal $p$ & $\\eta^2$ \\\\\n\\hline\n"
              + "\n".join(grow) + "\n\\hline\n\\end{tabular}\n")
        crow = []
        for key, d in metrics["classifiers"].items():
            p = d["permutation_p"]
            per = "/".join(str(d.get("n_per_class", {}).get(c, "")) for c in classes) if d.get("n_per_class") else ""
            crow.append(f"{key.replace('_', ' ')} & {d['n']} ({per}) & {d['cv_accuracy']:.3f} & {d['majority_baseline']:.3f} & "
                        f"{'$<$' + format(1 / (d['permutations'] + 1), '.3f') if p <= 1 / (d['permutations'] + 1) + 1e-9 else format(p, '.3f')} \\\\")
        t3 = ("\\begin{tabular}{lcccc}\n\\hline\nFeatures given to the classifier & Images (per class) & CV accuracy & Majority baseline & Permutation $p$ \\\\\n\\hline\n"
              + "\n".join(crow) + "\n\\hline\n\\end{tabular}\n")
        return {"keypoint detection": t1, "leg geometry": t2, "pose classifiers": t3}

    @staticmethod
    def _probe_table(metrics: dict) -> str:
        """Accuracy (95% bootstrap interval) of each model under every masking / background / stain condition."""
        models = metrics.get("models")
        if not models:
            return ""
        two = {"baseline", "augmented"} <= set(models)
        first = next(iter(models.values()))["conditions"]
        order = [c for c in ("original", "cells_masked", "cells_only", "scale_bar_masked", "person_masked", "person_only") if c in first]
        order += sorted((c for c in first if c.startswith("bg_")), key=lambda c: (first[c].get("protocol_set") == "held_out", c))
        order += sorted((c for c in first if c.startswith("stain_")), key=lambda c: (first[c].get("protocol_set") == "held_out", c))
        order += [c for c in first if c not in order]  # conditions a domain adds (grayscale, blur, ...)

        def label(c: str) -> str:
            tag = {"train": " (train)", "held_out": " (held out)"}.get(first[c].get("protocol_set", ""), "")
            return c.replace("_", " ") + tag

        def cell(m: str, c: str) -> str:
            v = models[m]["conditions"][c]
            return f"{v['accuracy']:.3f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]"

        cmp = (metrics.get("comparison") or {}).get("per_condition", {})
        rows = []
        for c in order:
            cols = [cell(m, c) for m in models]
            if two and c in cmp:
                p = cmp[c]["p"]
                cols.append("$<$0.001" if p < 0.001 else f"{p:.3f}")
            rows.append(f"{label(c)} & " + " & ".join(cols) + " \\\\")
        head = " & ".join(n.capitalize() for n in models) + (" & McNemar $p$" if two else "")
        spec = "l" + "c" * (len(models) + (1 if two else 0))
        return (f"\\begin{{tabular}}{{{spec}}}\n\\hline\nCondition & {head} \\\\\n\\hline\n"
                + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")

    @staticmethod
    def _stats_tables(metrics: dict) -> dict[str, str]:
        """Per-class background statistics with Kruskal-Wallis tests, and the background-only classifier."""
        if "background_only" not in metrics or "per_class" not in metrics:
            return {}
        classes = metrics["classes"]
        names = [c.replace("Malignant_", "").replace("_", " ") for c in classes]
        feats = [("bg_L", "Background $L^*$"), ("bg_a", "Background $a^*$"), ("bg_b", "Background $b^*$"),
                 ("cell_a", "Cell $a^*$"), ("cell_b", "Cell $b^*$"), ("cell_coverage", "Cell coverage")]
        rows = []
        for key, label in feats:
            vals = " & ".join(f"{metrics['per_class'][c][key]['mean']:.2f} $\\pm$ {metrics['per_class'][c][key]['std']:.2f}" for c in classes)
            k = (metrics.get("features") or {}).get(key) or {}
            p = k.get("p")
            ptxt = "n/a" if p is None else ("$<$0.001" if p < 0.001 else f"{p:.3f}")
            eta = "n/a" if k.get("eta_squared") is None else f"{k['eta_squared']:.2f}"
            rows.append(f"{label} & {vals} & {ptxt} & {eta} \\\\")
        t1 = (f"\\begin{{tabular}}{{l{'c' * len(classes)}cc}}\n\\hline\nFeature & {' & '.join(names)} & Kruskal $p$ & $\\eta^2$ \\\\\n\\hline\n"
              + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")
        b, c = metrics["background_only"], metrics["cells_only"]

        def line(name: str, d: dict) -> str:
            p = d["permutation_p"]
            return f"{name} & {d['cv_accuracy']:.3f} & {d['majority_baseline']:.3f} & {'$<$' + format(p, '.3f') if p <= 1 / (d['permutations'] + 1) + 1e-9 else format(p, '.3f')} \\\\"

        t2 = ("\\begin{tabular}{lccc}\n\\hline\nFeatures given to the classifier & CV accuracy & Majority baseline & Permutation $p$ \\\\\n\\hline\n"
              + line("Background only", b) + "\n" + line("Cells only", c) + "\n\\hline\n\\end{tabular}\n")
        return {"dataset statistics": t1, "background classifier": t2}

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
