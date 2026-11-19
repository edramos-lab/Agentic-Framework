import React, { useEffect, useRef, useState } from "react";
import PipelineBoard from "../components/PipelineBoard";
import { Button, PageHeader, StatusBadge } from "../components/ui";

/* A scripted run for previewing the cards without LLM keys, Colab or a Jetson.
   Each entry fires `at` seconds after Play and patches one agent / step. */

const NAMES = [
  "OutlineAgent",
  "ExperimentAgent",
  "LiteratureAgent",
  "PlottingAgent",
  "WritingAgent",
  "RefinementAgent",
];

const SCRIPT = [
  { at: 0.5, agent: "OutlineAgent", status: "running" },
  { at: 3, agent: "OutlineAgent", status: "success" },
  { at: 3.2, agent: "ExperimentAgent", status: "running" },
  { at: 3.2, agent: "LiteratureAgent", status: "running" },
  { at: 3.5, agent: "ExperimentAgent", step: { step_id: "train", label: "Train classifier", target: "colab", status: "running" } },
  { at: 6.5, agent: "LiteratureAgent", status: "success" },
  { at: 9, agent: "ExperimentAgent", step: { step_id: "train", label: "Train classifier", target: "colab", status: "success" } },
  { at: 9.2, agent: "ExperimentAgent", step: { step_id: "bench", label: "Edge benchmark (PT, ONNX, TensorRT)", target: "jetson", status: "running" } },
  { at: 14, agent: "ExperimentAgent", step: { step_id: "bench", label: "Edge benchmark (PT, ONNX, TensorRT)", target: "jetson", status: "success", detail: "5 variants, TensorRT INT8 fastest" } },
  { at: 14.2, agent: "ExperimentAgent", status: "success" },
  { at: 14.5, agent: "PlottingAgent", status: "running" },
  { at: 17, agent: "PlottingAgent", status: "success" },
  { at: 17.2, agent: "WritingAgent", status: "running" },
  { at: 22, agent: "WritingAgent", status: "success" },
  { at: 22.2, agent: "RefinementAgent", status: "running" },
  { at: 22.4, agent: "RefinementAgent", step: { step_id: "c1", label: "Cycle 1/3: review", status: "running" } },
  { at: 25, agent: "RefinementAgent", step: { step_id: "c1", label: "Cycle 1/3: review", status: "success", detail: "score 6.8/10 · improvement" } },
  { at: 25.2, agent: "RefinementAgent", step: { step_id: "c2", label: "Cycle 2/3: review", status: "running" } },
  { at: 28, agent: "RefinementAgent", step: { step_id: "c2", label: "Cycle 2/3: review", status: "success", detail: "score 7.4/10 · improvement" } },
  { at: 28.5, agent: "RefinementAgent", status: "success" },
];

const blank = () => ({
  status: "pending",
  agents: NAMES.map((name) => ({ name, status: "pending", message: "", steps: [] })),
});

function apply(run, ev) {
  const now = new Date().toISOString();
  const agents = run.agents.map((a) => {
    if (a.name !== ev.agent) return a;
    if (ev.step) {
      const steps = [...a.steps];
      const i = steps.findIndex((s) => s.step_id === ev.step.step_id);
      const prev = i >= 0 ? steps[i] : { started_at: now };
      const next = { ...prev, ...ev.step, finished_at: ev.step.status === "running" ? null : now };
      if (i >= 0) steps[i] = next;
      else steps.push(next);
      return { ...a, steps };
    }
    return {
      ...a,
      status: ev.status,
      started_at: a.started_at || now,
      finished_at: ev.status === "running" ? null : now,
    };
  });
  const allDone = agents.every((a) => a.status === "success");
  return { ...run, status: allDone ? "completed" : "running", agents };
}

export default function Demo() {
  const [run, setRun] = useState(blank);
  const [playing, setPlaying] = useState(false);
  const timers = useRef([]);

  const stop = () => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  };
  useEffect(() => stop, []);

  const play = (speed = 1) => {
    stop();
    setRun(blank());
    setPlaying(true);
    SCRIPT.forEach((ev) =>
      timers.current.push(setTimeout(() => setRun((r) => apply(r, ev)), (ev.at * 1000) / speed))
    );
    const end = (SCRIPT[SCRIPT.length - 1].at * 1000) / speed + 300;
    timers.current.push(setTimeout(() => setPlaying(false), end));
  };

  // A failed run, to see the red state
  const playFailure = () => {
    stop();
    setRun(blank());
    setPlaying(true);
    const fail = [
      { at: 0.4, agent: "OutlineAgent", status: "running" },
      { at: 1.5, agent: "OutlineAgent", status: "success" },
      { at: 1.7, agent: "ExperimentAgent", status: "running" },
      { at: 1.7, agent: "LiteratureAgent", status: "running" },
      { at: 2, agent: "ExperimentAgent", step: { step_id: "train", label: "Train classifier", target: "colab", status: "success" } },
      { at: 2.2, agent: "ExperimentAgent", step: { step_id: "bench", label: "Edge benchmark", target: "jetson", status: "running" } },
      { at: 4, agent: "ExperimentAgent", step: { step_id: "bench", label: "Edge benchmark", target: "jetson", status: "failed", detail: "Permission denied (publickey,password)" } },
      { at: 4.2, agent: "ExperimentAgent", status: "failed" },
      { at: 4.5, agent: "LiteratureAgent", status: "success" },
    ];
    fail.forEach((ev) => timers.current.push(setTimeout(() => setRun((r) => apply(r, ev)), ev.at * 1000)));
    timers.current.push(setTimeout(() => setPlaying(false), 4800));
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Pipeline preview"
        description="A scripted run, so you can see the cards change without keys, Colab or a Jetson."
      >
        <StatusBadge status={run.status} />
      </PageHeader>
      <div className="flex flex-wrap gap-3">
        <Button onClick={() => play(1)} disabled={playing}>
          Play successful run
        </Button>
        <Button variant="secondary" onClick={() => play(3)} disabled={playing}>
          Play 3× faster
        </Button>
        <Button variant="secondary" onClick={playFailure} disabled={playing}>
          Play a failure
        </Button>
      </div>
      <PipelineBoard agents={run.agents} runStatus={run.status} />
    </div>
  );
}
