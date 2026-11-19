import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  IconBook,
  IconChart,
  IconChevron,
  IconCloud,
  IconCpu,
  IconFlask,
  IconList,
  IconPen,
  IconRefresh,
  statusStyle,
} from "./ui";

/* ---------- What each agent is, and how the pipeline flows ---------- */

const AGENTS = {
  OutlineAgent: {
    title: "Outline Architect",
    role: "Plans sections, figures and experiments",
    icon: IconList,
  },
  ExperimentAgent: {
    title: "Experiment Runner",
    role: "Trains on Colab, benchmarks on the Jetson",
    icon: IconFlask,
  },
  LiteratureAgent: {
    title: "Literature Reviewer",
    role: "Finds and cites related work",
    icon: IconBook,
  },
  PlottingAgent: {
    title: "Figure Generator",
    role: "Plots the real experiment results",
    icon: IconChart,
  },
  WritingAgent: {
    title: "Section Writer",
    role: "Drafts the manuscript in your voice",
    icon: IconPen,
  },
  RefinementAgent: {
    title: "Reviewer & Refiner",
    role: "Review cycles until the gains stop",
    icon: IconRefresh,
  },
};

// Plotting waits for the experiments, so it sits after the parallel stage.
const STAGES = [
  { title: "Plan", agents: ["OutlineAgent"] },
  { title: "Run & read", note: "in parallel", agents: ["ExperimentAgent", "LiteratureAgent"] },
  { title: "Visualise", agents: ["PlottingAgent"] },
  { title: "Write", agents: ["WritingAgent"] },
  { title: "Review & refine", agents: ["RefinementAgent"] },
];

/* ---------- Colour tokens (full class names so Tailwind keeps them) ---------- */

const TONE = {
  pending: {
    card: "border-slate-200 bg-white",
    bar: "bg-slate-300",
    dot: "bg-slate-300",
    iconWrap: "bg-slate-100 text-slate-500",
  },
  running: {
    card: "border-blue-300 bg-blue-50/70 shadow-md shadow-blue-100",
    bar: "bg-blue-500",
    dot: "bg-blue-500",
    iconWrap: "bg-blue-100 text-blue-700",
  },
  success: {
    card: "border-emerald-200 bg-emerald-50/60",
    bar: "bg-emerald-500",
    dot: "bg-emerald-500",
    iconWrap: "bg-emerald-100 text-emerald-700",
  },
  failed: {
    card: "border-red-300 bg-red-50",
    bar: "bg-red-500",
    dot: "bg-red-500",
    iconWrap: "bg-red-100 text-red-700",
  },
  skipped: {
    card: "border-dashed border-slate-300 bg-slate-50",
    bar: "bg-slate-300",
    dot: "bg-slate-300",
    iconWrap: "bg-slate-100 text-slate-500",
  },
};
const tone = (status) => TONE[status] || TONE.pending;

const TARGET = {
  colab: { label: "Colab", icon: IconCloud, cls: "bg-amber-50 text-amber-800 ring-amber-200" },
  jetson: { label: "Jetson", icon: IconCpu, cls: "bg-violet-50 text-violet-800 ring-violet-200" },
};

/* ---------- Small helpers ---------- */

function useNow(active) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return undefined;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [active]);
  return now;
}

function elapsed(start, end, now) {
  if (!start) return "";
  const s = Math.max(0, Math.round(((end ? Date.parse(end) : now) - Date.parse(start)) / 1000));
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

// The backend reports "running" for an agent until it ends; sub-steps decide the rest.
const isLive = (status) => status === "running";
const isDone = (status) => status === "success" || status === "skipped";

function stageStatus(members) {
  if (members.some((a) => a.status === "failed")) return "failed";
  if (members.some((a) => isLive(a.status))) return "running";
  if (members.length && members.every((a) => isDone(a.status))) return "success";
  return "pending";
}

function TargetChip({ target }) {
  const t = TARGET[target];
  if (!t) return null;
  const TIcon = t.icon;
  return (
    <span
      className={`inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset ${t.cls}`}
    >
      <TIcon className="h-3 w-3" />
      {t.label}
    </span>
  );
}

function StatusChip({ status }) {
  const { cls, icon: StatusIcon } = statusStyle(status);
  return (
    <span
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium capitalize ring-1 ring-inset transition-colors duration-300 ${cls}`}
    >
      <StatusIcon className="h-3.5 w-3.5" />
      {status}
    </span>
  );
}

/* ---------- One sub-task inside a card ---------- */

function StepRow({ step, now }) {
  const { icon: StepIcon, cls } = statusStyle(step.status);
  return (
    <li className="flex items-start gap-2 text-xs">
      <span className={`mt-0.5 inline-flex shrink-0 rounded-full p-0.5 ring-1 ring-inset ${cls}`}>
        <StepIcon className="h-3 w-3" />
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="font-medium text-slate-800">{step.label}</span>
          <TargetChip target={step.target} />
          <span className="sr-only">{step.status}</span>
          <span className="ml-auto tabular-nums text-slate-500">
            {elapsed(step.started_at, step.finished_at, now)}
          </span>
        </div>
        {step.detail && (
          <p
            className={`mt-0.5 break-words ${
              step.status === "failed" ? "text-red-700" : "text-slate-600"
            }`}
          >
            {step.detail}
          </p>
        )}
      </div>
    </li>
  );
}

/* ---------- One agent card ---------- */

function AgentCard({ agent, now, index }) {
  const meta = AGENTS[agent.name] || { title: agent.name, role: "", icon: IconList };
  const t = tone(agent.status);
  const CardIcon = meta.icon;
  const steps = agent.steps || [];
  const defaultOpen = agent.status === "running" || agent.status === "failed";
  const [userOpen, setUserOpen] = useState(null);
  const open = userOpen ?? defaultOpen;
  const panelId = `steps-${agent.name}`;
  const targets = [...new Set(steps.map((s) => s.target).filter(Boolean))];
  const doneSteps = steps.filter((s) => s.status !== "running").length;

  return (
    <li
      style={{ animationDelay: `${index * 40}ms` }}
      className={`relative animate-rise overflow-hidden rounded-xl border transition-[background-color,border-color,box-shadow] duration-300 ${t.card}`}
    >
      <span
        aria-hidden="true"
        className={`absolute inset-y-0 left-0 w-1 transition-colors duration-300 ${t.bar}`}
      />

      <div className="space-y-3 py-4 pl-5 pr-4">
        <div className="flex items-start gap-3">
          <span
            className={`inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg transition-colors duration-300 ${t.iconWrap}`}
          >
            <CardIcon className="h-5 w-5" />
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-sm font-semibold text-slate-900">{meta.title}</h3>
              <StatusChip status={agent.status} />
            </div>
            <p className="mt-0.5 text-xs text-slate-600">{meta.role}</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
          {targets.map((x) => (
            <TargetChip key={x} target={x} />
          ))}
          {steps.length > 0 && (
            <span className="tabular-nums">
              {doneSteps}/{steps.length} steps
            </span>
          )}
          <span className="ml-auto tabular-nums font-medium text-slate-700">
            {elapsed(agent.started_at, agent.finished_at, now)}
          </span>
        </div>

        {agent.message && (
          <p
            className={`break-words text-xs ${
              agent.status === "failed" ? "text-red-700" : "text-slate-600"
            }`}
          >
            {agent.message}
          </p>
        )}

        {steps.length > 0 && (
          <div>
            <button
              type="button"
              aria-expanded={open}
              aria-controls={panelId}
              onClick={() => setUserOpen(!open)}
              className="inline-flex min-h-[44px] items-center gap-1 rounded-md text-xs font-medium text-slate-700 hover:text-slate-900"
            >
              <IconChevron
                className={`h-4 w-4 transition-transform duration-200 ${open ? "rotate-180" : ""}`}
              />
              {open ? "Hide steps" : "Show steps"}
            </button>
            {open && (
              <ul id={panelId} className="mt-1 space-y-2.5 border-t border-slate-200/80 pt-3">
                {steps.map((s) => (
                  <StepRow key={s.step_id} step={s} now={now} />
                ))}
              </ul>
            )}
          </div>
        )}
      </div>

      {agent.status === "running" && (
        <div
          role="progressbar"
          aria-label={`${meta.title} is running`}
          className="absolute inset-x-0 bottom-0 h-1 overflow-hidden bg-blue-100"
        >
          <span className="absolute inset-y-0 left-0 w-2/5 animate-shimmer rounded-full bg-blue-500" />
        </div>
      )}
    </li>
  );
}

/* ---------- The whole board ---------- */

export default function PipelineBoard({ agents = [], runStatus }) {
  const byName = useMemo(() => Object.fromEntries(agents.map((a) => [a.name, a])), [agents]);

  // Agents the backend reports but the layout does not know land in a last stage.
  const known = new Set(STAGES.flatMap((s) => s.agents));
  const extras = agents.filter((a) => !known.has(a.name)).map((a) => a.name);
  const stages = extras.length ? [...STAGES, { title: "Other", agents: extras }] : STAGES;

  const anyLive = agents.some((a) => isLive(a.status));
  const now = useNow(anyLive);

  const total = agents.length || 1;
  const finished = agents.filter((a) => isDone(a.status) || a.status === "failed").length;
  const pct = Math.round((finished / total) * 100);
  const failed = agents.some((a) => a.status === "failed");

  // Tell screen-reader users when an agent changes state.
  const prev = useRef({});
  const [announcement, setAnnouncement] = useState("");
  useEffect(() => {
    const changes = [];
    for (const a of agents) {
      if (prev.current[a.name] && prev.current[a.name] !== a.status) {
        changes.push(`${AGENTS[a.name]?.title || a.name} ${a.status}`);
      }
      prev.current[a.name] = a.status;
    }
    if (changes.length) setAnnouncement(changes.join(". "));
  }, [agents]);

  return (
    <section aria-labelledby="board-h" className="space-y-5">
      <div className="card p-4 sm:p-5">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 id="board-h" className="text-base font-semibold text-slate-900">
            Pipeline progress
          </h2>
          <p className="text-sm tabular-nums text-slate-600">
            {finished} of {total} agents finished · {pct}%
          </p>
        </div>
        <div
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct}
          aria-label="Overall pipeline progress"
          className="mt-3 h-2 overflow-hidden rounded-full bg-slate-100"
        >
          <div
            style={{ width: `${pct}%` }}
            className={`h-full rounded-full transition-[width,background-color] duration-500 ease-out ${
              failed ? "bg-red-500" : runStatus === "completed" ? "bg-emerald-500" : "bg-blue-500"
            }`}
          />
        </div>
      </div>

      <ol className="space-y-0">
        {stages.map((stage, si) => {
          const members = stage.agents.map((n) => byName[n]).filter(Boolean);
          if (!members.length) return null;
          const state = stageStatus(members);
          const last = si === stages.length - 1;
          return (
            <li key={stage.title} className="flex gap-3 sm:gap-4">
              {/* rail */}
              <div className="flex w-6 shrink-0 flex-col items-center" aria-hidden="true">
                <span className="relative mt-5 flex h-3.5 w-3.5 items-center justify-center">
                  {state === "running" && (
                    <span className="absolute inset-0 animate-ping2 rounded-full bg-blue-400" />
                  )}
                  <span
                    className={`relative h-3.5 w-3.5 rounded-full ring-4 ring-slate-50 transition-colors duration-300 ${tone(state).dot}`}
                  />
                </span>
                {!last && (
                  <span
                    className={`mt-1 w-0.5 flex-1 rounded transition-colors duration-500 ${
                      state === "success" ? "bg-emerald-300" : "bg-slate-200"
                    }`}
                  />
                )}
              </div>

              {/* content */}
              <div className="min-w-0 flex-1 pb-6">
                <p className="mb-2 mt-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Step {si + 1} · {stage.title}
                  {stage.note && (
                    <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-medium normal-case tracking-normal text-slate-600">
                      {stage.note}
                    </span>
                  )}
                </p>
                <ul className={`grid gap-3 ${members.length > 1 ? "md:grid-cols-2" : ""}`}>
                  {members.map((a, i) => (
                    <AgentCard key={a.name} agent={a} now={now} index={i} />
                  ))}
                </ul>
              </div>
            </li>
          );
        })}
      </ol>

      <p className="sr-only" role="status" aria-live="polite">
        {announcement}
      </p>
    </section>
  );
}
